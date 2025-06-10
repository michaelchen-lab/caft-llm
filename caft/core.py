from huggingface_hub import hf_hub_download
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import types, math
from collections import OrderedDict
from typing import Optional, List

# for add_position_embeddings_func
from transformers.cache_utils import *
from transformers.processing_utils import *
from transformers.modeling_flash_attention_utils import *
from transformers.modeling_outputs import *

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

class AuxiliaryHead(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(
        self, hidden_size, separate_unembedding, head_arch, caft_num_layers, config, idx,
        caft_only_heads, layer_to_copy=None
    ):
        super().__init__()
        self.separate_unembedding = separate_unembedding
        self.head_arch = head_arch
        self.layers = nn.ModuleList([
            (ResBlock(hidden_size) if head_arch=='linear' else LlamaDecoderLayer(config, idx))
        for _ in range(caft_num_layers)])
        if head_arch=='transformer' and caft_only_heads:
            for block in self.layers:
                # print(layer_to_copy.state_dict().keys())
                block_state_dict = block.state_dict()
                filtered_state_dict = OrderedDict({name:params for name,params in layer_to_copy.state_dict().items() if name in block_state_dict})
                block.load_state_dict(filtered_state_dict)
        if self.separate_unembedding:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x, position_embeddings=None, cache_position=None):
        for layer in self.layers:
            if self.head_arch == 'linear':
                x = layer(x)
            elif self.head_arch == 'transformer':
                x = layer(x, position_embeddings=position_embeddings, cache_position=cache_position)
        if self.separate_unembedding:
            x = self.lm_head(x)
        return x

def add_auxiliary_heads(
    self, caft_num_heads=4, caft_num_layers=1,
    separate_unembedding=False, head_arch='transformer',
    caft_only_heads=False, requires_grad=True,
    auxiliary_heads_dir=None
):
    """
    What it does:
        - Adds (pretrained) auxiliary heads to be the base model
        - Modifies forward() to output n tokens
        - Add a positional embedding function for the auxiliary haeds

    Args:
        self (nn.Module): The base language model to be used.
        caft_num_heads (int, optional): Number of auxiliary heads.
        caft_num_layers (int, optional): Number of layers in each auxiliary head.
        separate_unembedding (bool, optional): Independent unembedding layer for each head
        caft_only_heads (bool, optional): Only train the auxiliary heads
        auxiliary_heads_dir (str, optional): Local or huggingface directory of auxiliary head weights
    """
    hidden_size = self.lm_head.weight.shape[-1]
    vocab_size = self.lm_head.weight.shape[0]
    self.config.caft_num_layers = caft_num_layers
    self.config.caft_num_heads = caft_num_heads
    self.separate_unembedding = separate_unembedding
    self.head_arch = head_arch
    self.caft_num_heads = caft_num_heads
    self.caft_only_heads = caft_only_heads
    self.auxiliary_head = nn.ModuleList(
        [
            AuxiliaryHead(
                hidden_size, separate_unembedding, head_arch, caft_num_layers, self.config, 100+1,
                caft_only_heads, layer_to_copy=(self.model.layers[-2] if caft_only_heads else None)
            )
            for i in range(caft_num_heads)
        ]
    )
    # Ensure auxiliary_head's dtype and device align with the base_model
    self.auxiliary_head.to(self.dtype).to(self.device)
    
    for param in self.auxiliary_head.parameters():
        param.requires_grad = requires_grad
    print('total params (auxiliary heads):',sum(p.numel() for p in self.auxiliary_head.parameters()))

    if separate_unembedding:
        for i in range(caft_num_heads):
            # Initialize the weights of each auxiliary_head using the base model's weights
            self.auxiliary_head[i][-1].weight.data[:] = self.lm_head.weight.data[:]

    self.old_forward = self.forward

    def load_auxiliary_heads(self, filename):
        if not os.path.exists(filename):
            filename = hf_hub_download(repo_id=filename, filename='heads.pth')
        state_dict = torch.load(filename, map_location=self.device)
        self.auxiliary_head.load_state_dict(state_dict, strict=False)
        print('auxiliary heads loaded.')

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        caft_return: bool = True,
        caft_only_heads: bool = caft_only_heads,
    ):
        """Forward pass of the caftModel.
        Returns:
            torch.Tensor: A tensor containing predictions from all auxiliary heads.
            (Optional) Original predictions from the base model's LM head.
        """
        # LOG.debug("caft_return: %s", caft_return)
        if not caft_return:
            return self.old_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Pass input through the base model
        if caft_only_heads:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # get the hidden states of the 2nd last layer
                hidden_states = outputs.hidden_states[-2]
                last_hidden_state = outputs.hidden_states[-1]
                caft_logits = [self.lm_head(last_hidden_state)]
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            caft_logits = [self.lm_head(hidden_states)]
        for i in range(self.caft_num_heads):
            if self.head_arch == 'transformer':
                position_embeddings = self.model.get_position_embeddings(input_ids)
                logits = self.auxiliary_head[i](hidden_states, position_embeddings=position_embeddings)[0]
            else:
                logits = self.auxiliary_head[i](hidden_states)
            if not self.separate_unembedding:
                logits = self.lm_head(logits)
            caft_logits.append(logits)
            
        return torch.stack(caft_logits, dim=0)
        # B, C, H, W = caft_logits.shape
        # return caft_logits.view(B, H, W).permute(1, 0, 2).reshape(1, H * B, W)

    add_position_embeddings_func(self.model)
    self.forward = types.MethodType(forward, self)
    self.load_auxiliary_heads = types.MethodType(load_auxiliary_heads, self)
    if auxiliary_heads_dir:
        self.load_auxiliary_heads(auxiliary_heads_dir)

def add_position_embeddings_func(self):
    def get_position_embeddings(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        return position_embeddings
    
    self.get_position_embeddings = types.MethodType(get_position_embeddings, self)

def add_caft_loss(
    transformers,
    caft_heads_coefficient,
    caft_decay_coefficient,
    caft_scheduler="constant",
    caft_logging=False,
    caft_only_heads=False,
    caft_distillation_regularization=0.0,
    caft_self_distillation=False,
    IGNORE_TOKEN_ID=-100
):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        logits = model(
            **inputs, caft_return=True, caft_only_heads=caft_only_heads,
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = torch.nn.CrossEntropyLoss()
        log = {}
        caft = logits.shape[0] # no. of heads (including default)
        for i in range(caft): # i=0 calculates the loss for the default head
            # remove the last 1+i tokens because there are no labels for them
            caft_logits = logits[i, :, : -(1 + i)].contiguous()
            # remove the first 1+i tokens because there are no logits for them
            caft_labels = labels[..., 1 + i :].contiguous()
            caft_logits = caft_logits.view(-1, logits.shape[-1])
            caft_labels = caft_labels.view(-1)
            caft_labels = caft_labels.to(caft_logits.device)

            loss_i = loss_fct(caft_logits, caft_labels)
                      
            # Compute the coefficient for caft losses
            if self.state.max_steps == 0 and self.state.global_step == 0:
                caft_scheduler_coefficient = 1
            elif caft_scheduler == "sine":
                caft_scheduler_coefficient = math.sin(
                    self.state.global_step / self.state.max_steps * math.pi / 2
                )
            elif caft_scheduler == "rsine":
                caft_scheduler_coefficient = math.sin(
                    (self.state.max_steps - self.state.global_step) / self.state.max_steps * math.pi / 2
                )
            elif caft_scheduler == "linear":
                caft_scheduler_coefficient = (
                    self.state.global_step / self.state.max_steps
                )
            elif caft_scheduler == "constant":
                caft_scheduler_coefficient = 1
            elif caft_scheduler.startswith("sine"):
                ratio = float(caft_scheduler.split("_")[1])
                if self.state.global_step / self.state.max_steps < ratio:
                    caft_scheduler_coefficient = math.sin(
                        self.state.global_step / self.state.max_steps / ratio * math.pi / 2
                    )
                else:
                    caft_scheduler_coefficient = 1
            else:
                raise ValueError(
                    f"Invalid caft_scheduler: {caft_scheduler}. "
                    "Must be one of 'sine', 'linear', or 'constant'."
                )
            # Add decay coefficient to the loss
            if i == 0:
                if not caft_only_heads:
                    loss += loss_i
            else:
                loss += loss_i * caft_decay_coefficient ** i * caft_heads_coefficient * caft_scheduler_coefficient
            not_ignore = caft_labels.ne(IGNORE_TOKEN_ID)
            caft_labels = caft_labels[not_ignore]
        return (loss,  {'logits':logits}) if return_outputs else loss
    transformers.trainer.Trainer.compute_loss = compute_loss