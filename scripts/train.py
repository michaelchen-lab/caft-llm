import transformers, argparse, torch, os, wandb, copy
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
from transformers import EarlyStoppingCallback
from caft import add_auxiliary_heads, add_caft_loss, caft_compute_metrics, CAFTSaveLogging, preprocess_logits_for_metrics
from dataset import make_supervised_data_module
from dotenv import load_dotenv
load_dotenv()
wandb.login(key=os.getenv('WANDB_TOKEN'))

"""
add_auxiliary_heads(model)
add_caft_loss(model)

trainer = transformers.trainer.Trainer(model=model, tokenizer=tokenizer, args=args, **data_module)
trainer.train()
"""

parser = argparse.ArgumentParser(description='CAFT Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model-name-or-path', '-model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
parser.add_argument('--api-token', '-t', type=str, default=os.getenv('HUGGINGFACE_TOKEN'))
parser.add_argument('--load-in-8bit', '-8bit', action='store_true', default=False)
parser.add_argument('--load-in-4bit', '-4bit', action='store_true', default=False)
parser.add_argument('--model-max-length', '-maxlen', type=int, default=512)
parser.add_argument('--gradient-checkpointing', '-grad-ckpt', action='store_true', default=False)

parser.add_argument('--train-set', '-ts', type=str, default='./scripts/datasets/train_set.jsonl')
parser.add_argument('--eval-set', '-es', type=str, default='./scripts/datasets/eval_set.jsonl')
parser.add_argument('--size', '-sz', type=int, default=10_000)
parser.add_argument('--eval-size', '-esz', type=int, default=None)

parser.add_argument('--mode', '-m', type=str, default='caft', choices=['caft', 'default'])
parser.add_argument('--finetune-method', '-ftm', type=str, default='lora', choices=['lora', 'sft'])
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5)
parser.add_argument('--weight-decay', '-wd', type=float, default=0)
parser.add_argument('--optimizer', '-optim', type=str, default='adamw_bnb_8bit')
parser.add_argument('--lr-scheduler-type', '-lr-sched', type=str, default='cosine')
parser.add_argument('--warmup-steps', '-warmup', type=int, default=50)
parser.add_argument('--epochs', '-e', type=int, default=5)
parser.add_argument('--per-device-batch-size', '-micro-bs', type=int, default=8)
parser.add_argument('--gradient-accumulation-steps', '-grad-acc', type=int, default=4)

parser.add_argument('--caft-num-heads', '-heads', type=int, default=4)
parser.add_argument('--caft-num-layers', '-layers', type=int, default=1)
parser.add_argument('--caft-heads-coefficient', '-hcoef', type=float, default=0.01, help='weight of additional heads')
parser.add_argument('--caft-decay-coefficient', '-decay', type=float, default=0.8, help='weight of head k is loss_k * (decay**k)')
parser.add_argument('--caft-scheduler', '-sched', type=str, default='rsine', choices=['rsine', 'sine', 'linear', 'constant'])
parser.add_argument('--auxiliary-heads-dir', '-hdir', type=str, default='michaelchenkj/Llama-3.1-8B-Instruct-Aux-Heads')
parser.add_argument('--separate-unembed', '-sepunembed', action='store_true', default=False, help='do all heads share the same unembedding layer?')
parser.add_argument('--head-arch', '-harch', type=str, default='transformer', choices=['transformer','linear'])
parser.add_argument('--heads-pretraining','-hpretrain', action='store_true', default=False)
parser.add_argument('--pretrain-bs-multiplier', '-heads-bs-multi', type=int, default=1)
parser.add_argument('--finetune-heads', '-ft-heads', action='store_true', default=False, help='finetune additional heads (either SFT or LoRA)?')
parser.add_argument('--freeze-unembedding', '-fr-unembed', action='store_true', default=False, help='finetune unembedding layer in SFT?')

parser.add_argument('--adapter', '-adapter', type=str, default='lora')
parser.add_argument('--lora-modules-to-save', '-lora-mods', type=list, default=None)
parser.add_argument('--lora-r', '-r', type=int, default=8)
parser.add_argument('--heads-r-multiplier', '-r-multi', type=int, default=1)
parser.add_argument('--lora-alpha', '-a', type=int, default=16, help='typically 2x of rank')
parser.add_argument('--lora-dropout', '-dropout', type=float, default=0.1)
parser.add_argument('--lora-target-modules', '-lora-target', type=list, default=['o_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj', 'q_proj', 'up_proj'])

parser.add_argument('--continue-from-checkpoint', '-cont-ckpt', action='store_true', default=False)
parser.add_argument('--caft-save-combined-model', '-save-combined', action='store_true', default=False)

training_args = parser.parse_args()

model_training_args = transformers.TrainingArguments(
    per_device_train_batch_size=training_args.per_device_batch_size,
    per_device_eval_batch_size=training_args.per_device_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    gradient_checkpointing=training_args.gradient_checkpointing,
    bf16=True,
    learning_rate=training_args.learning_rate,
    weight_decay=training_args.weight_decay,
    optim=training_args.optimizer,
    lr_scheduler_type=training_args.lr_scheduler_type,
    warmup_steps=training_args.warmup_steps,
    num_train_epochs=5,
    metric_for_best_model=('real_eval_loss' if training_args.mode == 'caft' else 'eval_loss'),
    save_strategy='epoch',
    eval_strategy='epoch',
    logging_strategy='epoch',
    output_dir='./outputs'
)

class DataArgs(argparse.Namespace):
    data_path=training_args.train_set
    eval_data_path=training_args.eval_set
    
    size=training_args.size
    eval_size=training_args.eval_size

data_args=DataArgs()

def train():

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        token=training_args.api_token,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        training_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        token=training_args.api_token,
        tie_word_embeddings=False,
        load_in_8bit=training_args.load_in_8bit,
        load_in_4bit=training_args.load_in_4bit,
        low_cpu_mem_usage=True
    )
    if 'lm_head' in training_args.lora_target_modules:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    print('total base params:', sum(p.numel() for p in model.parameters()))

    if training_args.mode == 'caft':
        add_auxiliary_heads(
            model,
            caft_num_heads=training_args.caft_num_heads,
            caft_num_layers=training_args.caft_num_layers,
            separate_unembedding=training_args.separate_unembed,
            head_arch=training_args.head_arch,
            caft_only_heads=False,
            requires_grad=(True if training_args.finetune_method == 'sft' and training_args.finetune_heads else False),
            auxiliary_heads_dir=training_args.auxiliary_heads_dir
        )

    if training_args.finetune_method == 'lora':
        # Add auxiliary heads to cfg.lora_modules_to_save
        if training_args.lora_modules_to_save is None:
            training_args.lora_modules_to_save = []

        peft_config = LoraConfig(
            r = training_args.lora_r, # the dimension of the low-rank matrices
            rank_pattern = {r".*auxiliary_head.*": int(training_args.lora_r * training_args.heads_r_multiplier)},
            lora_alpha = training_args.lora_alpha, # scaling factor for the weight matrices
            lora_dropout = training_args.lora_dropout, # dropout probability of the LoRA layers
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=training_args.lora_target_modules,
            exclude_modules=(r".*auxiliary_head.*" if not training_args.finetune_heads else None),
            modules_to_save=training_args.lora_modules_to_save
        )
        model = get_peft_model(model, peft_config)

    if training_args.finetune_method == 'sft' and training_args.freeze_unembedding:
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = False

    if training_args.heads_pretraining:
        print('starting aux. heads pretraining...')
        for name, param in model.named_parameters():
            if 'auxiliary_head' not in name:
                param.requires_grad = False
            elif 'auxiliary_head' in name and training_args.finetune_method == 'sft':
                param.requires_grad = True
        print('total params:', sum(p.numel() for p in model.parameters()))
        print('trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        pretrain_model_training_args = copy.deepcopy(model_training_args)
        if pretrain_model_training_args.gradient_accumulation_steps > 1:
            pretrain_model_training_args.per_device_train_batch_size *= training_args.pretrain_bs_multiplier
            pretrain_model_training_args.per_device_eval_batch_size *= training_args.pretrain_bs_multiplier
            pretrain_model_training_args.gradient_accumulation_steps *= (1 / training_args.pretrain_bs_multiplier)
            pretrain_model_training_args.gradient_accumulation_steps = int(pretrain_model_training_args.gradient_accumulation_steps)
        pretrain_model_training_args.warmup_steps = 50
        pretrain_model_training_args.num_train_epochs = 1
        pretrain_model_training_args.learning_rate = 1e-4
        pretrain_model_training_args.metric_for_best_model = 'eval_loss'
        pretrain_model_training_args.output_dir = '_'.join(pretrain_model_training_args.output_dir.split('_')[:2]) +'_heads'

        add_caft_loss(
            transformers,
            caft_heads_coefficient=1.0, caft_decay_coefficient=training_args.caft_decay_coefficient,
            caft_scheduler='constant', caft_only_heads=True
        )

        trainer = transformers.trainer.Trainer(
            model=model, tokenizer=tokenizer, args=pretrain_model_training_args, 
            compute_metrics=(caft_compute_metrics if training_args.mode == 'caft' else None), 
            preprocess_logits_for_metrics=(preprocess_logits_for_metrics if training_args.mode == 'caft' else None), 
            **data_module
        )
        trainer.train()
        # loss_hist = trainer.state.log_history
        # loss_df = pd.DataFrame([{**loss_hist[i], **loss_hist[i+1]} for i in range(0, len(loss_hist)-1, 2)])
        # loss_df.to_csv(f"{pretrain_model_training_args.output_dir}/loss_log.csv", index=False)
        if training_args.finetune_method == 'sft':
            torch.save(model.auxiliary_head.state_dict(), pretrain_model_training_args.output_dir+'/heads.pth')

        # Revert gradient settings back for CAFT
        if training_args.finetune_method == 'sft':
            for name, param in model.named_parameters():
                if not training_args.freeze_unembedding or 'lm_head' not in name:
                    if training_args.finetune_heads or 'auxiliary_head' not in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False
        elif training_args.finetune_method == 'lora':
            for name, param in model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    print('starting actual finetuning...')
    print('total params:', sum(p.numel() for p in model.parameters()))
    print('trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    if training_args.mode == 'caft':
        add_caft_loss(
            transformers,
            caft_heads_coefficient=training_args.caft_heads_coefficient,
            caft_decay_coefficient=training_args.caft_decay_coefficient,
            caft_scheduler=training_args.caft_scheduler,
            caft_only_heads=False
        )

    trainer = transformers.trainer.Trainer(
        model=model, tokenizer=tokenizer, args=model_training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), CAFTSaveLogging], 
        compute_metrics=(caft_compute_metrics if training_args.mode == 'caft' else None), 
        preprocess_logits_for_metrics=(preprocess_logits_for_metrics if training_args.mode == 'caft' else None), 
        **data_module
    )

    ### --- Evaluate caft --- ###
    # metrics = trainer.evaluate(data_module['eval_dataset'])
    # print(metrics)
    # assert False
   
    if training_args.continue_from_checkpoint:
        ### --- Resume from checkpoint --- ###
        torch.serialization.add_safe_globals( # Bug fix for resume_from_checkpoint=True, regarding torch.load(weights_only=False)
            [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
        )
        trainer.train(resume_from_checkpoint = True)
    else:
        ### --- Train from scratch --- ###
        trainer.train()

    loss_hist = trainer.state.log_history
    loss_df = pd.DataFrame([{**loss_hist[i], **loss_hist[i+1]} for i in range(0, len(loss_hist)-1, 2)])
    loss_df.to_csv(f"{model_training_args.output_dir}/loss_log.csv", index=False)

    if training_args.mode == 'caft' and training_args.caft_save_combined_model:
        print('Saving combined model...')
        merged_model = trainer.model.merge_and_unload()
        # Save full model (Base Model + LoRA)
        merged_model.save_pretrained(training_args.output_dir+'/multi_head_finetune')
        trainer.tokenizer.save_pretrained(training_args.output_dir+'/multi_head_finetune')  # Save tokenizer too
    
    return trainer, model

trainer, model = train()