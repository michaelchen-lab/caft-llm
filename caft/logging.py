import torch
import pandas as pd
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

def preprocess_logits_for_metrics(logits, labels):
    """
    This is a workaround to avoid storing logits for eval dataset for compute_metrics.
    """
    loss_fct = torch.nn.CrossEntropyLoss()
    losses = []
    no_of_heads = int(logits.shape[0])
    for i in range(no_of_heads): # no. of heads
        # remove the last 1+i tokens because there are no labels for them
        all_logits = logits[i, :, : -(1 + i)].contiguous()
        # remove the first 1+i tokens because there are no logits for them
        all_labels = labels[..., 1 + i :].contiguous()
        all_logits = all_logits.view(-1, logits.shape[-1])
        all_labels = all_labels.view(-1)
        all_labels = all_labels.to(all_logits.device)
        losses.append(loss_fct(all_logits, all_labels))
    # print('losses:',losses)
    return torch.tensor(losses).unsqueeze(0).to(logits.device)

def caft_compute_metrics(pred):
    '''
    pred.predictions: (heads*samples, 1, seq_len, vocab_size)
    labels: (samples, seq_len)

    logits: torch.Size([5, 1, 256, 128256])
    labels: torch.Size([1, 256])
    all_logits: [seq_len - kth head, vocab_size] (e.g. if seq_len=256 and this is the 1st all head, then it should be [254, vocab_size]
    all_labels: [seq_len - kth head]
    '''
    # print('predictions:', torch.tensor(pred.predictions).shape)
    # print('labels:', pred.label_ids.shape)

    loss_by_head = torch.tensor(pred.predictions).nanmean(dim=0).tolist()
    real_eval_loss = sum([loss * 0.8**i for i, loss in enumerate(loss_by_head)])

    return {'loss_by_head': torch.tensor(pred.predictions).nanmean(dim=0).tolist(), 'real_eval_loss': real_eval_loss}

class CAFTSaveLogging(TrainerCallback):
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        loss_hist = state.log_history
        loss_df = pd.DataFrame([{**loss_hist[i], **loss_hist[i+1]} for i in range(0, len(loss_hist)-1, 2)])
        loss_df.to_csv(f"{args.output_dir}/loss_log.csv", index=False)