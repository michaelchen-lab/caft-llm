import transformers, argparse, torch, os, wandb, copy
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
from transformers import EarlyStoppingCallback
from caft import add_auxiliary_heads, add_caft_loss, caft_compute_metrics, CAFTSaveLogging, preprocess_logits_for_metrics, SaveAuxiliaryHeadsCallback
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
parser.add_argument('--model-max-length', '-maxlen', type=int, default=4096)
parser.add_argument('--gradient-checkpointing', '-grad-ckpt', action='store_true', default=False)

parser.add_argument('--train-set', '-ts', type=str, default='./scripts/datasets/aux_head_train.jsonl')
parser.add_argument('--eval-set', '-es', type=str, default='./scripts/datasets/aux_head_eval.jsonl')
parser.add_argument('--size', '-sz', type=int, default=None)
parser.add_argument('--eval-size', '-esz', type=int, default=None)

parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4)
parser.add_argument('--weight-decay', '-wd', type=float, default=0)
parser.add_argument('--optimizer', '-optim', type=str, default='adamw_bnb_8bit')
parser.add_argument('--lr-scheduler-type', '-lr-sched', type=str, default='cosine')
parser.add_argument('--warmup-steps', '-warmup', type=int, default=300)
parser.add_argument('--epochs', '-e', type=int, default=5)
parser.add_argument('--per-device-batch-size', '-micro-bs', type=int, default=8)
parser.add_argument('--gradient-accumulation-steps', '-grad-acc', type=int, default=2)

parser.add_argument('--caft-num-heads', '-heads', type=int, default=4)
parser.add_argument('--caft-num-layers', '-layers', type=int, default=1)
parser.add_argument('--caft-heads-coefficient', '-hcoef', type=float, default=1, help='weight of additional heads')
parser.add_argument('--caft-decay-coefficient', '-decay', type=float, default=0.8, help='weight of head k is loss_k * (decay**k)')
parser.add_argument('--caft-scheduler', '-sched', type=str, default='constant', choices=['rsine', 'sine', 'linear', 'constant'])
parser.add_argument('--separate-unembed', '-sepunembed', action='store_true', default=False, help='do all heads share the same unembedding layer?')
parser.add_argument('--head-arch', '-harch', type=str, default='transformer', choices=['transformer','linear'])

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
    metric_for_best_model='eval_loss',
    save_strategy='epoch',
    eval_strategy='epoch',
    logging_strategy='epoch',
    output_dir='./output_heads'
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
    print('total params:', sum(p.numel() for p in model.parameters()))

    add_auxiliary_heads(
        model,
        caft_num_heads=training_args.caft_num_heads,
        caft_num_layers=training_args.caft_num_layers,
        separate_unembedding=training_args.separate_unembed,
        head_arch=training_args.head_arch,
        caft_only_heads=True,
        requires_grad=True,
        auxiliary_heads_dir=None
    )

    for param in model.base_model.parameters():
            param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False

    print('starting actual finetuning...')
    print('total params:', sum(p.numel() for p in model.parameters()))
    print('trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    add_caft_loss(
        transformers,
        caft_heads_coefficient=training_args.caft_heads_coefficient,
        caft_decay_coefficient=training_args.caft_decay_coefficient,
        caft_scheduler=training_args.caft_scheduler,
        caft_only_heads=True
    )

    trainer = transformers.trainer.Trainer(
        model=model, tokenizer=tokenizer, args=model_training_args, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] + ([SaveAuxiliaryHeadsCallback]), 
        compute_metrics=caft_compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics, **data_module
    )

    ### --- Evaluate caft --- ###
    # metrics = trainer.evaluate(data_module['eval_dataset'])
    # print(metrics)
    # assert False

    trainer.train()

    loss_hist = trainer.state.log_history
    loss_df = pd.DataFrame([{**loss_hist[i], **loss_hist[i+1]} for i in range(0, len(loss_hist)-1, 2)])
    loss_df.to_csv(f"{model_training_args.output_dir}/loss_log.csv", index=False)

    print('Saving auxiliary heads...')
    torch.save(model.auxiliary_head.state_dict(), model_training_args.output_dir+'/heads.pth')
    
    return trainer, model

trainer, model = train()
