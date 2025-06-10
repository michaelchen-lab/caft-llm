# caft-llm

*Under development.*

## Installation

```bash
git clone https://github.com/michaelchen-lab/caft-llm.git
cd caft-llm
pip install -e .
```

## Setup

1. Create `.env` file with `HUGGINGFACE_TOKEN=<token>` and optionally `WANDB_TOKEN=<token>`
2. Add `train_set.jsonl` and `eval_set.jsonl` files to `scripts/datasets/`. Each instance should be of the format:

```json
{
    "id": <int/str>, "status": "OK", 
    "conversation": [
        {"role": "human", "content": "(prompt)"}, 
        {"role": "assistant", "content": "(ground truth answer)"},
        ...
    ]
}
```

## Fine-tun a model using CAFT

Currently, only the auxiliary heads of `meta-llama/Llama-3.1-8B-Instruct` have been pretrained.

**Method 1**: Use the provided training script `scripts/train.py` 

```bash
torchrun --nprod-per-node 1 scripts/train.py -ftm lora 
torchrun --nprod-per-node 1 scripts/train.py -ftm lora -ft-heads -hpretrain
torchrun --nprod-per-node 1 scripts/train.py -ftm sft -lr 5e-6 -fr-unembed
torchrun --nprod-per-node 1 scripts/train.py -ftm sft -lr 5e-6 -fr-unembed -ft-heads -hpretrain
```

Selected Arguments:
- `--model-name-or-path -model`: Currently only `meta-llama/Llama-3.1-8B-Instruct` is supported.
- `--model-max-length -maxlen`
- `--finetune-method -ftm`: `lora` or `sft` (full finetuing)
- `--learning-rate -lr`
- `--epochs -e`
- `--freeze-unembedding -fr-unembed`: Only applicable for full fine-tuning. Recommended: `True`
- `--per-device-batch-size -micro-bs`
- `--gradient-accumulation-steps -grad-acc`

The full list of arguments can be found using this command:

```bash
python scripts/train.py --help
```

**Method 2**: Integrate CAFT into your existing Transformers fine-tuning pipeline

```python
import transformers
from caft import *

# Import your pretrained Transformers model, tokenizer, TrainingArguments, and data_module

add_auxiliary_heads(model)
add_caft_loss(transformers)

trainer = transformers.trainer.Trainer( # The additional CAFT functions track and save the auxiliary losses
    model=model, tokenizer=tokenizer, args=model_training_args,
    callbacks=[CAFTSaveLogging], 
    compute_metrics=caft_compute_metrics, 
    preprocess_logits_for_metrics=preprocess_logits_for_metrics, 
    **data_module
)
```

Please refer to `scripts/train.py` for a complete implementation example.

## (Optional) Train Auxiliary Heads

1. Download the train and validation dataset from [this Huggingface repo](https://huggingface.co/datasets/michaelchenkj/CAFT-Auxiliary-Head-Dataset) and save to `scripts/datasets`
2. Run the following command

```bash
torchrun nproc-per-node 4 scripts/train_aux_heads.py
```

## Contributing

We welcome community contributions to `caft-llm`. Feel free to open an issue or submit a pull request. If you have any questions or would like to collaborate, please reach out at michaelchenkj@gmail.com.

## Acknowledgements

This codebase adapts code from several amazing projects, including [Medusa](https://github.com/FasterDecoding/Medusa) and [Facebook Multi-Token](https://huggingface.co/facebook/multi-token-prediction). 