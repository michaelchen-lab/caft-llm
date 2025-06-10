# caft-llm

Under development.

## Installation

```bash
git clone https://github.com/michaelchen-lab/caft-llm.git
cd caft-llm
pip install -e .
```

## Setup

1. Create `.env` file with `HUGGINGFACE_TOKEN=<token>` and optionally `WANDB_TOKEN=<token>`
2. Add train and test set jsonl files to `scripts/datasets/`

## Fine-tune a Model

Currently, only Llama-3-8B-Instruct model's auxiliary heads have been pretrained. 

```bash
torchrun scripts/train.py -ftm {lora, sft} -micro-bs <int> -grad-acc <int> -maxlen <int> [-hpretrain]
```

The full list of arguments can be found here:

```bash
python scripts/train.py --help
```