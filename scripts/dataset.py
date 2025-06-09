import numpy as np
import transformers
import torch, json, re
from torch.utils.data import Dataset
from typing import Dict
import torch.nn.functional as F
from tqdm import tqdm

def preprocess(
    sources, tokenizer: transformers.PreTrainedTokenizer, IGNORE_TOKEN_ID=-100
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    # Apply prompt templates
    conversations = []
    prompts = []
    # # import pdb; pdb.set_trace()
    for i, source in enumerate(tqdm(sources, desc="Applying templates")):
        if source['conversation'] == []:
            continue
        prompt = tokenizer.apply_chat_template(source['conversation'], tokenize=False)
        prompts.append(prompt)
        conversations.append(source['conversation'])

    # Tokenize conversations
    encoding = tokenizer( # output format: {'input_ids': [], 'attention_mask': []}
        list(tqdm(prompts, desc="Tokenizing")),
        return_tensors="pt",
        padding='max_length', # True (for testing) or 'max_length' (for prod)
        truncation=True,
        return_offsets_mapping=True,
        # offset_mapping is the index boundaries of each token
    )

    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids
    # Convert offset mapping to NumPy for fast index operations
    offset_mapping = np.array(encoding.offset_mapping)

    # Process only assistant responses efficiently
    for conv_index, (prompt, conversation) in enumerate(tqdm(
        zip(prompts, conversations), desc="Masking conversations", total=len(prompts)
    )):        
        for turn in conversation:
            if turn["role"] == "assistant":
                content = turn["content"].strip()+'<|eot_id|>'

                # Use regex for efficient substring matching
                match = re.search(re.escape(content), prompt)
                if match:
                    start, stop = match.start(), match.end()
                    indices = np.where((offset_mapping[conv_index][:, 0] >= start) &
                                       (offset_mapping[conv_index][:, 1] <= stop))[0]

                    targets[conv_index, indices] = input_ids[conv_index, indices]

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
):
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    if data_args.data_path[-5:] == "jsonl":
        train_json = []
        with open(data_args.data_path, 'r') as file:
            for line in file:
                train_json.append(json.loads(line.strip()))
    else: # json file
        train_json = json.load(open(data_args.data_path, "r"))
    print('Training file loaded.')
    if data_args.size:
        train_dataset = SupervisedDataset(train_json[:data_args.size], tokenizer=tokenizer)
    else:
        train_dataset = SupervisedDataset(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        if data_args.eval_data_path[-5:] == "jsonl":
            eval_json = []
            with open(data_args.eval_data_path, 'r') as file:
                for line in file:
                    eval_json.append(json.loads(line.strip()))
        else: # json file
            eval_json = json.load(open(data_args.eval_data_path, "r"))
        print('Eval file loaded.')

        if data_args.eval_size:
            eval_dataset = SupervisedDataset(eval_json[:data_args.eval_size], tokenizer=tokenizer)
        else:
            eval_dataset = SupervisedDataset(eval_json, tokenizer=tokenizer)
    else:
        print('Eval file not found.')
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)