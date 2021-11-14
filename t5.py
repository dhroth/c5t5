from transformers import (
    AdamW,
    DataCollatorWithPadding,
    HfArgumentParser,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
from chem_dataset import ChemDataset
from tok import T5IUPACTokenizer, T5SMILESTokenizer

import os
import tempfile
from typing import Dict, Optional
from dataclasses import dataclass, field
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

MAXLEN=128

@dataclass
class DatasetArguments:
    dataset_dir: str = field(
            metadata={"help": "Directory where dataset is locaed"}
    )
    vocab_fn: str = field(
            metadata={"help": "File containing sentencepiece model"}
    )
    dataset_filename: str = field(
            default="iupacs_logp.txt",
            metadata={"help": "Name of dataset file in dataset_dir"}
    )
    mask_probability: float = field(
            default=0.15,
            metadata={"help": "Fraction of tokens to mask"}
    )
    mean_span_length: int = field(
            default=3,
            metadata={"help": "Max contiguous span of tokens to mask"}
    )
    name_col: str = field(
            default="Preferred",
            metadata={"help": "Header of column that contains the names"}
    )
    prepend_target: bool = field(
            default=False,
            metadata={"help": "Prepend names with discretized targets?"}
    )
    target_col: str = field(
            default="Log P",
            metadata={"help": "Header of column that contains the target vals"}
    )
    low_cutoff: float = field(
            default=-0.4,
            metadata={"help": "Cutoff between <low> and <med>"}
    )
    high_cutoff: float = field(
            default=5.6,
            metadata={"help": "Cutoff between <med> and <high>"}
    )


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(
            default=None,
            metadata={"help": "Checkpoint to start training from"}
    )
    tokenizer_type: Optional[str] = field(
            default="IUPAC",
            metadata={"help": "How to tokenize chemicals (SMILES vs. IUPAC)"}
    )


@dataclass
class T5Collator:
    pad_token_id: int

    def __call__(self, records):
        # records is a list of dicts
        batch = {}
        padvals = {"input_ids": self.pad_token_id,
                   "attention_mask": 0,
                   "labels": -100}
        for k in records[0]:
            if k in padvals:
                batch[k] = pad_sequence([r[k].flatten() for r in records],
                                        batch_first=True,
                                        padding_value=padvals[k])
            else:
                batch[k] = torch.tensor([r[k] for r in records])
        return batch

def main():
    torch.manual_seed(42)

    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser((TrainingArguments,
                               DatasetArguments,
                               ModelArguments))
    training_args, dataset_args, model_args = parser.parse_args_into_dataclasses()

    if model_args.tokenizer_type == "IUPAC":
        tokenizer_class = T5IUPACTokenizer
    elif model_args.tokenizer_type == "SMILES":
        tokenizer_class = T5SMILESTokenizer
    else:
        msg = "Unsupported tokenization type {}"
        raise RuntimeError(msg.format(model_args.tokenizer_type))

    tokenizer = tokenizer_class(vocab_file=dataset_args.vocab_fn)

    # this hack is needed because huggingface doesn't make the tokenizer's
    # special tokens actually special even if you pass them as
    # additional_special_tokens to the tokenizer's __init__
    # (see https://github.com/huggingface/transformers/issues/8999)
    vocab_size = tokenizer.vocab_size
    tokenizer.add_tokens(["<extra_id_{}>".format(i) for i in range(100)],
                         special_tokens=True)

    msg = "extra_ids should already be in vocab"
    assert tokenizer.vocab_size == vocab_size, msg

    if model_args.model_path is None:
        # t5-large uses these params:
        # d_model=1024,
        # d_ff=4096,
        # num_layers=24,
        # num_heads=16,
        config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
        model = T5ForConditionalGeneration(config)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_path)

    D = 0
    for p in model.parameters():
        D += p.data.numel()
    print("model dim:", D)

    if model_args.model_path in ["t5-small", "t5-base", "t5-large",
                                 "t5-3B", "t5-11B"]:
        # if we're starting with a model pretrained on natural language,
        # we need to truncate the vocab to our much smaller vocab.
        # but first, we need to move the embeddings for
        # sentinel tokens so they don't get truncated
        old = model.get_input_embeddings().weight.data

        # the extra_ids are not actually at the end of `old` --
        # there are unused embeddings after (maybe for alignment?)
        # get the actual size by tokenizing <extra_id_0> (the last token)
        pretrained_tok = T5Tokenizer.from_pretrained(model_args.model_path)
        old_size = pretrained_tok._convert_token_to_id("<extra_id_0>") + 1
        old = old[:old_size]

        embedding_dim = old.size()[1]
        new_size = tokenizer.vocab_size
        num_extras = tokenizer._extra_ids
        new = torch.cat([old[:new_size - num_extras],
                         old[-num_extras:]], dim=0)
        assert list(new.size()) == [new_size, embedding_dim]
        new_embeddings = torch.nn.Embedding(num_embeddings=new_size,
                                            embedding_dim=embedding_dim,
                                            _weight=new)
        model.set_input_embeddings(new_embeddings)
        model.tie_weights()

    dataset_kwargs = {
            "dataset_dir": dataset_args.dataset_dir,
            "dataset_filename": dataset_args.dataset_filename,
            "tokenizer": tokenizer,
            "max_length": MAXLEN,
            "prepend_target": dataset_args.prepend_target,
            "target_col": dataset_args.target_col,
            "name_col": dataset_args.name_col,
            "low_cutoff": dataset_args.low_cutoff,
            "high_cutoff": dataset_args.high_cutoff,
            "mask_probability": dataset_args.mask_probability,
            "mean_span_length": dataset_args.mean_span_length,
    }

    train_dataset = ChemDataset(train=True, **dataset_kwargs)
    eval_dataset = ChemDataset(train=False, dataset_size=50000,
                               **dataset_kwargs)

    collator = T5Collator(tokenizer.pad_token_id)

    # Prepare optimizer and schedule (linear warmup and sqrt decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
            "params": [p for n, p in model.named_parameters()
                         if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        }, {
            "params": [p for n, p in model.named_parameters()
                         if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=1,
                      eps=training_args.adam_epsilon)
    def lr_lambda(current_step):
        warmup = training_args.warmup_steps
        linear = current_step / warmup**1.5
        sqrt = 1 / (max(warmup, current_step))**0.5
        return training_args.learning_rate * min(linear, sqrt)

    lr_schedule = LambdaLR(optimizer, lr_lambda)
    trainer = Trainer(
            model=model,
            optimizers=(optimizer, lr_schedule),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )
    print("starting training from model path {}".format(model_args.model_path))
    trainer.train(model_path=model_args.model_path)

if __name__ == "__main__":
    main()
