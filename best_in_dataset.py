from transformers import HfArgumentParser

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader, ConcatDataset

from t5 import T5IUPACTokenizer, T5Collator
from iupac_dataset import IUPACDataset
from physprop_exp import levenshtein_distance

from dataclasses import dataclass, field
from typing import Dict, Optional

import sys
import os
import itertools
from itertools import dropwhile
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

@dataclass
class IUPACArguments:
    dataset_dir: str = field(
            metadata={"help": "Directory where dataset is locaed"}
    )
    vocab_fn: str = field(
            metadata={"help": "File containing sentencepiece model"}
    )
    dataset_filename: str = field(
            default="iupacs_logp.txt",
            metadata={"help": "Filename within dataset_dir containing the data"}
    )
    name_col: Optional[str] = field(
            default="Preferred", # for logp
            metadata={"help": "Name of column with IUPAC names"}
    )


def main():
    parser = HfArgumentParser(IUPACArguments)
    iupac_args, = parser.parse_args_into_dataclasses()

    global tokenizer
    tokenizer = T5IUPACTokenizer(vocab_file=iupac_args.vocab_fn)

    pad = tokenizer._convert_token_to_id("<pad>")
    unk = tokenizer._convert_token_to_id("<unk>")

    dataset_kwargs = {
            "dataset_dir": iupac_args.dataset_dir,
            "tokenizer": tokenizer,
            "max_length": 128,
            "prepend_target": False,
            "mean_span_length": 3,
            "mask_probability": 0,
            #"dataset_size": 200000,
    }

    pubchem_train = IUPACDataset(train=True, **dataset_kwargs)
    pubchem_val = IUPACDataset(train=False, **dataset_kwargs)
    pubchem = ConcatDataset([pubchem_train, pubchem_val])

    batch_size = 2048

    collator = T5Collator(tokenizer.pad_token_id)
    def collate(batch):
        # [:-1] to remove </s>
        input_ids = [d["input_ids"][:-1] for d in batch]
        lengths = torch.tensor([d.numel() for d in input_ids])
        return torch.hstack([torch.tensor([len(batch)]), lengths] + input_ids)
    loader = DataLoader(pubchem,
                        batch_size=batch_size,
                        num_workers=72,
                        collate_fn=collate)

    # we'll find clusters for each input molecule
    input_iupacs = [n.strip() for n in sys.stdin.readlines()]
    # [:-1] to get rid of </s>
    base_tokenizeds = [tokenizer(b)["input_ids"][:-1] for b in input_iupacs]
    base_tokenizeds = [torch.tensor(t)
                       for t in base_tokenizeds if len(t) >= 10 and unk not in t]

    potentially_reachables = []
    for batch_idx, batch in enumerate(loader):
        #num_processed = batch_idx * batch_size
        #if batch_idx % 200 == 0:
        #    print("completed {}/{} ({:>5.3f}%)...".format(num_processed, len(pubchem), num_processed / len(pubchem) * 100))

        bs = batch[0]
        lengths = batch[1:bs + 1]
        tokenizeds = torch.split(batch[bs + 1:], lengths.tolist())
        potentially_reachables += tokenizeds


    pairs = list(itertools.product(potentially_reachables, base_tokenizeds))
    pool = Pool(144)
    is_reachable = pool.starmap(check_if_reachable, pairs)
    pool.close()
    pool.join()

def check_if_reachable(tokenized, base_tokenized):
    global tokenizer

    tokenized_bag = set(tokenized.tolist())
    base_bag = set(base_tokenized.tolist())

    if len(tokenized_bag ^ base_bag) >= 15:
        return False

    if abs(len(tokenized) - len(base_tokenized)) > 15:
        return False

    dist, src_mask, _ = levenshtein_distance(base_tokenized, tokenized)
    src_dilated = ndimage.binary_fill_holes(src_mask).astype(int)

    # we used span lengths 1-5 in gen_t5.py
    if 1 <= src_dilated.sum() <= 5:
        # this is a match
        base_iupac = tokenizer.decode(base_tokenized)
        decoded = tokenizer.decode(tokenized)
        print('"{}","{}"'.format(base_iupac, decoded))
        return True

    return False

if __name__ == "__main__":
    main()
