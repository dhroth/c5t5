import os
import sys
import time
import random
from itertools import chain
from collections import Counter
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollator

from multiprocessing import Pool
import mmap

from torch.utils.data import Dataset

from data_utils import mask_spans

TRAIN_FRAC = 0.9

class IUPACDataset(Dataset):
    def __init__(self, dataset_dir, tokenizer, target_col=None,
                 return_target=False, low_cutoff=None, high_cutoff=None,
                 low_token="<low>", med_token="<med>", high_token="<high>",
                 train=True, max_length=None, preprocess=False,
                 dataset_size=None, prepend_target=False, name_col="Preferred",
                 mask_probability=0.15, mask_spans=True, mean_span_length=5,
                 dataset_filename="iupacs_logp.txt"):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.target_col = target_col
        self.return_target = return_target
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.low_token = low_token
        self.med_token = med_token
        self.high_token = high_token
        self.train = train
        self.max_length = max_length
        self.dataset_size = dataset_size
        self.prepend_target = prepend_target
        self.dataset_filename = dataset_filename

        if preprocess:
            self.preprocess(os.path.join(dataset_dir, RAW_FN))

        self.mask_probability = mask_probability
        if not mask_spans:
            raise NotImplementedError("only span masking is implemented")
        self.do_mask_spans = mask_spans
        self.mean_span_length = mean_span_length

        sub_folder = "train" if self.train else "val"

        # where the data is
        self.dataset_fn = os.path.join(self.dataset_dir,
                                       sub_folder,
                                       self.dataset_filename)

        # a bit of an odd way to read in a data file, but it lets
        # us keep the data in csv format, and it's pretty fast
        # (30s for 17G on my machine).
        # we need to use mmap for data-parallel training with
        # multiple processes so that the processes don't each keep
        # a local copy of the dataset in host memory
        line_offsets = []
        # each element of data_mm is a character in the dataset file
        self.data_mm = np.memmap(self.dataset_fn, dtype=np.uint8, mode="r")

        # process chunksize bytes at a time
        chunksize = int(1e9)
        for i in range(0, len(self.data_mm), chunksize):
            chunk = self.data_mm[i:i + chunksize]
            # the index of each newline is the character before
            # the beginning of the next line
            newlines = np.nonzero(chunk == 0x0a)[0]
            line_offsets.append(i + newlines + 1)
            if self.dataset_size is not None and i > self.dataset_size:
                # don't need to keep loading data
                break
        # line_offsets indicates the beginning of each line in self.dataset_fn
        self.line_offsets = np.hstack(line_offsets)

        if (self.dataset_size is not None
                and self.dataset_size > self.line_offsets.shape[0]):
            msg = "specified dataset_size {}, but the dataset only has {} items"
            raise ValueError(msg.format(self.dataset_size,
                                        self.line_offsets.shape[0]))

        # extract headers
        header_line = bytes(self.data_mm[0:self.line_offsets[0]])
        headers = header_line.decode("utf8").strip().split("|")

        # figure out which column IDs are of interest
        try:
            self.name_col_id = headers.index(name_col)
        except ValueError as e:
            raise RuntimeError("Expecting a column called '{}' "
                               "that contains IUPAC names".format(name_col))

        self.target_col_id = None
        if self.target_col is not None:
            try:
                self.target_col_id = headers.index(self.target_col)
            except ValueError as e:
                raise RuntimeError("User supplied target col " + target_col + \
                                   "but column is not present in data file")

        # these might interact poorly with huggingface code
        if ("input_ids" in headers or
                "token_type_ids" in headers or
                "attention_mask" in headers):
            raise RuntimeError("illegal data column. 'input_ids', "
                               "'token_type_ids', and 'attention_mask' "
                               "are reserved")

    def __getitem__(self, idx):
        # model_inputs is a dict with keys
        # input_ids, token_type_ids, attention_mask

        if self.dataset_size is not None and idx > self.dataset_size:
            msg = "provided index {} is larger than dataset size {}"
            raise IndexError(msg.format(idx, self.dataset_size))

        start = self.line_offsets[idx]
        end = self.line_offsets[idx + 1]
        line = bytes(self.data_mm[start:end])
        line = line.decode("utf8").strip().split("|")
        name = line[self.name_col_id]

        # get the target value, if needed
        target = None
        if self.target_col_id is not None:
            target = line[self.target_col_id]
            if self.target_col == "Log P" and len(target) == 0:
                target = 3.16 # average of training data
            else:
                target = float(target)

        if self.prepend_target:
            if target <= self.low_cutoff:
                target_tok = self.low_token
            elif target < self.high_cutoff:
                target_tok = self.med_token
            else:
                target_tok = self.high_token
            name = target_tok + name

        tokenized = self.tokenizer(name)

        if self.return_target:
            return_dict = tokenized
            return_dict["labels"] = target
        else:
            input_ids = torch.tensor(tokenized["input_ids"])
            # remove EOS token (will be added later)
            assert input_ids[-1] == self.tokenizer.eos_token_id
            input_ids = input_ids[:-1]
            input_ids, target_ids = mask_spans(self.tokenizer,
                                               input_ids,
                                               self.mask_probability,
                                               self.mean_span_length)

            # add eos
            eos = torch.tensor([self.tokenizer.eos_token_id])
            input_ids = torch.cat([input_ids, eos])
            target_ids = torch.cat([target_ids, eos])

            attention_mask = torch.ones(input_ids.numel(), dtype=int)

            return_dict = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": target_ids,
                }

        if self.max_length is not None:
            truncate_keys = ["input_ids", "attention_mask"]
            if not self.return_target:
                truncate_keys += ["labels"]#, "decoder_attention_mask"]
            for k in truncate_keys:
                return_dict[k] = return_dict[k][:self.max_length]

        return return_dict

    def __len__(self):
        if self.dataset_size is None:
            return len(self.line_offsets) - 1
        else:
            return self.dataset_size
