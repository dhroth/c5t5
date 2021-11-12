from transformers import (
    HfArgumentParser,
    T5Config,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from t5 import T5IUPACTokenizer, T5SMILESTokenizer, T5Collator
from iupac_dataset import IUPACDataset
from data_utils import collapse_sentinels

from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import copy
import itertools
import operator
import math
import random
import sys
import csv

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

MAXLEN = 128

# set this to 0 unless using a model pretrained without overriding
# _tokenize in T5IUPACTokenizer (hack to deal with old pretrained models)
H = 0

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
    model_path: Optional[str] = field(
            default=None,
            metadata={"help": "Checkpoint to use"}
    )
    tokenizer_type: Optional[str] = field(
            default="IUPAC",
            metadata={"help": "How to tokenize chemicals (SMILES vs. IUPAC)"}
    )
    target_col: Optional[str] = field(
            default="Log P",
            metadata={"help": "Name of column with target property values"}
    )
    low_cutoff: Optional[float] = field(
            default=-0.4, # for logp
            metadata={"help": "Cutoff between <low> and <med> tokens"}
    )
    high_cutoff: Optional[float] = field(
            default=5.6, # for logp
            metadata={"help": "Cutoff between <med> and <high> tokens"}
    )
    name_col: Optional[str] = field(
            default="Preferred", # for logp
            metadata={"help": "Name of column with IUPAC names"}
    )
    conversion_pairs: Optional[str] = field(
            default="high_low",
            metadata={"help": "high_low means molecules with ground truth <low> " +
                              "will be generated with <high>, and vice versa. " +
                              "all_all means all molecules will be generated " +
                              "with all of <low>, <med>, and <high>"}
    )
    num_orig_iupacs: Optional[int] = field(
            default=10,
            metadata={"help": "how many starting molecules to generate from"}
    )
    masks_per_iupac: Optional[int] = field(
            default=-1,
            metadata={"help": "how many masks to use per source molecule (-1=all)"}
    )
    balanced_sample: Optional[bool] = field(
            default=True,
            metadata={"help": "Use an equal number of source iupacs per tgt val"}
    )
    orig_iupacs: Optional[str] = field(
            default=None,
            metadata={"help": "File containing starting molecules " +
                              "(fmt: chem,property_val)"}
    )


def mask_name(inputs, masked_ids, tokenizer):
    orig_inputs = inputs
    inputs = inputs.clone()

    masked_ids = torch.tensor(masked_ids)
    mask = torch.zeros_like(inputs).bool()
    mask[masked_ids] = True
    inputs[mask] = -1
    inputs[~mask] = torch.arange(inputs.numel())[~mask]
    inputs = torch.unique_consecutive(inputs)
    mask = inputs == -1
    inputs[~mask] = orig_inputs[inputs[~mask]]
    inputs[mask] = tokenizer.sentinels(torch.arange(mask.sum()))
    return inputs


def generate(model, tokenizer, inputs_list, masked_indices, n_candidates=1):
    if not isinstance(masked_indices[0], (list, tuple)):
        raise ValueError("must supply a list of masks")

    # inputs_list is a 1D list of tensors
    # masked_indices is a 3D list

    batch = []
    split_sizes = []
    for inputs, ms in zip(inputs_list, masked_indices):
        orig_inputs = inputs.clone()

        # add to batch, where each element is inputs with a different mask
        for m in ms:
            batch.append(mask_name(inputs, m, tokenizer).cuda())
        split_sizes.append(len(ms) * n_candidates)

    pad = tokenizer.pad_token_id
    model.eval()

    minibatches = itertools.zip_longest(*[iter(batch)] * 16,
                                        fillvalue=torch.tensor([]))
    minibatch_gen = []
    for minibatch in minibatches:
        minibatch = pad_sequence(minibatch,
                                 batch_first=True,
                                 padding_value=pad)

        minibatch_gen.append(model.generate(minibatch,
                                            do_sample=False,
                                            pad_token_id=pad,
                                            decoder_start_token_id=pad,
                                            num_return_sequences=n_candidates))
    # truncate last minibatch to correct size
    last_minibatch_size = (len(batch) - 1) % 16 + 1
    minibatch_gen[-1] = minibatch_gen[-1][:last_minibatch_size,:]
    assert sum(m.size(0) for m in minibatch_gen) == len(batch)

    max_len = max([g.size(1) for g in minibatch_gen])
    padded = [torch.cat([g, pad * torch.ones(g.size(0),
                                             max_len - g.size(1)).long().to(g.device)],
                          dim=1)
              if g.size(1) < max_len else g
              for g in minibatch_gen]
    generated = torch.cat(padded, dim=0)

    generated_split = generated.split(split_sizes)

    all_interleaved = []
    base_batch_idx = 0
    for generated, orig in zip(generated_split, inputs_list):
        interleaved = {}
        n_invalid = 0
        for i in range(generated.shape[0]):
            try:
                batch_idx = (base_batch_idx + i) // n_candidates
                # get rid of initial pad token on generated

                # this will throw a ValueError if the generated
                # sentinels are invalid
                gen = collapse_sentinels(tokenizer,
                                         batch[batch_idx],
                                         generated[i,1:])
                if (gen == tokenizer.unk_token_id).sum() > 0:
                    raise ValueError("unk in generated name")

                # remove <high>/<med>/<low> token
                decoded = tokenizer.decode(gen[1+H:])

                # if two distinct token sequences yield the same decoded
                # IUPAC name, prefer the one with fewer tokens
                is_dup = decoded in interleaved
                is_shorter = False
                if is_dup:
                    is_shorter = gen[1+H:].numel() < interleaved[decoded].numel()
                if not is_dup or (is_dup and is_shorter):
                    interleaved[decoded] = gen[1+H:].cpu()
            except ValueError:
                n_invalid += 1

        interleaved = [(decoded,
                        levenshtein_distance(orig[1+H:-1], tokens))
                       for decoded, tokens in interleaved.items()]
        all_interleaved.append(interleaved)

        assert generated.shape[0] % n_candidates == 0
        base_batch_idx += generated.shape[0] // n_candidates
    return all_interleaved

def mask_ids(length, span_lengths):
    max_id = length - span_lengths[-1] + 1
    comb = itertools.combinations(range(2, max_id),
                                  len(span_lengths))
    sli = range(len(span_lengths))
    masked = []
    for c in comb:
        new = list(itertools.chain(
                  *[range(start, start + slen)
                    for i, start, slen in zip(sli, c, span_lengths)
                   ]
              ))
        # check that it's actually len(span_lengths) spans
        nbreaks = sum([new[i+1] > new[i] + 1 for i in range(len(new) - 1)])
        if nbreaks == len(span_lengths) - 1:
            masked.append(new)
    return masked

masks_cache = {}
def get_masks(length, span_lengths):
    key = (length, tuple(map(tuple, span_lengths)))
    if key in masks_cache:
        return masks_cache[key]
    else:
        masks = [(sl, m) for sl in span_lengths for m in mask_ids(length, sl)]
        masks_cache[key] = masks
        return masks

def main():
    torch.manual_seed(42)

    parser = HfArgumentParser(IUPACArguments)
    iupac_args, = parser.parse_args_into_dataclasses()

    # get the list of molecules to generate from
    if iupac_args.tokenizer_type == "IUPAC":
        tokenizer_class = T5IUPACTokenizer
    elif iupac_args.tokenizer_type == "SMILES":
        tokenizer_class = T5SMILESTokenizer
    else:
        msg = "Unsupported tokenization type {}"
        raise RuntimeError(msg.format(iupac_args.tokenizer_type))

    tokenizer = tokenizer_class(vocab_file=iupac_args.vocab_fn)

    # get the trained model
    config = T5Config.from_pretrained(iupac_args.model_path)
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))

    # load weights from checkpoint
    model_fn = os.path.join(iupac_args.model_path, "pytorch_model.bin")
    state_dict = torch.load(model_fn, map_location="cpu")
    model.load_state_dict(state_dict)
    model.tie_weights()

    model.eval()
    model = model.cuda()

    collator = T5Collator(tokenizer.pad_token_id)

    low = tokenizer._convert_token_to_id("<low>")
    med = tokenizer._convert_token_to_id("<med>")
    high = tokenizer._convert_token_to_id("<high>")

    if iupac_args.conversion_pairs == "high_low":
        orig_iupacs = {"low": [], "high": []}
    elif iupac_args.conversion_pairs in ["high", "all_all"]:
        orig_iupacs = {"low": [], "med": [], "high": []}
    else:
        assert False, "invalid conversion_pairs arg"

    if iupac_args.orig_iupacs is None:
        dataset_kwargs = {
                "dataset_dir": iupac_args.dataset_dir,
                "tokenizer": tokenizer,
                "max_length": MAXLEN,
                "prepend_target": True,
                "low_cutoff": iupac_args.low_cutoff,
                "high_cutoff": iupac_args.high_cutoff,
                "target_col": iupac_args.target_col,
                "name_col": iupac_args.name_col,
                "dataset_size": 1000000,
                "mean_span_length": 3,
                "mask_probability": 0,
                "dataset_filename": iupac_args.dataset_filename
        }
        eval_dataset = IUPACDataset(train=False, **dataset_kwargs)

        iupacs_per_key = math.ceil(iupac_args.num_orig_iupacs / len(orig_iupacs.keys()))

        N = iupac_args.num_orig_iupacs
        i = 0
        while len(list(itertools.chain(*orig_iupacs.values()))) < N:
            input_ids = eval_dataset[i]["input_ids"]
            too_long = input_ids.numel() > 70
            has_unk = (input_ids == tokenizer.unk_token_id).sum() > 0
            if not has_unk and not too_long:
                first = input_ids[H].item()
                key = {low: "low", med: "med", high: "high"}[first]
                if key in orig_iupacs:
                    if iupac_args.balanced_sample:
                        # get iupacs_per_key for each key
                        if len(orig_iupacs[key]) <= iupacs_per_key:
                            orig_iupacs[key].append(eval_dataset[i])
                    else:
                        # take every non-unk-containing iupac
                        orig_iupacs[key].append(eval_dataset[i])
            else:
                # ignore names with <unk> in them and very long names
                pass
            i += 1

        assert len(list(itertools.chain(*orig_iupacs.values()))) == N
    else:
        # the user provided a file with original iupacs to use
        with open(iupac_args.orig_iupacs, "r") as orig_iupacs_f:
            reader = csv.DictReader(orig_iupacs_f, fieldnames=["iupac", "target"])
            for line in reader:
                if len(line["iupac"].strip()) == 0:
                    # ignore blank inputs
                    continue
                tokenized = tokenizer(line["iupac"])
                prop_token = {"low": low, "med": med, "high": high}[line["target"]]
                input_ids = tokenized["input_ids"]
                if H == 0:
                    input_ids = [prop_token] + input_ids
                elif H == 1:
                    input_ids = [input_ids[0], prop_token] + input_ids[H:]
                else:
                    assert False, "invalid H"
                tokenized["input_ids"] = torch.tensor(input_ids)
                orig_iupacs[line["target"]].append(tokenized)

    generated_iupacs = []
    for datum in itertools.chain(*orig_iupacs.values()):
        inputs = datum["input_ids"]
        labels = datum["labels"]

        #span_lengths = [[1], [2], [3], [1, 1], [1, 2], [2, 1], [2, 2]]
        # if you change span_lengths, you need to change the code in
        # best_in_dataset.py too if you want best_in_dataset.py to correctly
        # find molecules that could have been generated by gen_t5.py
        span_lengths = [[1], [2], [3], [4], [5]]

        if iupac_args.conversion_pairs == "high_low":
            # only change from <low> to <high>
            if inputs[H] == low:
                orig_logp = "low"
                new_logps = ["high"]
            elif inputs[H] == high:
                orig_logp = "high"
                new_logps = ["low"]
        elif iupac_args.conversion_pairs == "all_all":
            # try all of <low>, <med> and <high> for all molecules
            orig_logp = {low: "low", med: "med", high: "high"}[inputs[H].item()]
            new_logps = ["low", "med", "high"]
        elif iupac_args.conversion_pairs == "high":
            # only use <high> for all inputs
            orig_logp = {low: "low", med: "med", high: "high"}[inputs[H].item()]
            new_logps = ["high"]

        for new_logp in new_logps:
            inputs[H] = {"low": low, "med": med, "high": high}[new_logp]

            # don't print out <high>/<med>/<low> and </s>
            orig = tokenizer.decode(inputs[1+H:-1])
            base_out_dict = {"orig": orig,
                             "orig_logp": orig_logp,
                             "new_logp": new_logp}

            masks = get_masks(inputs.numel(), span_lengths)

            if iupac_args.masks_per_iupac > -1:
                masks = random.sample(masks, iupac_args.masks_per_iupac)

            # sort by slen and then group by slen
            grouped = itertools.groupby(sorted(masks, key=lambda x:x[0]),
                                        operator.itemgetter(0))

            for slens, group in grouped:
                masks = [t[1] for t in group]
                generated_iupacs.append(
                        base_out_dict |
                        {"nspans": len(slens),
                         "span_lengths": ",".join(map(str, slens)),
                         "gen": (inputs.clone(), masks)}
                )

    # actually generate now
    gen = generate(model,
                   tokenizer,
                   [d["gen"][0] for d in generated_iupacs],
                   [d["gen"][1] for d in generated_iupacs])

    for i, g in enumerate(gen):
        generated_iupacs[i]["gen"] = g

    # print output
    headers = ["orig", "orig_logp", "new_logp", "nspans", "span_lengths",
               "levenshtein_distance", "generated_iupac"]
    print(",".join(headers))
    unique_iupac = set()
    for record in generated_iupacs:
        # orig, orig_logp, final_logp, gen, nspans, span_lengths
        try:
            orig_idx = record["gen"].index((record["orig"], 0))
            record["gen"].pop(orig_idx)
        except ValueError:
            # orig not in generated, so no need to remove it
            pass
        for iupac, edit_distance in record["gen"]:
            cols = [record["orig"], record["orig_logp"],
                    record["new_logp"], str(record["nspans"]),
                    record["span_lengths"], str(edit_distance), iupac]
            # check if equal to orig since it's possible to have
            # an edit distance > 0 but tokenize.decode() to the same
            # IUPAC name
            if iupac not in unique_iupac and iupac != record["orig"]:
                unique_iupac.add(iupac)
                print('"' + '","'.join(cols) + '"')

# from https://rosettacode.org/wiki/Levenshtein_distance#Python
def levenshtein_distance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

if __name__ == "__main__":
    main()
