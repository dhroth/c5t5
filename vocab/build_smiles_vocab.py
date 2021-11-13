import sys

from t5 import T5SMILESTokenizer

# bit of a hack, but a valid vocab file isn't actually needed
# for tokenization, since T5SMILESTokenizer._tokenize
# doesn't use a vocab
# This file creates a valid vocab file, which we need  later
tok = T5SMILESTokenizer("build_smiles_vocab.py")

# file with a list of all SMILES in the dataset
# one SMILES per line, & a header line is expected
smiles_fn = sys.argv[1]

vocab = set()
with open(smiles_fn, "r") as f:
    for i, smiles in enumerate(f):
        if i == 0:
            # skip header
            continue
        tokenized = tok._tokenize(smiles)
        vocab.update(set(tokenized))

print("\n".join(sorted(list(vocab))))
