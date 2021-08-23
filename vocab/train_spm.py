import sentencepiece as spm
import sys
from collections import Counter

# file with a list of IUPAC names (can be just 1 line if you want)
iupacs_fn = int(sys.argv[1])


with open("opsin_vocab_reduced.txt", "r") as f:
    words = f.read().split("\n")
words = list(map(str, range(100))) + words

vocab_size = len(words) + 1

user_defined_symbols = words

print("num user defined:", len(user_defined_symbols))

args = {"input": sys.argv[1],
        "model_type": "unigram",
        "model_prefix": "iupac_spm".format(vocab_size),
        "vocab_size": vocab_size,
        "input_sentence_size": 100000,
        "shuffle_input_sentence": True,
        "user_defined_symbols": user_defined_symbols,
        "split_by_number": False,
        "split_by_whitespace": False,
        "hard_vocab_limit": False,
        "max_sentencepiece_length": 32,
        "character_coverage": 0.99,
        "pad_id": 0,
        "eos_id": 1,
        "unk_id": 2,
        "bos_id": -1}

spm.SentencePieceTrainer.train(**args)
