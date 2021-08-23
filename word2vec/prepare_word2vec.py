from t5 import T5IUPACTokenizer
import sys


# vocab file for the tokenizer
vocab_fn = sys.argv[1]

# file containing one IUPAC name per row
iupacs_fn = sys.argv[2]

tokenizer = T5IUPACTokenizer(vocab_fn)

with open(iupacs_fn, "r") as f:
    for i, iupac in enumerate(f):
        if i % 100000 == 0:
            print(i, file=sys.stderr)
        print(" ".join(tokenizer.convert_ids_to_tokens(
                           tokenizer(iupac)["input_ids"][:-1]
                      )))
