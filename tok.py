from transformers import T5Tokenizer
import re

class T5IUPACTokenizer(T5Tokenizer):
    def prepare_for_tokenization(self, text, is_split_into_words=False,
                                 **kwargs):
        return re.sub(" ", "_", text), kwargs

    def _decode(self, *args, **kwargs):
        # replace "_" with " ", except for the _ in extra_id_#
        text = super()._decode(*args, **kwargs)
        text = re.sub("extra_id_", "extraAidA", text)
        text = re.sub("_", " ", text)
        text = re.sub("extraAidA", "extra_id_", text)
        return text

    def sentinels(self, sentinel_ids):
        return self.vocab_size - sentinel_ids - 1

    def sentinel_mask(self, ids):
        return ((self.vocab_size - self._extra_ids <= ids) &
                (ids < self.vocab_size))

    #def _tokenize(self, text, sample=False):
    #    pieces = super()._tokenize(text, sample=sample)
    #    # sentencepiece adds a non-printing token at the start. Remove it
    #    return pieces[1:]

class T5SMILESTokenizer(T5Tokenizer):
    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i)
                                         for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x),
                                          additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens "
                    "({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the "
                    "extra_ids tokens"
                )

        super(T5Tokenizer, self).__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        with open(self.vocab_file, "r") as f:
            self.vocab = list(map(str.strip, f.readlines()))
        self.reverse_vocab = {w: i for i, w in enumerate(self.vocab)}

    def sentinels(self, sentinel_ids):
        return self.vocab_size - sentinel_ids - 1

    def sentinel_mask(self, ids):
        return ((self.vocab_size - self._extra_ids <= ids) &
                (ids < self.vocab_size))


    @property
    def vocab_size(self):
        return len(self.vocab) + self._extra_ids

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, d):
        self.__dict__ = d

    def _tokenize(self, text):
        tokens = []
        i = 0
        in_brackets = False
        while i < len(text):
            if text[i] in ["[", "<"]:
                in_brackets = True
                tokens.append("")

            if in_brackets:
                tokens[-1] += text[i]
            else:
                if text[i] in ["r", "l"]:
                    # handle Cl & Br
                    tokens[-1] += text[i]
                else:
                    tokens.append(text[i])

            if text[i] in ["]", ">"]:
                in_brackets = False
            i += 1
        return tokens

    def _convert_token_to_id(self, token):
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        else:
            return self.reverse_vocab[token]

    def _convert_id_to_token(self, index):
        if index < len(self.vocab):
            token = self.vocab[index]
        else:
            token = "<extra_id_{}>".format(self.vocab_size - 1 - index)
        return token

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix):
        raise NotImplementedError()


