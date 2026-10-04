from collections.abc import Iterable, Iterator
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # Build a mapping from byte tuples to token IDs
        self.byte_to_token = {byte: i for i, byte in self.vocab.items()}

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        # Step 1: Pre-tokenize
        if self.special_tokens is None:
            self.special_tokens = []
        corpus = re.split(r"|".join("("+re.escape(token)+")" for token in self.special_tokens), text)
        tokens = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in corpus:
            if chunk in self.special_tokens:
                tokens.append(self.byte_to_token[chunk.encode("utf-8")])
                continue
            for match in re.finditer(PAT, chunk):
                word = match.group(0).encode("utf-8")
                # Convert the word to byte tuples
                word = tuple(bytes([byte]) for byte in word)
                # Apply the merges
                for merge in self.merges:
                    i=0
                    new_word = []
                    while(i < len(word)):
                        if word[i:i+2] == merge:
                            new_word.append(merge[0]+merge[1])
                            i+=2
                        else:
                            new_word.append(word[i])
                            i+=1
                    word = tuple(new_word)
                # Convert the word to a token ID
                for byte in word:
                    if byte in self.byte_to_token:
                        tokens.append(self.byte_to_token[byte])
        return tokens

    def decode(self, ids: list[int]) -> str:
        # Convert the token IDs to bytes and then to a string
        text = b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
        return text


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass