from collections.abc import Iterable, Iterator
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.merges_dict = {merges[i]:i for i in range(len(merges))}
        self.special_tokens = special_tokens
        # Build a mapping from byte tuples to token IDs
        self.byte_to_token = {byte: i for i, byte in self.vocab.items()}

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        # Step 1: Pre-tokenize
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            corpus = re.split(r"|".join("("+re.escape(token)+")" for token in sorted_special_tokens), text)
        else:
            corpus = [text]
        tokens = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in corpus:
            if chunk == "" or chunk is None:
                continue
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.byte_to_token[chunk.encode("utf-8")])
                continue
            for match in re.finditer(PAT, chunk):
                word = match.group(0).encode("utf-8")
                # Convert the word to byte tuples
                word = tuple(bytes([byte]) for byte in word)
                # Apply the merges
                # Enumerate the byte pairs
                byte_pairs = []
                for i in range(len(word)-1):
                    byte_pairs.append(word[i:i+2])

                while True:
                    min_rank = len(self.merges)
                    min_rank_byte_pair = None
                    for byte_pair in byte_pairs:
                        if byte_pair in self.merges_dict:
                            if self.merges_dict[byte_pair] < min_rank:
                                min_rank = self.merges_dict[byte_pair]
                                min_rank_byte_pair = byte_pair
                    if min_rank_byte_pair is not None:
                        # Replace the byte pair with the merged byte
                        i = 0
                        new_word = []   
                        while i < len(word):
                            if word[i:i+2] == min_rank_byte_pair:
                                new_word.append(min_rank_byte_pair[0]+min_rank_byte_pair[1])
                                i+=2
                            else:
                                new_word.append(word[i])
                                i+=1
                        word = tuple(new_word)
                        # Reenumerate the byte pairs
                        byte_pairs = []
                        for i in range(len(word)-1):
                            byte_pairs.append(word[i:i+2])
                        continue
                    else:
                        break
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
        for text in iterable:
            for token in self.encode(text):
                yield token