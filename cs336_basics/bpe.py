import regex as re
from collections import defaultdict



def train_bpe(corpus_path: str, vocab_size: int = 10000, special_tokens: list[str] = []):
    """
    Train a BPE model on a corpus.
    """
    # Load the corpus
    with open(corpus_path, "r") as f:
        corpus = f.read()

    # Vocabulary initialization
    # Our initial vocabulary is simply the set of all bytes. Since there are 256 possible byte values, our initial vocabulary is of size 256.
    vocabulary = {i:bytes([i]) for i in range(256)}
    # Add special tokens to the vocabulary
    vocabulary.update({i+len(vocabulary):special_token.encode("utf-8") for i, special_token in enumerate(special_tokens)})
    
    # Pre-tokenization
    pre_token_counts = defaultdict(int)
    merges :list[tuple[bytes, bytes]] = [] # merges are ordered by order of creation
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # When using it in your code, however, you should use re.finditer to avoid storing the pre-tokenized words as you construct your mapping from pre-tokens to their counts.
    # Split the corpus using all special tokens, re.escape() to escape special tokens
    corpus = re.split(r"|".join(re.escape(token) for token in special_tokens), corpus)
    for chunk in corpus:
        for match in re.finditer(PAT, chunk):
            word = match.group(0)
            pre_token_counts[word] += 1

    # Convert pre-tokens to bytes tuples
    pre_token_counts = {tuple(bytes([byte]) for byte in word.encode("utf-8")): count for word, count in pre_token_counts.items()}
    # Training
    for i in range(vocab_size - len(vocabulary)):
        new_pre_token_counts = defaultdict(int)

        # Get the most frequent byte pair
        byte_pair_counts = defaultdict(int)
        for word, count in pre_token_counts.items():
            for j in range(len(word)-1):
                byte_pair_counts[word[j:j+2]] += count
        # Get the most frequent byte pair, choose the lexicographically largest one if there are multiple
        if len(byte_pair_counts) == 0:
            break # No more byte pairs to merge
        most_frequent_pair = max(byte_pair_counts, key=lambda x: (byte_pair_counts[x], x))

        # Make the new byte pair a single byte in the vocabulary
        new_byte_pair = most_frequent_pair[0] + most_frequent_pair[1]
        vocabulary[len(vocabulary)] = new_byte_pair

        # Add the new byte pair to the merges
        merges.append(most_frequent_pair)

        # Update the pre-token counts by replacing the most frequent pair with the new byte pair
        for word, count in pre_token_counts.items():
            new_word = []
            k = 0
            while(k < len(word)):
                if word[k:k+2] == most_frequent_pair:
                    # Replace the most frequent pair with the new byte pair
                    new_word.append(new_byte_pair)
                    k+=2
                else:
                    new_word.append(word[k])
                    k+=1
            new_pre_token_counts[tuple(new_word)] += count

        # Update the pre-token counts
        pre_token_counts = new_pre_token_counts

    return vocabulary, merges