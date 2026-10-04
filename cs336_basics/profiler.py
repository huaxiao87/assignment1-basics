import cProfile
import pstats
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

tokenizer = get_tokenizer_from_vocab_merges_path(
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
)

cProfile.run("tokenizer.encode('Hello, world!')", "profile.txt")
pstats.Stats("profile.txt").sort_stats("time").print_stats()