import os, regex, json
from typing import BinaryIO, Iterable, Iterator

class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.encode_vocab = {v: k for k, v in vocab.items()}  # Reverse mapping for encoding

    @classmethod
    def from_files(cls, vocab_file, merges_filepath, special_tokens=None):
        with open(vocab_file, "rb") as vf:
            vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in json.load(vf).items()}

        with open(merges_filepath, "r", encoding="utf-8") as mf:
            merges = [tuple(map(bytes.fromhex, merge)) for merge in json.load(mf)]

        return cls(vocab, merges, special_tokens)

    def bpe_merge(self, token_bytes: bytes) -> list[bytes]:
        tokens = [bytes([b]) for b in token_bytes]
        for left, right in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == left and tokens[i + 1] == right:
                    tokens[i:i + 2] = [left + right]
                i += 1
        return tokens

    def encode(self, text: str) -> list[int]:
        token_ids = []
        text_bytes = text.encode("utf-8")
        # split by special tokens
        if self.special_tokens:
            special_tokens_bytes = [token.encode("utf-8") for token in self.special_tokens]
            special_tokens_set = set(special_tokens_bytes)
            special_tokens_bytes.sort(key=len, reverse=True)  # Sort by length to match longer tokens first
            pattern = b"(" + b"|".join(
                regex.escape(token) 
                for token in special_tokens_bytes
            ) + b")"
            parts = regex.split(pattern, text_bytes)
        else:
            special_tokens_set = set()
            parts = [text_bytes]

        
        # pre-tokenization
        for part in parts:
            if not part:
                continue

            if part in special_tokens_set:
                if part not in self.encode_vocab:
                    raise ValueError(f"Special token {part} not found in vocabulary.")
                token_ids.append(self.encode_vocab[part])
                continue

            # normal pre-tokenization using regex
            pre_token_iter = regex.finditer(self.PAT, part.decode("utf-8"))
            for match in pre_token_iter:
                tokens_bytes = match.group().encode("utf-8")
                bpe_tokens = self.bpe_merge(tokens_bytes)
                for bpe_token in bpe_tokens:
                    if bpe_token not in self.encode_vocab:
                        raise ValueError(f"Token {bpe_token} not found in vocabulary.")
                    token_ids.append(self.encode_vocab[bpe_token])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocabulary.")
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
    

if __name__ == "__main__":
    # Example usage
    vocab_file = "data/vocab.json" 
    merges_file = "data/merges.json"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens)

    text = "Hello, world! This is a test."
    encoded_ids = tokenizer.encode(text)
    print("Encoded IDs:", encoded_ids)