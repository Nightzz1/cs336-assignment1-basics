import os, json
import regex as re
import multiprocessing as mp
from collections import Counter, defaultdict

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(
    file_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    pre_token_frequency: dict[tuple[bytes, ...], int] = {}

    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
    special_tokens_bytes.sort(key=len, reverse=True)  # Sort by length to match longer tokens first
    pattern = b"|".join(
        re.escape(token) 
        for token in special_tokens_bytes
    )

    with open(file_path, "rb") as f:
        f.seek(start)
        bytes_to_read = end - start
        bytes_read = 0
        while bytes_read < bytes_to_read:
            f.seek(start + bytes_read)
            MAX_CHUNK_SIZE = 4096  # 4KB

            # Read chunks of data till find a special token or reach the end of the chunk
            chunk_data = bytearray(
                f.read(min(MAX_CHUNK_SIZE, bytes_to_read - bytes_read))
            )
            bytes_read += len(chunk_data)
            special_token_found = any(token in chunk_data for token in special_tokens_bytes)
            chunks_to_check = chunk_data.copy()
            
            while not special_token_found and bytes_read < bytes_to_read:
                new_chunk_data = f.read(min(MAX_CHUNK_SIZE, bytes_to_read - bytes_read))
                bytes_read += len(new_chunk_data)
                if not new_chunk_data:
                    if bytes_read < bytes_to_read:
                        raise ValueError("Reached end of file before reading the expected number of bytes.")
                    break
                chunk_data.extend(new_chunk_data)

                chunks_to_check.extend(new_chunk_data)
                special_token_found = any(token in chunks_to_check for token in special_tokens_bytes)
                chunks_to_check = bytearray(new_chunk_data)

            parts = re.split(pattern, chunk_data)
            parts_to_process = parts[:-1] if bytes_read < bytes_to_read else parts
            for part in parts_to_process:
                if part:
                    token_iter = re.finditer(PAT, part.decode("utf-8"))
                    for match in token_iter:
                        token_bytes = match.group().encode("utf-8")
                        pre_token_frequency_index = ()
                        for i in range(len(token_bytes)):
                            pre_token_frequency_index += (token_bytes[i:i+1],)
                        if pre_token_frequency_index in pre_token_frequency:
                            pre_token_frequency[pre_token_frequency_index] += 1
                        else:
                            pre_token_frequency[pre_token_frequency_index] = 1
            if bytes_read < bytes_to_read:
                bytes_read -= len(parts[-1])  # Adjust bytes_read to account for the last incomplete part

    return pre_token_frequency

def get_pair_counts(
    token_tuple: tuple[bytes, ...]
) -> Counter[tuple[bytes, bytes]]:
    return Counter(zip(token_tuple, token_tuple[1:]))

def merge_pair(
    token_tuple: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
    new_token: bytes,
) -> tuple[bytes, ...]:

    result = []
    i = 0

    while i < len(token_tuple):
        if (
            i + 1 < len(token_tuple)
            and token_tuple[i] == pair[0]
            and token_tuple[i + 1] == pair[1]
        ):
            result.append(new_token)
            i += 2
        else:
            result.append(token_tuple[i])
            i += 1

    return tuple(result)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
):
    current_size = 0
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    pre_token_frequency: dict[tuple[bytes, ...], int] = {}

    for token in special_tokens:
        # If token is an empty string, raise an error
        if token == "":
            raise ValueError("Special tokens cannot be empty strings.")

    with open(input_path, "rb") as f:
        # Find chunk boundaries for parallel processing
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Create a pool of processes to process chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                process_chunk,
                [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
            )

            for result in results:
                for token_tuple, count in result.items():
                    if token_tuple in pre_token_frequency:
                        pre_token_frequency[token_tuple] += count
                    else:
                        pre_token_frequency[token_tuple] = count

    for i in range(256):
        vocab[current_size] = bytes([i])
        current_size += 1
    for special_token in special_tokens:
        vocab[current_size] = special_token.encode("utf-8")
        current_size += 1

    sequences = list(pre_token_frequency.keys())
    counts = list(pre_token_frequency.values())

    pair_frequency: Counter[tuple[bytes, bytes]] = Counter()

    pair_to_sequences: dict[
        tuple[bytes, bytes],
        set[int]
    ] = defaultdict(set)

    for seq_id, (token_tuple, count) in enumerate(
        zip(sequences, counts)
    ):
        local_pairs = get_pair_counts(token_tuple)

        for pair, occurrences in local_pairs.items():
            pair_frequency[pair] += occurrences * count
            pair_to_sequences[pair].add(seq_id)
    

    # carry out BPE merges until we reach the desired vocab size
    while current_size < vocab_size and pair_frequency:
        most_frequent_pair = max(
            pair_frequency,
            key=lambda p: (pair_frequency[p], p),
        )

        merges.append(most_frequent_pair)

        new_token = b"".join(most_frequent_pair)
        vocab[current_size] = new_token
        current_size += 1

        affected_seq_ids = list(
            pair_to_sequences.get(most_frequent_pair, ())
        )

        for seq_id in affected_seq_ids:

            old_seq = sequences[seq_id]
            sequence_frequency = counts[seq_id]

            old_pairs = get_pair_counts(old_seq)

            # merge
            new_seq = merge_pair(
                old_seq,
                most_frequent_pair,
                new_token,
            )

            new_pairs = get_pair_counts(new_seq)

            # Update pair frequencies
            all_changed_pairs = (
                old_pairs.keys() | new_pairs.keys()
            )

            for pair in all_changed_pairs:
                old_count = old_pairs.get(pair, 0)
                new_count = new_pairs.get(pair, 0)

                delta = (
                    new_count - old_count
                ) * sequence_frequency

                if delta != 0:
                    pair_frequency[pair] += delta

                    if pair_frequency[pair] == 0:
                        del pair_frequency[pair]

            # Update inverted index
            for pair in old_pairs.keys() - new_pairs.keys():
                pair_to_sequences[pair].discard(seq_id)

                if not pair_to_sequences[pair]:
                    del pair_to_sequences[pair]

            for pair in new_pairs.keys() - old_pairs.keys():
                pair_to_sequences[pair].add(seq_id)

            sequences[seq_id] = new_seq
        
    return vocab, merges


if __name__ == "__main__":
    file_dir = "tests/fixtures/"
    file_name = "corpus.en"
    file_path = os.path.join(file_dir, file_name)
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)

    # Output the vocab and merges to json files
    output_dir = "data/"
    vocab_name = "vocab.json"
    vocab_path = os.path.join(output_dir, vocab_name)
    merges_name = "merges.json"
    merges_path = os.path.join(output_dir, merges_name)

    # dict[int, bytes] -> dict[str, str]
    vocab_json = { str(token_id): token.hex() for token_id, token in vocab.items() }
    # list[tuple[bytes, bytes]] -> list[list[str]]
    merges_json = [ [left.hex(), right.hex()] for left, right in merges]

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2)

    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_json, f, indent=2)
