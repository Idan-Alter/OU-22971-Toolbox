from __future__ import annotations

import argparse
import json
import socket
import time
from collections import Counter
from pathlib import Path

import ray

NUM_REDUCERS = 3
DEFAULT_DOCS_FILE = "mr_job_docs.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MapReduce word count with mapper bucket refs and batched partial reductions."
    )
    parser.add_argument("--repeat", type=positive_int, default=20, help="How many times to repeat the mini-corpus.")
    parser.add_argument(
        "--docs-file",
        type=str,
        default=DEFAULT_DOCS_FILE,
        help="Text file containing one tokenized document per line. Useful for showing that --working-dir files ship with the job.",
    )
    parser.add_argument(
        "--docs-per-chunk",
        type=positive_int,
        default=3,
        help="How many tokenized documents each mapper receives.",
    )
    parser.add_argument(
        "--reduce-batch-size",
        type=positive_int,
        default=10,
        help="Launch a partial reduce once this many ready buckets accumulate for a reducer.",
    )
    parser.add_argument(
        "--straggler-delay-s",
        type=non_negative_float,
        default=1.0,
        help="Artificial delay added to the final mapper chunk.",
    )
    parser.add_argument(
        "--reduce-delay-per-bucket-s",
        type=non_negative_float,
        default=0.02,
        help="Artificial delay multiplied by the number of buckets in each partial reduce.",
    )
    parser.add_argument("--top-k", type=positive_int, default=12, help="How many global words to print.")
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Optional Ray address. Leave unset for local execution; use 'auto' on a running cluster.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path for a JSON summary artifact, e.g. /workspace/mr_chunks_output.json on the head node.",
    )
    return parser.parse_args()


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def non_negative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def build_corpus(seed_docs: list[list[str]], repeat: int) -> list[list[str]]:
    return [doc[:] for _ in range(repeat) for doc in seed_docs]


def chunked(items: list[list[str]], chunk_size: int) -> list[list[list[str]]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_tokens(docs: list[list[str]]) -> list[str]:
    return [token for doc in docs for token in doc]


def format_top(counter: Counter[str], top_k: int) -> str:
    return ", ".join(f"{word}:{count}" for word, count in counter.most_common(top_k))


def reducer_for_word(word: str) -> int:
    first = word[0].lower()
    if first <= "i":
        return 0
    if first <= "r":
        return 1
    return 2


@ray.remote(num_returns=NUM_REDUCERS + 1)
def count_words_with_delay(
    chunk_id: int,
    docs: list[list[str]],
    slow_id: int,
    straggler_delay_s: float,
):
    buckets = [Counter() for _ in range(NUM_REDUCERS)]

    for word in flatten_tokens(docs):
        buckets[reducer_for_word(word)][word] += 1

    if chunk_id == slow_id:
        time.sleep(straggler_delay_s)

    return chunk_id, *buckets


@ray.remote
def partial_reduce(delay_per_bucket_s: float, *buckets: Counter[str]) -> Counter[str]:
    time.sleep(delay_per_bucket_s * len(buckets))

    total = Counter()
    for bucket in buckets:
        total.update(bucket)
    return total


def resolve_docs_file(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def load_seed_docs(path: Path) -> list[list[str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    docs = [line.split() for line in lines if line]
    if not docs:
        raise ValueError(f"No tokenized documents found in {path}")
    return docs


def sample_directory(path: Path, limit: int = 12) -> list[str]:
    return sorted(entry.name for entry in path.iterdir())[:limit]


def main() -> None:
    args = parse_args()

    if args.address:
        ray.init(address=args.address)
    else:
        ray.init(include_dashboard=False)

    driver_host = socket.gethostname()
    driver_cwd = Path.cwd().resolve()
    script_path = Path(__file__).resolve()
    docs_file = resolve_docs_file(args.docs_file)
    seed_docs = load_seed_docs(docs_file)
    working_dir_entries = sample_directory(driver_cwd)

    corpus = build_corpus(seed_docs, args.repeat)
    chunks = chunked(corpus, args.docs_per_chunk)
    slow_id = len(chunks) - 1
    expected_counts = Counter(flatten_tokens(corpus))

    banner = "=" * 72
    print(f"\n{banner}")
    print("PACKAGED WORKING DIRECTORY ON DRIVER")
    print(banner)
    print(f"driver_host       = {driver_host}")
    print(f"script_path       = {script_path}")
    print(f"driver_cwd        = {driver_cwd}")
    print(f"docs_file         = {docs_file}")
    print(f"working_dir_files = {working_dir_entries}")
    print(f"seed_docs         = {len(seed_docs)} loaded from {docs_file.name}")
    print(f"{banner}")
    print("RUN CONFIG")
    print(banner)
    print(f"cluster_resources = {ray.cluster_resources()}")
    print(
        f"corpus_size       = {len(corpus)} docs | chunks={len(chunks)} | "
        f"reducers={NUM_REDUCERS} | reduce_batch_size={args.reduce_batch_size}"
    )
    print(
        f"slow_chunk_id     = {slow_id} | straggler_delay={args.straggler_delay_s:0.2f}s | "
        f"reduce_delay_per_bucket={args.reduce_delay_per_bucket_s:0.2f}s"
    )
    print(f"{banner}\n")

    t0 = time.perf_counter()

    map_output_refs = [
        count_words_with_delay.remote(chunk_id, docs, slow_id, args.straggler_delay_s)
        for chunk_id, docs in enumerate(chunks)
    ]

    # The first return is a small completion token; the rest are reducer-specific bucket refs.
    pending_chunks = [result[0] for result in map_output_refs]
    mapper_completion_order: list[int] = []
    ready_bucket_refs = {reducer_id: [] for reducer_id in range(NUM_REDUCERS)}
    partial_reduce_refs = {reducer_id: [] for reducer_id in range(NUM_REDUCERS)}

    while pending_chunks:
        ready_refs, pending_chunks = ray.wait(pending_chunks, num_returns=1)
        ready_chunk_id = ray.get(ready_refs[0])
        mapper_completion_order.append(ready_chunk_id)

        for reducer_id, bucket_ref in enumerate(map_output_refs[ready_chunk_id][1:]):
            ready_bucket_refs[reducer_id].append(bucket_ref)

        print(f"mapper {ready_chunk_id:>2} finished -> routed {NUM_REDUCERS} bucket refs")

        for reducer_id in range(NUM_REDUCERS):
            if len(ready_bucket_refs[reducer_id]) >= args.reduce_batch_size:
                batch = ready_bucket_refs[reducer_id]
                partial_reduce_refs[reducer_id].append(
                    partial_reduce.remote(args.reduce_delay_per_bucket_s, *batch)
                )
                print(
                    f"launched partial reduce for reducer {reducer_id} "
                    f"with {len(batch)} ready buckets"
                )
                ready_bucket_refs[reducer_id] = []

    for reducer_id in range(NUM_REDUCERS):
        if ready_bucket_refs[reducer_id]:
            batch = ready_bucket_refs[reducer_id]
            partial_reduce_refs[reducer_id].append(
                partial_reduce.remote(args.reduce_delay_per_bucket_s, *batch)
            )
            print(
                f"final partial reduce for reducer {reducer_id} "
                f"with {len(batch)} ready buckets"
            )
            ready_bucket_refs[reducer_id] = []

    for reducer_id in range(NUM_REDUCERS):
        print(
            f"final merge for reducer {reducer_id} "
            f"with {len(partial_reduce_refs[reducer_id])} partial results"
        )

    final_reduce_refs = [
        partial_reduce.remote(args.reduce_delay_per_bucket_s, *partial_reduce_refs[reducer_id])
        for reducer_id in range(NUM_REDUCERS)
    ]
    final_reduce_outputs = ray.get(final_reduce_refs)
    runtime = time.perf_counter() - t0

    total_counts = Counter()
    reducer_summaries = []

    print("\nReducer outputs:")
    for reducer_id, counts in enumerate(final_reduce_outputs):
        total_counts.update(counts)
        reducer_summaries.append(
            {
                "reducer_id": reducer_id,
                "unique_words": len(counts),
                "num_partial_reductions": len(partial_reduce_refs[reducer_id]),
                "top_words": [{"word": word, "count": count} for word, count in counts.most_common(args.top_k)],
            }
        )
        print(
            f"reducer {reducer_id} | unique_words={len(counts):>3} | "
            f"top words: {format_top(counts, 4)}"
        )

    matches_direct_count = total_counts == expected_counts

    print("\nGlobal word count (top words):")
    print(format_top(total_counts, args.top_k))
    print(f"matches direct count = {matches_direct_count}")
    print(f"\nmapper completion order = {mapper_completion_order}")
    print(f"runtime = {runtime:0.2f}s")

    if args.output_file:
        summary = {
            "driver_host": driver_host,
            "script_path": str(script_path),
            "driver_cwd": str(driver_cwd),
            "docs_file": str(docs_file),
            "working_dir_entries": working_dir_entries,
            "seed_docs": len(seed_docs),
            "repeat": args.repeat,
            "docs_per_chunk": args.docs_per_chunk,
            "num_chunks": len(chunks),
            "num_reducers": NUM_REDUCERS,
            "reduce_batch_size": args.reduce_batch_size,
            "slow_chunk_id": slow_id,
            "straggler_delay_s": args.straggler_delay_s,
            "reduce_delay_per_bucket_s": args.reduce_delay_per_bucket_s,
            "runtime_s": runtime,
            "mapper_completion_order": mapper_completion_order,
            "reducer_outputs": reducer_summaries,
            "top_words": [{"word": word, "count": count} for word, count in total_counts.most_common(args.top_k)],
            "matches_direct_count": matches_direct_count,
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary artifact to {args.output_file}")


if __name__ == "__main__":
    main()
