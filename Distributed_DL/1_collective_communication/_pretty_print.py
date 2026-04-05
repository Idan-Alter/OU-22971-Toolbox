"""Pretty-print small multi-line blocks for the collective communication demos."""

def print_block(title: str, *lines: str) -> None:
    body = "\n".join([title, *[f"  {line}" for line in lines], ""])
    print(body, flush=True)
