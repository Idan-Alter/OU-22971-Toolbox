from pathlib import Path
import socket

import ray


def ping(label: str, i: int) -> str:
    host = socket.gethostname()
    msg = f"{label} task {i} ran on {host}"
    print(msg, flush=True)
    return msg


# Demo-only instrumentation: these custom resources make placement visible in the smoke test.
# Requires the custom "head" resource, which exists only on the head node.
@ray.remote(resources={"head": 0.001})
def ping_head(i: int) -> str:
    return ping("head", i)


# Requires the custom "worker" resource, which exists only on worker nodes.
@ray.remote(resources={"worker": 0.001})
def ping_worker(i: int) -> str:
    return ping("worker", i)


if __name__ == "__main__":
    ray.init(address="auto")

    print(f"Driver connected from {socket.gethostname()}", flush=True)
    results = ray.get(
        [
            ping_head.remote(0),
            ping_head.remote(1),
            ping_worker.remote(2),
            ping_worker.remote(3),
        ]
    )
    print("Results:", results, flush=True)

    out_path = Path("/workspace/smoke_test_output.txt")
    out_path.write_text(
        "Ray smoke test succeeded\n" + "\n".join(results) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote artifact to {out_path}", flush=True)
