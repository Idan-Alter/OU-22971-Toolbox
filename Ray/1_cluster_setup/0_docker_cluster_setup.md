# Ray Unit 1 - Docker Cluster Setup

## Setup

1. Install **Docker Desktop**.

2. Run the Docker commands below from `Ray/1_cluster_setup`.

The Docker lifecycle commands do not depend on Conda.
If you also want to run the host-side `ray job ...` commands from this guide, activate the Ray environment from the `Ray` folder first:

```powershell
conda activate 22971-ray
```

## Cluster setup

We will **use Docker** to spin up a small local Ray (virtual) cluster with:
- **1 head node**
- **N worker nodes** (chosen by you at startup)

`1_cluster_setup/head_workspace` is mounted into the **head** container as `/workspace`.

That gives us a dedicated place for small output files without mixing them into the lesson files themselves.

### 0. `Dockerfile`

```dockerfile
# Base Linux image with Miniconda already installed.
FROM continuumio/miniconda3:26.1.1-1

# All project files inside the container will live here.
WORKDIR /workspace

# Compose builds from the Ray part root, so this copies the part-local env spec.
COPY environment.yml /tmp/environment.yml

# Create the full course environment inside the image.
# Every container built from this image will have the same Python stack.
RUN conda env create -f /tmp/environment.yml && conda clean -afy
```

The env spec becomes part of the Docker build.
Both head and workers get the same `22971-ray` environment.
We pin the base image tag so this setup stays reproducible.

---

### 1. `docker-compose.yml`

```yaml
services:
  ray-head:
    build:
      context: ..
      dockerfile: 1_cluster_setup/Dockerfile
    container_name: ray-head
    hostname: ray-head
    working_dir: /workspace
    volumes:
      - ./head_workspace:/workspace
    ports:
      - "8265:8265"
    shm_size: "3gb"
    healthcheck:
      test:
        - CMD-SHELL
        - >-
          bash -lc "source /opt/conda/etc/profile.d/conda.sh &&
          conda activate 22971-ray &&
          ray status --address=$(hostname -i):6379 >/dev/null 2>&1"
      interval: 5s
      timeout: 5s
      retries: 12
      start_period: 10s
    command: >
      bash -lc "source /opt/conda/etc/profile.d/conda.sh &&
      conda activate 22971-ray &&
      ray start --head --node-ip-address=$(hostname -i) --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --resources='{\"head\": 1}' --block"

  ray-worker:
    build:
      context: ..
      dockerfile: 1_cluster_setup/Dockerfile
    working_dir: /workspace
    shm_size: "3gb"
    depends_on:
      ray-head:
        condition: service_healthy
    command: >
      bash -lc "source /opt/conda/etc/profile.d/conda.sh &&
      conda activate 22971-ray &&
      ray start --node-ip-address=$(hostname -i) --address=ray-head:6379 --resources='{\"worker\": 1}' --block"
```

### Walkthrough

#### `services`
We define two service types:
- `ray-head`: the cluster entry point
- `ray-worker`: a worker template that we scale to any number of containers

The Docker build uses:
- `context: ..` so the image can see the part-local `Ray/environment.yml`
- `dockerfile: 1_cluster_setup/Dockerfile` so we can keep the Docker-specific files in this folder

#### `healthcheck` + `depends_on`
The head container now publishes a healthcheck that runs `ray status` against its local GCS address.
That is more reliable than plain container ordering because Docker can wait until the Ray head is actually accepting connections.

The worker uses:
1. `depends_on: condition: service_healthy`
2. the head's healthcheck result

So workers join only after the head is genuinely ready, which avoids cold-start race conditions.
We also bind Ray to each container's actual bridge-network IP via `$(hostname -i)`, which is more reliable in Docker than asking Ray to reuse the service hostname as its node IP.

#### `command`
Each container runs a shell command that:
1. loads Conda into the shell
2. activates `22971-ray`
3. starts the appropriate Ray process
4. stays alive via `--block`

We also attach one small custom resource per node type:
- the head advertises `head`
- workers advertise `worker`

That lets the smoke test pin specific tasks at the decorator level so we can prove work lands on both node types.

#### `ray-head`
Starts the cluster head.

Key flags:
- `--head`: create the cluster
- `--node-ip-address=$(hostname -i)`: bind Ray to the container's actual Docker-network IP
- `--port=6379`: address used by workers
- `--dashboard-host=0.0.0.0`: expose the dashboard outside the container
- `--dashboard-port=8265`: dashboard / Jobs API port
- `--block`: keep the container alive

#### `ray-worker`
Connects to the head and joins the cluster as a worker.

#### `volumes`

```yaml
volumes:
  - ./head_workspace:/workspace
```

We mount only `head_workspace` into the **head** container:
- files written to `/workspace` by the driver appear on the host under `1_cluster_setup/head_workspace`
- only the head has this bind mount; a worker writing to its own local filesystem writes inside that worker container, not into `head_workspace`
- this keeps generated artifacts separate from the markdown, compose file, and lesson scripts
- job submission still works because `ray job submit --working-dir .` uploads the current folder to the cluster through the Jobs API; it does **not** rely on this bind mount

The worker nodes do **not** need a host mount for cluster communication.

#### `ports`

```yaml
- "8265:8265"
```

- `8265`: dashboard + Jobs API exposed to the host

The Ray cluster address `6379` still exists, but it is only used inside the Docker network by the worker containers via `ray-head:6379`.

#### `shm_size: "3gb"`
Ray uses **shared memory** heavily for passing objects between processes on the same node.
If `/dev/shm` is too small, performance degrades and large objects may fail in awkward ways.
So we explicitly reserve a larger shared-memory segment for each container to keep Ray's object store in `/dev/shm` instead of falling back to the disk based `/tmp`.

---

### 2. Build and start the cluster

Build the image:

```powershell
docker compose build
```

Start a cluster with **1 worker**:

```powershell
docker compose up -d --scale ray-worker=1
```

Want more workers? Change only the scale value:

```powershell
docker compose up -d --scale ray-worker=3
```

---

### 3. Check that the cluster is alive

Open the dashboard:

```text
http://localhost:8265
```

Open Docker Desktop and check that:
- `ray-head` is running
- the `ray-worker` containers are running
- container logs do not show startup failures

You can also inspect the cluster from the head container:

```powershell
docker exec ray-head bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate 22971-ray && ray status"
```

---

### 4. Submit a smoke-test job

The file `smoke_test_job.py` in this folder:
- connects to the cluster
- launches a few tiny tasks on both the head and the worker
- prints output to the console
- writes a small file to `/workspace/smoke_test_output.txt`

Submit it from PowerShell:

```powershell
ray job submit --address http://127.0.0.1:8265 --working-dir . -- python smoke_test_job.py
```

What this does:
- uploads the current working directory to the cluster
- runs the script on the head node
- streams `stdout` and `stderr` back to your terminal

In this setup, with `ray job submit`, the driver process runs on the head container, not on your host machine.

If all is well, you should see:
- driver output
- task output from both node types
- a message saying the artifact file was written

---

### 5. Get the output file

The job writes:

```text
/workspace/smoke_test_output.txt
```

Inside Docker, `/workspace` is the mounted `head_workspace` folder on the **head** container.
So on the host machine, the same file appears here:

```text
head_workspace/smoke_test_output.txt
```

---

### 6. Inspect logs later

For longer jobs, submit without waiting:

```powershell
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python smoke_test_job.py
```

Ray prints a submission ID.
Use it later with:

```powershell
ray job status --address http://127.0.0.1:8265 <submission_id>
```

```powershell
ray job logs --address http://127.0.0.1:8265 <submission_id>
```

---

### 7. Stop the cluster

```powershell
docker compose down
```

## Virtual cluster use

Use this Docker setup to test distributed behavior on one machine:

- develop locally with `ray.init()`
- test cluster packaging and placement with `ray job submit` on this virtual cluster
- benchmark on a real physical cluster

This virtual cluster is useful for correctness and workflow testing, but not for trustworthy multi-node performance numbers, because all containers still share one physical machine.
