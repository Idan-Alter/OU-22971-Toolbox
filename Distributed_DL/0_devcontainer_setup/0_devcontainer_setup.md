# Distributed DL Unit 0 - Dev Container Setup

The goal of this unit is simple:

- get one Linux runtime for Part 3
- open the checked-in dev container from VS Code
- verify `torchrun`
- run one tiny DDP smoke test

**Note:** if you're using a **Linux machine**, you can skip this unit and create the `22971-td` environment locally. 

---

## Setup

1. Install **Docker Desktop** or another Docker engine that works with VS Code Dev Containers.

2. Install the `Dev Containers` **VS Code** extension.

---

## Dev container setup

### 1. Open the Part 3 folder in VS Code

From a terminal in the repo root:

```powershell
cd Distributed_DL
code .
```

**Important:** Make sure the top-level folder in the current VS Code window is `Distributed_DL`.
If you open the repo root instead, the Dev Containers extension may use a different config and build the wrong image.

### 2. Reopen the folder in the container

- open the Command Palette (`Ctrl+Shift+P`)
- run `Dev Containers: Reopen in Container`

On the first run, VS Code will build the image and then reopen the current folder inside the container.

Because you opened `Distributed_DL`, the dev container mounts that full folder into the container as `/workspace`.
That means the rest of Part 3 is available immediately after the container opens.

### 3. Verify the environment

Open a terminal inside the container and activate the environment:

```bash
conda activate 22971-td
```

Check PyTorch distributed:

```bash
python -c "import torch; import torch.distributed as dist; print(torch.__version__); print(dist.is_available())"
```

Expected result:

- a PyTorch version string
- `True`

Check the launcher:

```bash
torchrun --help
```

Expected result:

- the `torchrun` CLI help text prints

This local container is CPU-only, so `torch.cuda.is_available()` being `False` is expected.

### 4. Run the smoke test

From `/workspace` inside the container:

```bash
torchrun --standalone --nproc_per_node=2 0_devcontainer_setup/1_ddp_smoke_test.py
```

Expected behavior:

- torchrun logs some startup info
- the script prints something like:

```text
DDP smoke test summary
---------------------
rank 0
  local_rank=0 world_size=2
  backend=gloo device=cpu
  loss=1.170029 ddp_step=ok
rank 1
  local_rank=1 world_size=2
  backend=gloo device=cpu
  loss=1.327966 ddp_step=ok
```
**Notes:**
1. It's fine if the exact loss numbers differ slightly.
2. We'll go over the technical concepts in the collective communication unit.
---

## `.devcontainer` walkthrough

### `Dockerfile`

The Dockerfile:

- uses a Miniconda base image
- creates the full `22971-td` environment
- loads Conda automatically in interactive Bash shells

### `devcontainer.json`

`devcontainer.json` is the Dev Containers extension's config file for your development environment.
It tells VS Code how to build the image, what folder to mount, which folder to open, and what one-time setup command to run.

```json
"build": {
  "dockerfile": "Dockerfile",
  "context": ".."
}
```

`context: ".."` points Docker at the current folder.
That lets the Dockerfile see the part-level `environment.yml`.

#### `workspaceMount`

```json
"workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=cached"
```

You open the `Distributed_DL` folder in VS Code, and docker bind-mounts that same folder into the container.

This keeps the whole part editable and runnable inside the container.

#### `workspaceFolder`

```json
"workspaceFolder": "/workspace"
```

Once the container opens, `/workspace` becomes the working root for the whole part.

#### `postCreateCommand`

This runs one quick smoke check after the container is created:

1. load Conda into the shell
2. activate `22971-td`
3. import PyTorch
4. verify `torch.distributed` is available
