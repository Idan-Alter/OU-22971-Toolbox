

# MLOps Unit 1 - Reproducible ML Pipeline Setup (Conda)

## Setup

Before starting, make sure the following are installed:

- **Visual Studio Code**
- **VS Code Python extension**
- **Miniconda** (recommended over Anaconda)

---

## 0. Start from a clean terminal (Command Prompt / PowerShell)

Make sure no Conda environment is active.

```bash
conda deactivate
```

## 1. Create a fresh environment

Create a new Conda environment with an explicit Python version.

```bash
conda create -n 22971-mlflow python=3.12
```

Do not install any packages yet.

---

## 2. Activate the environment

```bash
conda activate 22971-mlflow
```

After activation, the environment name appears at the beginning of the command prompt.

Example:
```
(22971-mlflow) C:\Users\yourname>
```

This indicates which Conda environment is currently active. All further commands apply only to this environment.

---

## 3. Install dependencies deliberately

Install core packages using Conda:

```bash
conda install numpy pandas scikit-learn matplotlib
```

Install pip-only packages via the active interpreter:

```bash
python -m pip install mlflow
```
#### Rules
- Do not install packages into `base`.
- Do not run `pip install` outside an active environment.
- **Install dependencies in one batch whenever possible** (avoids conflicts).
- **Rule of thumb for installers**:
  - Use **Conda** for numerical / compiled packages (NumPy, SciPy, PyTorch, etc.).
  - Use **pip (via `python -m pip`)** for pure-Python tools and ecosystem packages (MLflow, CLI tools, utilities).
---

## 4. Verify the environment

Run the pipeline:

```bash
python generate_data.py
python ml_pipeline.py
```

If this fails, fix the environment **before continuing**.

---

## 5. Export an environment file

Freeze the environment **after verification** (i.e., after `python ml_pipeline.py` works).

You have two options:

---

### Option 1: minimal export (`--from-history`)

```bash
conda env export --from-history > environment.yml
```

This file records **only what you explicitly asked Conda to install** (your "intent").

**Pros**
- Small, readable, stable.
- More portable across machines.

**Cons**
- Does **not** pin versions unless **you** explicitly pinned them at install.
- Does **not** reliably capture `pip`-installed packages (pip has no conda history).
---

### Option 2: full export (vanilla `conda env export`)

```bash
conda env export > environment.lock.yml
```

This exports the **exact state** of the environment:
- package versions
- build strings
- transitive dependencies
- pip section (if pip packages exist)

**Pros**
- Best for exact reproducibility on the same OS/platform.

**Cons**
- Large and hard to read.
- Less portable across OSes / machines.
- Often includes a machine-specific `prefix:` line (remove it before sharing).

---

### Option 3: Manual hybrid 

- Use **Option 1** (`--from-history`) as the main `environment.yml`.
- Augment using **Option 1** output:
  - **Pin versions** for the core dependencies.
  - Add `pip` history manually.

Example:
```yaml
name: 22971-mlflow
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.12
  - numpy=2.4.*
  - pandas=2.3.*
  - scikit-learn=1.8.*
  - matplotlib=3.10.*
  - pip
  - pip:
      - mlflow==3.8

```

### Option 4: the real-world case

Use **someone else's `environment.yml`**.

This is the most common scenario in practice:  
a project already provides an environment file, and your job is to **recreate it**, not design it.

**Pros**
- Zero decision-making.
- Fast onboarding.
- Highest consistency within a team or course.

**Cons**
- You inherit all design choices (good and bad).
- Debugging environment issues requires understanding how the file was created.
- You may need to adapt it for your OS / hardware.

**Rule of thumb**
> If an `environment.yml` exists, **use it first**.  
> Only create a new one if you are the first author of the project.

---


## 6. Prove reproducibility

Delete and recreate the environment from **environment.yml**:

```bash
conda deactivate
conda env remove -n 22971-mlflow
conda env create -f environment.yml
conda activate 22971-mlflow
python ml_pipeline.py
```

The output should match the original run.

---

> **Note:** `uv` is a modern, much faster, alternative to pip. Check it out: https://docs.astral.sh/uv/.
