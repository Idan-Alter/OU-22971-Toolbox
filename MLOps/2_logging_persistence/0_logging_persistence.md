

# MLOps Unit 2 - Experiment Logging and Artifact Persistence

## Setup

Activate the conda environment created in **Unit 1**:

```bash
conda activate 22971-mlflow
```

## Logging
Right now, `ml_pipeline.py` writes its outputs to fixed filenames, which means the only artifacts we keep are the outputs from the most recent run.
 
What if we want to inspect a run from a month ago or compare models trained on different versions of the data?

That's the problem experiment logging is meant to solve.

#### Minimal solution: run a logging wrapper.

`logging_wrapper.py` essentially automates the following:

1. **Create a new, timestamped run directory**

   Example:
   ```text
   runs/
     2026-01-16_11-42-03/
   ```

2. **Change the working directory to that folder**

   From this point on, any relative output path points *inside the run*.

3. **Execute `ml_pipeline.py` from its original location**

   Conceptually:
   ```bash
   python ../1_conda_environments/ml_pipeline.py \
     --data ../1_conda_environments/data/clean.csv
   ```

4. **Let all generated outputs land in the run directory**

   After the run:
   ```text
   runs/2026-01-16_11-42-03/
     confusion_matrix.png
     stdout.log
   ```

   No files are overwritten.
   Previous runs remain intact.

5. **Capture stdout/stderr to a log file**

   Example contents of `stdout.log`:
   ```text
   Best parameters: {'C': 1.0, 'penalty': 'l2'}
   Test accuracy: 0.83
   ```

   Console output becomes part of the run artifact.

Nothing about the experiment logic changes.
Only *where the side effects go*.

**Pros**
- No changes to the original Python code.
- Zero new tooling or dependencies.
- Prevents silent overwrites of past results.

**Cons**
- Fragile by construction: breaks if folder layout and script names change.
- Debugging is brittle: reproducing a failure often requires the exact same filesystem state.
- No support for comparison or querying: folders must be inspected manually.


**Afterthought: There has to be a better way to do this.**

## Artifact persistence

Right now, our scripts output metrics and plots. What if we want to use the fitted model on new data?


---

#### Minimal solution: pickle the model

Python objects have no standard `.save()` method, so persistence is handled via `pickle`.

To save our models, we need to change `ml_pipeline.py`:

After `GridSearchCV` finished fitting, save the best model:  
```python
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)
```

Now each run directory contains not just logs and plots, but the actual trained artifact.

Example:
```text
runs/2026-01-16_11-42-03/
  confusion_matrix.png
  stdout.log
  model.pkl
```
---

We can reload it later:
  ```python
  with open("model.pkl", "rb") as f:
      model = pickle.load(f)
  ```

**Pros**
- Trivial to use: works for most Python objects.
- Preserves the exact trained model.
- No additional tooling or formats required.

**Cons**
- Fragile across time:
  - breaks with Python version changes
  - breaks with library upgrades
- Opaque:
  - no embedded metadata (data version, parameters, metrics)
  - no notion of "best" or "latest"
- Unsafe by design:
  - loading pickles executes arbitrary code
- Python-only:
  - cannot be inspected or used outside Python
- Some models cannot be pickled at all (e.g. distributed or remote objects).

**Is there a better way to do this as well?**

## Experiment tracking systems

Experiment tracking systems solve these problems in a more structured way.

We'll use MLflow going forward, but it's not the only option. There are several popular systems in this space; Weights & Biases is probably the most popular.


