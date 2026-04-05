# MLOps Unit 8 - Capstone Project Design Doc
## Manual monitoring -> optional retraining -> champion promotion (Metaflow + NannyML + MLflow)

This document specifies the MLOps capstone project.

---

## Goal

Build a small but real, **manually run** monitoring and retraining workflow and serve a production model.

You are expected to learn on your own how to use:

- **Metaflow**: A basic workflow orchestrator.
- **NannyML**: A library for model monitoring and drift detection.
- **Giskard** (optional): A library for pre-deployment model testing.

---

## Problem description: Green Taxi tip prediction

You are building a **tip prediction model** for NYC Green Taxi trips using TLC trip record data.

- **Input:** trip data available at prediction time (zones, time features, distance, passenger count, etc.).
- **Target:** `tip_amount` (regression).
- **Changes over time:** seasonality, rider behavior, pricing rules, geography, schema issues.

---

## Feature engineering

Implement a feature engineering step that:
- transforms raw columns into model-ready features:
  - time features from pickup datetime (hour, day-of-week, month)
  - location features (zone IDs, one-hot or target encoding, latitude and longitude, or embeddings if you want a more advanced option)
  - log/clip transforms for heavy-tailed numeric fields (distance, fare-like fields if used)
- produces a stable schema
- is reused consistently for training, evaluation, and inference


### Modeling freedom
Model choice and hyperparameter optimization are **up to you**.  
The focus of this project is building the control loop:
- correctness + reproducibility of the pipeline
- quality gates + decision logging (integrity / performance / promotion)
- clarity of evidence in MLflow (metrics + artifacts + decisions)

---

## Workflow overview

You will build a **manual** workflow you run from the command line. Each time you download a new dataset file (e.g., a new month of NYC Green Taxi data), you run the flow. The flow performs:

**new batch -> integrity gate -> feature engineering -> performance gate -> retrain? -> promote? -> redeploy server**

Everything logs to MLflow so decisions are auditable.

---

## Flow steps

### Step A - load data
- load reference dataset
- load batch dataset

### Step B - integrity gate (two layers)
Run integrity checks on the **raw** batch (before feature engineering), so you can catch schema/pipeline issues early.

#### Layer 1: hard rules (fail-fast)
Examples:
- required columns missing
- invalid datetimes (if required)
- target missing (if your evaluation needs labels immediately)
- impossible values beyond tolerance (negative trip_distance, dropoff before pickup)

If hard rules fail:
- log reasons to MLflow
- write `decision.json` with `action="reject_batch"`
- stop (no evaluation, no retrain)

#### Layer 2: NannyML checks (soft gate)
Examples:
- missingness spike vs reference
- unseen categorical values vs reference

Soft gate behavior:
- log results to MLflow (tables + summary metrics)
- do NOT stop automatically
- set a warning tag like `integrity_warn=true`
- optionally increase retrain likelihood

### Step C - feature engineering
Build a consistent model input table for **both** reference and batch:
- apply the same filters and transformations
- produce the same feature columns (stable schema)
- handle missing values and types consistently
- log the feature spec (e.g., a JSON list of feature names + dtypes) to MLflow so you can debug schema issues.


### Step D - load champion model
Try:
- `models:/<model_name>@champion`

Bootstrap case (no champion yet):
- run feature engineering on reference
- train initial model, register it, set `@champion`

### Step E - model gate (performance)
Evaluate champion on the **engineered** batch features:
- RMSE (or MAE)
- optionally additional diagnostics (residual distribution, slice performance, SHAP)

Compute:
- `rmse_champion`
- `rmse_baseline` (defined next)
- `rmse_increase_pct`

Decide whether to retrain (explore the NannyML docs for decision rules):
- `retrain_needed=true/false`
- `retrain_reason="..."`

Log:
- metrics above
- `retrain_recommended` as tag
- `decision.json` always


### Step F - retrain (conditional)
If retrain is needed:
- build a training set using engineered features from an updated time window (rolling or expanding)
- train a candidate model (model + hyperparameter choices are up to you)
- evaluate the candidate on the SAME engineered evaluation batch.
- log candidate metrics and artifacts to MLflow

### Step G - candidate acceptance (gate for promotion)
Candidate should be promoted only if:
- it passes gates (no hard failures; optional: no critical integrity warnings)
- it beats champion by a meaningful margin
- it does not win because of obvious overfitting or leakage

Then register and maybe promote and redeploy the server.

---

## Model registry logic

### Model naming strategy
Pick one registered model name:
- `green_taxi_tip_model`

Every training run (initial or retrain) creates a new **version** under this name.

### What "register" means (mechanics)
When you train a candidate:
1) `mlflow.sklearn.log_model(..., name="model")`
2) register it as a new model version under `green_taxi_tip_model`
3) attach tags:
   - `role=candidate`
   - `trained_on_batches=...`
   - `eval_batch_id=...`
   - `validation_status=pending|rejected|approved`
   - `decision_reason=...`

### Bootstrap (no champion exists yet)
If `@champion` doesn't exist:
- register the initial model
- set alias `@champion` -> that version
- tags:
  - `role=champion`
  - `promotion_reason=bootstrap`

### Promotion criteria (when to flip `@champion`)
Promote only if ALL hold:

**P1: Candidate evaluation is valid**
- evaluation metrics exist (no missing labels)
- evaluation dataset is logged (lineage)

**P2: Candidate beats champion (meaningfully)**
- `rmse_candidate < rmse_champion * (1 - min_improvement)`
- choose `min_improvement` ~ 1-2% to avoid churn

> **Grading default:** set `min_improvement = 1%` as a parameter and log it.

**P3: Stability check (avoid one-batch overfit)**
Pick one:
- evaluate on two slices (new batch + stable reference slice)
- OR require "does not regress the reference slice by > Y%"
- OR require persistence (candidate wins on two consecutive batches)

**P4: Integrity sanity**
- no hard integrity failures (obvious)
- optional: block promotion if `integrity_warn=true` and warning is severe

### Promotion mechanics (alias flip)
Promotion is:
- set alias `@champion` to candidate version

Also tag:
- old champion: `role=previous_champion`, `demoted_at=...`
- new champion: `role=champion`, `promoted_at=...`, `promotion_reason=...`

### Anti-footgun rules
- Never promote without evaluation metrics.
- Never promote if the batch was rejected by hard integrity rules.
- Always log `decision.json` describing:
  - criteria used
  - metric values
  - final decision

---

## 7) Stretch goals (optional)

### Stretch A - automation / event triggering
Goal: Trigger the flow automatically when a new data slice arrives.

Hints:
- Keep the Metaflow flow manual and pure; add automation *around* it, not inside it.
- Write a tiny polling script that watches a folder for new files and invokes the flow; schedule it with `cron`.
- For real systems, explore event-driven triggering in the Metaflow docs (production backends only).

### Stretch B - Giskard model scanning
Optional extra gate:
- after evaluation, run a vulnerability scan (slice failures, robustness issues, etc.)
- treat scan results as an additional "do not promote" condition
- log the HTML report to MLflow

Hint:
- Explore Giskard docs for details

### Stretch C - web deployment
Deploy a containerized model to a cloud service.

---

## 8) Deliverables

1. **Link to a GitHub repo** containing:
   - `README.md` with: setup steps, the exact command(s) to run, and where to look in the MLflow UI (experiment name).
  
2. **Short video (5-10 min)** showing:
   - **Code walkthrough:** the Metaflow flow structure (steps) and where these happen in code:
     - integrity gate (hard rules + NannyML)
     - evaluation gate (champion metrics)
     - retrain + promotion decision logic
   - **MLflow UI inspection:** open the run(s) and show *actual evidence*, including:
     - key metrics (e.g., `rmse_champion`, `rmse_increase_pct`, and `rmse_candidate` if retraining ran)
     - the `decision.json` artifact
     - integrity artifacts/reports (NannyML output) and any `integrity_warn` tagging
     - tags that reflect decisions (e.g., `retrain_recommended`, `promotion_recommended`)
   - **Evidence of retraining and promotion (automatic within the run):**
     - show a run where the workflow **decides and executes** retraining without manual intervention once started
     - show the newly registered model version in MLflow Model Registry
     - show the `@champion` alias (or equivalent) being updated as a result of the run
   - **Inference demo (offline, Unit 6):**
     - run batch inference on a new data slice and log predictions as an MLflow artifact (e.g., `predictions.parquet`)

### Required demo pattern

The video **must include three separate runs** of the workflow:

1. **Baseline run (no action taken)**  
   - The workflow completes normally.  
   - No retraining is triggered.  
   - No promotion occurs.  
   - Evidence shown in MLflow:
     - champion evaluation metrics
     - `retrain_recommended=false`
     - `promotion_recommended=false`
     - `decision.json` explaining the outcome.

2. **Retrain + promotion run (automatic within the flow)**  
   - The workflow detects degradation and **decides** to retrain.  
   - Retraining executes automatically once the run starts.  
   - A candidate model is evaluated and **promoted**.  
   - Evidence shown:
     - candidate vs champion metrics
     - new model version registered in the MLflow Model Registry
     - `@champion` alias updated
     - decision tags and `decision.json` justifying promotion.  


3. **Failure + resumption run (workflow robustness)**  
   - You intentionally raise an exception in a mid-flow step  
     (e.g. retrain step).  
   - The workflow fails.
   - You then fix the issue and resume flow execution from the failed step.
   - Evidence shown:
     - the flow resumes from the failed step (not from the beginning)
     - previously completed steps are not re-executed
     - MLflow logging handles this gracefully (failed run + new run)
     - final decisions and artifacts reflect the successful resumed execution

---

## Appendix: Metaflow primer

### Why an orchestrator?

Once your script becomes a **workflow** (gates, branching, optional retrain/promotion), you want:
- explicit step boundaries (integrity vs eval vs retrain)
- reproducible runs
- the ability to re-run from a step without redoing everything
- a clean audit trail of decisions

**Metaflow** is an accessible solution.


### Core structure
- `FlowSpec` - define a workflow as a Python class.
- `@step` - define a pipeline step.
- `self.next(...)` - explicitly define control flow between steps (including branching).

### Artifacts (step outputs / run-scoped state)
- Assign attributes on `self` (e.g. `self.metrics`, `self.batch_df`) to pass outputs between steps.
- `self.*` artifacts are **automatically persisted when a step finishes successfully** and restored in downstream steps.
- Only instance variables on `self` are persisted as artifacts; normal local variables are not.

### Branching
- Conditional branching using `self.next(step_a if condition else step_b)`.


### Execution & iteration
- Run locally: `python <flow>.py run`
- Resume after failure or from a chosen step: `python <flow>.py resume [stepname]`

### Conceptual distinction (important)
- **Metaflow artifacts:** persisted step outputs within a single run (checkpointed at step boundaries).
- **MLflow logging:** persistent experiment records and model registry across runs.

Metaflow can do a lot more, but you are not expected to know it deeply. Specific advanced features that are out of scope of this project:
- cloud backends, scheduling, event triggering
- parallelization/scale features
- advanced decorators beyond `@step`


---

## Metaflow starter
Save this code block in `flow_starter.py` and use it as the basis for your flow.

```python

from metaflow import FlowSpec, Parameter, step


class MLFlowCapstoneFlow(FlowSpec):
    reference_path = Parameter("reference-path")
    batch_path = Parameter("batch-path")
    model_name = Parameter("model-name", default="green_taxi_tip_model")

    @step
    def start(self):
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step
    def load_data(self):
        self.ref, self.batch = load_reference(self.reference_path), load_batch(self.batch_path)
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        ok, report = run_integrity_checks(self.ref, self.batch)  # hard + NannyML
        self.next(self.load_champion if ok else self.end)

    @step
    def load_champion(self):
        # TODO: Add relevant steps and flow logic.
        pass


if __name__ == "__main__":
    MLFlowCapstoneFlow()

```
