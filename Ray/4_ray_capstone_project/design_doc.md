# Ray Unit 4 - Capstone Project Design Doc
## TLC-backed per-zone recommendations under skew

This document specifies the Ray capstone project.

---

## Goal

Build a small distributed system that looks at recent NYC TLC Green Taxi pickup demand in each zone and gives one simple recommendation about what that zone needs next:

- `NEED`: this zone looks unusually busy right now
- `OK`: this zone looks normal right now

This is a toy dispatcher-style, human-facing support tool. The system watches demand over time, zone by zone, and keeps producing updated recommendations as new data arrives.

To make that concrete, the project uses historical taxi data as a replay. The system walks through the data in small fixed time windows and, for each zone, decides whether demand looks elevated or normal.

This capstone is mainly about distributed-systems behavior, not forecasting sophistication. The interesting part is what happens when some zones take longer than others, some work finishes late, or the same write is retried. You should show a clear runtime architecture with actor-owned state, scoring tasks, and deterministic rules for when a time window is considered finished even when some zones are slow.

Fault tolerance is also a required goal. The runtime should treat retries, duplicate delivery, and late results as normal possibilities, and any state mutation that matters for the replay should be written idempotently.

Students should also compare two execution strategies on the same replay:

- a blocking baseline where scoring tasks return their decisions to the controller and the controller performs the accepted state updates
- an asynchronous controller where scoring tasks report decisions to `ZoneActor`s and the driver closes ticks by polling actor readiness, meaning it repeatedly asks actors whether they already have a decision for the current `tick`

The main evaluation questions are:

- does that asynchronous controller handle uneven zone completion more cleanly than the blocking baseline?
- does the system preserve clear output semantics even when per-zone completion order varies?
- can the runtime show bounded concurrency, deterministic fallback behavior, and useful observability on the same TLC replay window?
- does the runtime remain correct when decision writes are retried, duplicated, or arrive after a time window has already been finalized?

---

## Problem description: Green Taxi recommendation replay

You are building a replay-based recommendation system using real NYC Green Taxi trip data.

Use two adjacent monthly Green Taxi parquet files:

- a reference month
- a replay month

The reference month teaches the system what "normal" looks like.

The reference month is used to:

- select the **active pickup zones**: zones with enough data to allow prediction.
- build simple per-zone historical baselines
- define what "normal" demand looks like for different times of day and days of week


The replay month is the month the system walks through step by step to generate recommendations and is used to:

- drive a time-ordered stream of actual pickup pressure
- compare blocking and asynchronous execution on the exact same data
- measure how skew and late zones affect runtime behavior

As the replay month is processed, the system moves forward in fixed 15-minute windows. In the rest of this document, each window is called a `tick`.

At each `tick`, each active zone in that selected subset should produce one recommendation for the next `tick`:

- `NEED`: this zone looks busier than its recent norm
- `OK`: this zone does not look elevated right now

The recommendation rule should stay intentionally simple. This capstone is not about forecasting sophistication. You may use a lightweight rule based on recent observed demand, a zone baseline, or another equally simple thresholded signal derived from the reference data.

---

## Data preprocessing

- validate that the two files are adjacent months from the same year
- aggregate pickup demand into fixed 15-minute ticks
- select a deterministic subset of the busiest pickup zones from the reference month; these become the active zones used throughout the replay
- build a reference baseline table by `zone_id`, `hour_of_day`, and `day_of_week`
- build a replay table with one row per `(zone_id, tick)`
- keep preprocessing as a separate `prepare` step so the runtime can consume prepared assets rather than raw parquet

Validation requirements:

- include one pandas cross-check that confirms the prepared replay counts match a direct grouped reference calculation on a sample window
- make active-zone selection deterministic under a fixed seed and fixed input files

---

## System overview

The system should follow this control loop:

**reference-month prep -> actor initialization -> tick replay -> per-zone scoring -> tick finalization under partial readiness -> metrics and artifacts**

Conceptually:

- the **controller** is the main driver loop: it advances the replay, marks ticks as active, launches zone work, and decides when a `tick` is finished
- the reference month provides the baseline information that driver loop starts from
- the replay month becomes a time-ordered stream of actual zone demand
- each zone owns its own mutable state through an actor
- in blocking mode, scoring tasks return decision payloads to the controller and the controller writes accepted decisions into actors
- in async mode, scoring tasks report decisions to actors and the driver polls actor readiness before closing a tick
- the driver compares blocking and asynchronous execution on the same replay
- slow zones should not force the asynchronous controller to stall forever

The project is mainly about:

- actor-owned state
- per-zone scoring tasks
- skew and uneven completion
- bounded concurrency
- deterministic partial-readiness behavior
- fault tolerance and idempotent state writes
- execution strategy versus output semantics

---

## Runtime architecture

Required runtime pieces:

### `ZoneActor`

One actor per active zone.

Each `ZoneActor` owns the mutable state for that zone. Keep that state focused on what the actor must remember to deduplicate and safely finalize tick-level decisions. For example:

- recent observed demand history
- current active tick state
- the currently reported decision for the active tick, if any
- a keyed history of accepted decisions by `tick_id`
- counters for duplicate reports, late reports, and fallbacks

The actor may also keep a prepared replay partition or replay cursor if that is convenient for the implementation, but that is an implementation detail rather than the main teaching point.

Only `ZoneActor` is allowed to mutate durable zone state for that zone.
In blocking mode, the controller decides which accepted writes to send into each actor.
In async mode, the actor itself owns any reported decision for the active tick and decides whether an incoming report is on time, duplicate, or too late to accept.

### Scoring task

Use a Ray remote task for per-zone scoring.

Input:

- one `ZoneSnapshot`

Output:

- `zone_id`
- `tick_id`
- binary decision: `NEED` or `OK`
- task latency or timing metadata

Mode-specific reporting:

- in blocking mode, the task returns this decision payload to the controller
- in async mode, the task reports this decision payload to the relevant `ZoneActor` for that zone and `tick`

Requirements:

- scoring logic should be deterministic from the snapshot input
- task must be retry-safe
- task should receive only the snapshot it needs
- task must not write directly to final artifacts or global metrics
- in async mode, the report path to the actor must be safe under duplicates and late arrival

### Driver loop

Use one driver process to advance ticks.

Common responsibilities:

- marking the current `tick` as active for each `ZoneActor`
- collecting snapshots from all `ZoneActor`s
- launching per-zone scoring tasks
- logging timing and skew metrics

Blocking mode responsibilities:

- wait for all task results for the current `tick`
- decide which task results become accepted writes
- write those accepted decisions into the relevant actors

Async mode responsibilities:

- observe which actors have a reported decision for the current `tick`
- keep zone work bounded in flight with a config such as `max_inflight_zones`
- finalize each tick using an explicit partial-readiness policy
- close each tick and tell actors when to accept fallback instead of waiting

Here, `polling` means the driver periodically asks actors for their current tick status instead of waiting on task-return references.

Requirements:

- implement a blocking baseline that waits for all zone results
- implement an asynchronous controller that closes ticks from polled actor readiness instead of waiting for every zone
- if zone work is late or missing, apply a deterministic fallback policy instead of waiting forever
- make any accepted state write idempotent at the `(zone_id, tick_id)` level or an equivalent stable key
- make the actor-side handling of duplicate or late task reports explicit and observable

### Skew model

The system must simulate skew so the execution difference is visible.

You may do this by:

- adding artificial sleep to a configurable fraction of zones
- making some zone tasks intermittently slower than others
- optionally combining organic replay variability with artificial delay

The blocking baseline should show strong sensitivity to the slowest zones. The asynchronous controller should show more predictable tick completion behavior under the same skew conditions.

---

## Flow steps

### Step A - load the monthly datasets

- read the reference-month and replay-month parquet files into local Python dataframes

### Step B - build the prepared assets

- identify the busiest active pickup zones from the reference month
- aggregate reference demand into 15-minute ticks
- build the zone/time baseline table from the reference month
- aggregate replay-month pickups into one row per `(zone_id, tick)`
- write prepared assets for the runtime

### Step C - initialize the runtime

- create one `ZoneActor` per active zone
- give each actor ownership of its own prepared replay partition
- initialize any global run configuration and output locations

### Step D - advance one replay tick

At the start of each tick:

- tell each actor that this `tick` is now active
- ask each actor for the snapshot needed for the next recommendation
- keep the snapshot minimal and derived from actor-owned state

### Step E - run per-zone scoring

Blocking mode:

- submit each zone snapshot to a scoring task
- collect all task returns in the controller for the current `tick`

Async mode:

- submit each zone snapshot to a scoring task
- have each scoring task report its result to that zone's actor for the current `tick`

### Step F - finalize the tick under partial readiness

Blocking mode:

- wait until all task results for the current `tick` have been returned to the controller
- close the `tick` only after the controller has a complete result set

Async mode:

- check which actors have already received a report for the current `tick` by asking actors for their current tick status
- decide whether the policy says to keep waiting or to close the `tick`

You must define and log a deterministic policy for late zones.

Examples:

- finalize after `completion_fraction` of zones finish
- finalize when `tick_timeout_s` expires
- apply the default fallback policy `always_previous` to late zones

Policy requirements:

- must be explicit in config
- `fallback_policy` should default to `always_previous`
- must be visible in logs and artifacts
- must behave the same way on repeated runs with the same inputs and seed

### Step G - close the tick in each actor

Blocking mode:

- have the controller write the accepted decision for each zone into its actor
- ensure duplicate accepted writes for the same zone and tick are safe to replay

Async mode:

- ask each actor to finalize the current `tick` using either its reported decision or the fallback policy
- ensure duplicate reports for the same zone and tick are safe to replay
- late results that arrive after finalization should be logged and ignored by the actor

In both modes:

- update actor state needed for the next tick

### Step H - finalize artifacts

- aggregate latency, tick-level metrics, and actor-accepted decisions
- compare blocking and asynchronous runs on the same replay window and configuration

---

## Metrics and artifacts

At minimum, log:

- per-zone task latency
- mean zone latency
- max zone latency
- max / mean latency ratio
- number of zones completed before tick finalization
- number of zones that used the fallback policy
- total tick latency

For async mode, also log:

- number of late task reports ignored
- number of duplicate task reports ignored

---

## Failure model and invariants

Assume the runtime may experience retries, duplicate delivery, and late arrivals of completed zone work.

Required invariants:

- every durable actor write must be idempotent
- writes should be keyed by a stable identifier such as `(zone_id, tick_id)`
- in blocking mode, the controller must not write the same accepted decision twice for the same zone and tick
- in async mode, duplicate decision delivery must not update the same zone twice for the same tick
- in async mode, a `ZoneActor` must not accept reports for ticks that are inactive or already closed
- late results that arrive after tick finalization must not overwrite the accepted tick outcome
- final metrics and artifacts should not double-count duplicated zone results
- final artifacts should be derived from actor state, not from raw task completions alone

---

## Interfaces and defaults

Keep the project organized around Python scripts rather than around a CLI-spec-first interface.

A simple script layout is:

- `prepare.py` to read the reference and replay parquet files and write the prepared assets
- `run.py` to load the prepared assets, run the replay, and support `blocking`, `async`, or `stress` mode

Suggested runtime config fields:

- `n_zones`
- `tick_minutes`
- `max_inflight_zones`
- `tick_timeout_s`
- `completion_fraction`
- `slow_zone_fraction`
- `slow_zone_sleep_s`
- `fallback_policy`
- `seed`

Default behavior:

- `fallback_policy` should default to `always_previous`
- students should choose, implement, and document a consistent first-use edge-case policy for zones that do not yet have a previous accepted decision


---

## Deliverables

1. A GitHub repo containing:

- Python scripts for preparation and replay execution, such as `prepare.py` and `run.py`
- a short `README.md`

The README should contain:

- exact setup commands
- exact commands for running the Python scripts locally
- the Docker-cluster run command via `ray job submit`
- a short explanation of the decision rule
- a short explanation of the partial-readiness policy

2. Output artifacts, for example:

- `run_config.json`
- `metrics.csv`
- `latency_log.json`
- `tick_summary.json`

3. A **short video (5-10 min)** showing:

- **Code walkthrough**, specifically:
  - the actor / task split in code
  - the blocking baseline
  - the asynchronous controller
  - the effect of skew
  - the fallback behavior for late zones
- **Live demo execution** on a virtual Docker-based cluster
- **Analysis and discussion** of the output artifacts

---

## Required demo pattern

The demo must include three separate runs.

### 1. Blocking baseline

- scoring tasks return decision payloads to the controller
- the controller waits for all results and writes accepted decisions into actors

Expected behavior:
- tick latency is dominated by the slowest zones
- skew hurts visibly

### 2. Async controller

- scoring tasks report decisions directly to actors
- the driver polls actor readiness and closes ticks under the configured policy

Expected behavior:
- lower sensitivity to stragglers
- more controlled tick-completion behavior

### 3. Skew stress test

- increase skew aggressively while keeping the same replay structure
- compare the blocking and async behaviors under the same harsher conditions

Expected behavior:

- blocking degrades sharply
- the async controller still progresses cleanly with explicit fallback usage

---

## Stretch goals (optional)

### Stretch A - delayed arrivals

Extend the runtime so some zone-level effects take time to become visible in future ticks.

You should deliberately withhold some input data instead of making it visible to the scoring logic immediately. That withheld data should only become available after one or more later `tick`s.

Show and explain:

- what data is being withheld
- how long it is withheld before release
- how the delayed release changes the zone's later snapshots or decisions
- how the actor and driver handle the fact that "real" demand information can appear after the system has already moved on

The goal is to reason about delayed information, not just delayed task completion.

### Stretch B - adaptive load balancing through zone subactors

Create a small set of repeat straggler zones that become slow again and again across many `tick`s, then have the zone actor apply explicit subactor-creation logic for those zones.

The zone actor should detect or track repeated straggler behavior and decide when to create one or more helper actors or subactors for that zone.

Show and explain:

- how repeat straggler zones are created or identified
- what rule the actor uses to create subactors
- what work moves from the main zone actor to the helper actor or subactor
- how this changes execution behavior for hot zones
- how you preserve the same final per-zone decision semantics even after adding subactors
