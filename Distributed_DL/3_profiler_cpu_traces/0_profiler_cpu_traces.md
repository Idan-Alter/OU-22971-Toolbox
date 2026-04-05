# Distributed DL Unit 3 - PyTorch Profiler: Probe the Toy Script with Traces

So far we've been timing runs and estimating memory footprints manually. The profiler is a better way to get the info we want, and much more.

---

## Setup

Same as in Unit 2:
1. Open Part 3 in VS Code:

   ```powershell
   cd Distributed_DL
   code .
   ```

2. Reopen the folder in the dev container:

   - open the Command Palette (`Ctrl+Shift+P`)
   - run `Dev Containers: Reopen in Container`

3. Activate the environment inside the container:

   ```bash
   conda activate 22971-td
   ```

4. Run all commands in this lesson from `/workspace` inside the container.

---

## Analyzing a trace

The main output of the profiler is a trace file. We'll go over one now.

1. Run:

   ```bash
   torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name baseline
   ```

   This file uses the same image-shaped training logic as the model in the previous unit, but logs the run with the profiler.

2. Open `/traces/baseline_rank0.json` at [https://ui.perfetto.dev/](https://ui.perfetto.dev/).

   This is the log of two training steps on rank 0.

3. Open the main python thread.

4. Observe the lengths of:

   - `next_batch`
   - `forward`
   - `loss`
   - `backward`
   - `grad_sync`
   - `optimizer_step`

   Within `next_batch`, also inspect:

   - `plan_batch_indices`
   - `scatter_batch_indices`
   - `index_batch`

   Diagnosis checks:

   - can you identify the local compute region?
   - can you identify the synchronization region?
   - is one phase much longer than the rest?

5. Open `/traces/baseline_rank1.json` at [https://ui.perfetto.dev/](https://ui.perfetto.dev/).

   This is rank 1's log for the same training step.

   Diagnosis check:

   - does the rank `1` trace look broadly similar to rank `0`?

Connection back to Unit 2:
This is the healthy reference step you established in Unit 2.
Every later trace should be compared against it.

## `profile_manual_data_parallel.py` walkthrough

What changed from `2_training_challenges/manual_data_parallel_demo.py`?

1. In `main()`, the last two training steps are run within the profiler context:

   ```python
   with profile(
       activities=[ProfilerActivity.CPU],
       record_shapes=True,
   ) as prof:
       ...
   ```

   The first `steps - 2` iterations are used as warmup and are not logged.

2. Within the profile context, certain parts of the training loop are marked with a `record_function` context manager:

   ```python
   with record_function("phase_name"):
       ...
   ```

   Everything run within this context will appear in the trace under `phase_name`.
   Batch preparation is grouped under `next_batch`, with `plan_batch_indices`, `scatter_batch_indices`, and `index_batch` nested inside it.
   In this version, `index_batch` includes direct `FakeData` lookup plus PIL-to-tensor conversion.

3. The trace is exported with `prof.export_chrome_trace()` and saved to a file.

**Note:** each rank runs its own profiler instance, and logs its own trace.

## Memory tracing

1. Run:

   ```bash
   torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name baseline_memory --profile-memory
   ```

   `--profile-memory` flips the relevant flag in the profiler definition and makes it log memory events.

2. Open `/traces/baseline_memory_rank0.json` in Perfetto and look for `[memory]` events mixed in with the operator timeline.

   Each `[memory]` event records one allocation or deallocation. The most useful fields are:

   - `Bytes`: the size of this one memory event
   - `Bytes > 0`: an allocation
   - `Bytes < 0`: a deallocation
   - `Total Allocated`: the amount of tracked memory still allocated after this event

Diagnosis questions:

- which phase seems to allocate the most memory?
- where does `Total Allocated` climb the fastest?
- where do the largest allocation events appear?
- does the bigger-batch or larger-network trace keep `Total Allocated` higher for longer than baseline?

---

## Bottleneck diagnosis with trace evidence

The table below is a quick guide for what kind of trace pattern usually supports a bottleneck diagnosis.

| "training challenges" diagnosis | Trace evidence to look for |
|---|---|
| `compute` | wider `forward` and `backward` regions |
| `communication` | wider `grad_sync` and `extra_sync` regions |
| `waiting` | visible imbalance around `sleep_before_sync` and synchronization |
| `memory` | more `[memory]` events and a higher `Total Allocated` curve |

---

## Exercise: resource probes

Run each command, compare its trace against `baseline_rank0.json` and `baseline_rank1.json`, and identify the main bottleneck.

```bash
torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name bigger_batch --batch-size 256
```

```bash
torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name larger_network --base-channels 64 --conv-blocks 5
```

```bash
torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name extra_sync --extra-sync-mb 256
```

```bash
torchrun --standalone --nproc_per_node=2 3_profiler_cpu_traces/profile_manual_data_parallel.py --trace-name slow_rank --slow-rank 0 --sleep-before-sync 1.0
```

For each trace, answer:

- which region got wider relative to baseline?
- is that wider region local compute, communication, or waiting?
- do both ranks still have the same shape?
- if not, where do they start to diverge?

Hints:

- `bigger_batch`: wider `next_batch`, `forward`, and `backward`; less change in `grad_sync`
- `larger_network`: wider `forward` and `backward`, and often heavier `grad_sync`
- `extra_sync`: a visible `extra_sync` region with local compute close to baseline
- `slow_rank`: a visible `sleep_before_sync` on rank `0` and waiting around synchronization on rank `1`
