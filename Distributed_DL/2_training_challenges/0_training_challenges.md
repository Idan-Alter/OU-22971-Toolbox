# Distributed DL Unit 2 - Parallel Training Challenges

## Setup

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

## Manual data parallel mental model

- keep one model replica per rank
- give each replica a different shard of the minibatch
- run forward and backward locally on each replica
- synchronize gradients across replicas before the optimizer step
- apply the same update on every rank so the model weights stay aligned

**Note**:  PyTorch has a high level abstraction that hides the collective communication from the user: `DistributedDataParallel`. We won't be using it yet.


Before continuing, make sure you understand how `manual_data_parallel_demo.py` works.

---

## What this unit is really teaching

In this unit we intentionally use a small convnet and a lazy fake image dataset so three systems questions stay visible:

- which work is local compute?
- which work is synchronized communication?
- how does one local change become a global slowdown?

By the end of Unit 2, you should be able to look at one run and say which of these moved first:

- compute
- memory
- communication
- waiting

---

## Baseline run

Prediction before running:

- local batch preparation, local compute, and gradient synchronization should all be visible
- neither local work nor synchronized work should outweigh the other by an order of magnitude
- both ranks should show similar average step times
- this should become the healthy reference step for the rest of the unit

Run:

```bash
torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py
```

What to notice:

- the `config` block tells you the shape of the run
- the `training step flow` block shows the order of local work and synchronization
- the `average step summary` block gives you a balanced reference point
- the `memory estimate per rank` block gives you the baseline parameter, gradient, optimizer-state, input-batch, and activation estimates

Questions to ask:

- which parts of the step are local work?
- which parts of the step are synchronized work?
- do the two ranks already look broadly balanced?

---

## More data

Prediction before running:

- local work should grow before communication does
- the input-batch and activation estimates should increase
- the synchronized part of the step should still be present, but it should not be the first thing to dominate

Run:

```bash
torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --batch-size 256
```

What changed in the step:

- each rank now indexes a larger local image batch from `FakeData`

What signal should move first:

- compute first
- memory second

Questions to ask:

- did step time grow mostly because of more local work?
- did the memory estimate grow even though the communication pattern stayed the same?

---

## Larger network

Prediction before running:

- local compute should grow
- memory should grow in multiple categories at once
- synchronized work may also grow, because there are more gradient bytes to move

Run:

```bash
torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --base-channels 64 --conv-blocks 5
```

What changed in the step:

- each rank now holds and trains a wider, deeper convnet

What signal should move first:

- compute and memory first
- communication may grow as a secondary effect

Questions to ask:

- did local work grow, synchronized work grow, or both?
- which memory categories grew together?

---

## More communication

Prediction before running:

- local compute should stay close to baseline
- the synchronized part of the step should grow first

Run:

```bash
torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --extra-sync-mb 256
```

What changed in the step:

- after the real gradient sync, the script runs one extra dummy `all_reduce`

What signal should move first:

- communication

Questions to ask:

- did total step time grow even though the model and batch stayed the same?
- does the slowdown point to synchronized work rather than local work?

---

## One slow rank

Prediction before running:

- one rank should delay the other at synchronization
- waiting should grow more than local compute

Run:

```bash
torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --slow-rank 0 --sleep-before-sync 1.0
```

What changed in the step:

- rank `0` sleeps for one second before entering gradient synchronization

What signal should move first:

- waiting

Questions to ask:

- do both ranks slow down even though only one rank slept?
- where does one local delay become a global delay?

---

## Takeaway

More devices do not automatically mean faster training.

Each step is a tradeoff between:

- compute
- memory
- communication
- waiting

When you read the step summary and memory estimate, ask:

- did total step time grow?
- did memory estimates grow?
- did synchronized work grow?
- did one slow rank make another rank wait?

That is the diagnostic vocabulary we need going forward.

Unit 3 keeps the same image-shaped manual step and asks a new question:
what trace evidence confirms your diagnosis?

---

## Optional exercise

If you want extra practice before the profiler, rerun the toy step with one larger knob at a time and write down what bottleneck moved first:

```bash
torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --batch-size 512
```

```bash
torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --base-channels 64 --conv-blocks 6
```

```bash
torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --extra-sync-mb 512
```

```bash
torchrun --standalone --nproc_per_node=2 2_training_challenges/manual_data_parallel_demo.py --slow-rank 0 --sleep-before-sync 1.5
```

Try to classify each run as mostly:

- compute-bound
- memory-heavy
- communication-heavy
- waiting-heavy
