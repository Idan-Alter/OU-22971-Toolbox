# Distributed DL Unit 6 - Capstone Project Design Doc
## SimCLR-like distributed training with manual ResNet sharding

This document specifies the Distributed DL capstone project.

---

## Goal

Build a distributed training system that learns from paired image augmentations in a SimCLR-like way, manually shards a small ResNet across devices, and studies its behavior through manual batch-size sweeps and manual trace analysis.

The capstone should prove that you can:

- implement a distributed contrastive training loop
- manually partition a CNN into sequential stages on different ranks
- reason about compute, memory, communication, waiting, and stage imbalance
- capture and inspect per-rank traces
- use trace evidence to justify manual tuning decisions
- rerun the job with a new configuration and explain why it improved or failed to improve

This capstone is about distributed-systems behavior. The actual semantic quality of the learned embeddings is immaterial.

---

## Dataset

Use an ImageNet-like synthetic workload based on `torchvision.datasets.FakeData` to train a `resnet18`-based feature extractor with a SimCLR-like objective.

Use deterministic `FakeData` with these defaults:

- `image_size=(3, 224, 224)`
- `num_classes=1000`
- a fixed `dataset_size`
- a fixed `seed`

Labels from `FakeData` are not used in this task. They only exist because `FakeData` produces image-label pairs.

As part of the training algorithm, create two augmented views for each loaded image. Use this fixed augmentation recipe:

- `RandomResizedCrop(224)`
- `RandomHorizontalFlip`
- `ColorJitter`
- `RandomGrayscale`

The two generated views are called a **positive pair**.

## ML task

This project is a distributed-systems implementation problem, so it does not require a deep understanding of self-supervised learning. You only need to understand the SimCLR-style contrastive loss at a high level:

- draw a minibatch of source images
- create two different augmented views from each source image; these two views form a **positive pair**
- feed all views through the CNN and get one embedding vector per view
- calculate cosine-style similarities between embeddings
- for one specific view `X`, define a classification problem in which:
  - each other view in the minibatch is a candidate class
  - the correct class is `X`'s positive pair, `Y`
  - all other views act as negatives
  - the logits are `X`'s similarity scores against the candidate views
  - self-similarity is excluded from the candidate set
- apply cross-entropy loss to those logits: take the `Y`-th entry of `-log(softmax(logits))`
- average the per-view losses to get the contrastive loss

Calculating this loss in a distributed system introduces complications:

1. The similarity calculation requires embeddings from the full global minibatch to be materialized where the loss is computed.
2. The softmax couples the gradients of many views together, which complicates the backward pass when ranks exchange detached tensors. To keep the project focused on distributed-systems behavior, we will simplify the loss calculation later in the design and use an approximation that avoids this complication.

---


## System overview

You should implement this workflow:

**prepare deterministic dataset metadata -> launch with `torchrun` -> train a distributed contrastive learning model with a sharded encoder -> capture traces -> manually analyze -> manually choose the next batch size -> rerun -> compare**

## Detailed system specification

Implement a manually sharded, two-stage training system, not just a standard data-parallel loop.

Sharding the model into stages will be done sequentially such that the first few layers will form stage 0, and the rest will form stage 1. For example:
  - **stage 0**: `conv1`, `bn1`, `relu`, `maxpool`, `layer1`, `layer2`
  - **stage 1**: `layer3`, `layer4`, `avgpool`, flatten, projection head
It is up to you to decide the split point, but you should aim for even workloads on the ranks that receive the different stages.

**Note:** The implementation must not use high-level PyTorch distributed abstractions such as `DistributedDataParallel`, `DistributedSampler`, pipeline helpers, or autograd-aware distributed wrappers. Use only low-level `torch.distributed` communication primitives.

### Topology

- initialize `torch.distributed` on the default `world_group` with an even number of ranks, at least `4`
- define a logical `pair_group(k) = (2k, 2k+1)`, where each pair forms one sharded model replica
- define `stage0_group` as all even ranks; these ranks own stage 0 and perform stage-0 gradient synchronization
- define `stage1_group` as all odd ranks; these ranks own stage 1, compute the contrastive loss, gather embeddings, and perform stage-1 gradient synchronization
- initialize the baseline `resnet18` and split it
- broadcast stage-0 parameters within `stage0_group` and stage-1 parameters within `stage1_group` for one-time replica alignment
- initialize an optimizer on each rank

### Forward pass

- on each even rank, materialize one local source-image batch from deterministic `FakeData`
- apply the chosen random augmentation recipe twice per source image
- interleave the two views into one local tensor of shape `[local_view_batch, 3, 224, 224]`
- run stage 0 on the even rank
- explicitly transfer the stage output activation tensor to the paired odd rank inside the relevant `pair_group`
- on the odd rank, receive the boundary activation into a fresh tensor, mark it as requiring gradients, and run stage 1
- `all_gather` embeddings across `stage1_group`
- split the gathered embeddings into two groups:
  - local embeddings (produced on the current rank), which must remain attached to the local autograd graph and therefore receive gradients during backpropagation
  - remote embeddings (received from other ranks), which participate in the loss as fixed values and are not differentiated through on this rank
- compute **an approximation** to the SimCLR contrastive loss:
  - compute similarity scores of all local embeddings with all other embeddings
  - compute the per-view loss **for local embeddings only**
  - approximate the global loss with the average of local view losses

### Backward path

- backpropagate the loss through stage 1 on the odd rank
  - **Note:** This will calculate an approximation of the gradients, since remote embeddings were detached before localization on the current rank.
- after `loss.backward()`, extract the gradient of the boundary activation tensor received from the paired even rank
- explicitly send that boundary gradient from the odd rank back to the even rank
- on the even rank, continue stage-0 backward with `boundary_activation.backward(returned_gradient)`
- synchronize stage-0 gradients only within `stage0_group`
- synchronize stage-1 gradients only within `stage1_group`
- average gradients within each stage group so replicas apply the same optimizer step
- run the optimizer step on each rank


### Profiling path

The script must support a profiler-enabled run that exports one trace JSON per rank.

Use these required profiler span names so manual analysis and optional automation are consistent:

- `prepare_views`
- `stage0_forward`
- `send_boundary`
- `recv_boundary`
- `stage1_forward`
- `gather_embeddings`
- `loss_calculation`
- `send_boundary_grad`
- `recv_boundary_grad`
- `stage0_backward`
- `grad_sync_stage0`
- `grad_sync_stage1`
- `optimizer_step`

The script should also save a compact metrics summary that includes at least:

- `local_batch_size`
- `global_batch_size`
- `images/s`
- per-rank step time
- loss

### Implementation hints and build order

- first define the communication structure and print it once at startup: `world_group`, each rank's logical `pair_group`, `stage0_group`, and `stage1_group`
- get stage initialization and one-time replica alignment correct before continuing to the training step
- get the pair-local activation handoff working first: even rank runs stage 0, then sends the boundary activations to the paired odd rank
- on the odd rank, receive the boundary activation into a fresh tensor, mark it as requiring gradients, run stage 1, `all_gather` across `stage1_group`, and compute the contrastive loss using the local live embeddings and the detached remote embeddings
- get the boundary-gradient return path correct next: after `loss.backward()`, send the returned boundary gradient back to the even rank and continue stage-0 backward with `boundary_activation.backward(returned_gradient)`
- only after that works, add stage-local gradient synchronization with collectives inside `stage0_group` and `stage1_group`
- point-to-point communication belongs inside each `pair_group` for activations and returned boundary gradients; collectives belong inside the stage groups for `all_gather` and gradient averaging
- use fixed microbatch shapes and `drop_last=True` at first so send and receive buffer management stays simple
- add the required profiling path only after the forward, backward, and synchronization paths are correct
- while bringing the system up, add simple validation checks:
  - verify that every participating rank enters sends, receives, and collectives in the same order so the job does not deadlock
  - assert that boundary activation tensors and returned boundary gradients have the expected fixed shapes on both sides of the pair
  - assert that loss values, embeddings, and gradients remain finite before the optimizer step
  - after stage-local gradient synchronization and the optimizer step, compare one or two parameter tensors across each stage group once to confirm the replicas stayed aligned

### Manual batch size sweep and trace analysis

After implementing the distributed training script, you should:

- run a short manual sweep over a wide range of `local_batch_size` values
- save rank traces and metric summaries for each run
- treat global `images/s` as the primary systems metric for choosing the batch size
- use the traces to explain why `images/s` improved or degraded across runs
- choose the batch size that maximizes `images/s`

**Notes:**

- In real SimCLR, very large global batches often matter for representation quality. This project intentionally evaluates distributed-systems behavior, not embedding quality.
- This capstone uses a **local loss gradient approximation**, not exact global gradients.
- Even with that simplification, the contrastive loss still breaks the common intuition that large batch sizes are a free lunch:
  - The odd ranks do not just run stage 1; they also materialize the gathered embedding set needed for the local loss.
  - The `all_gather` in `stage1_group` grows communication with global batch size.
  - The similarity and softmax compute in the loss calculation also grows with global batch size.
  - Because of that extra loss-side and communication-side work, the odd ranks are expected to be somewhat heavier on compute.

Take this into account in your analysis and:

- inspect the odd-rank traces specifically for `gather_embeddings` and `loss_calculation`, since that is where the contrastive-loss overhead appears
- compare the run length of `stage0_forward + stage0_backward` against `stage1_forward + loss_calculation`
- estimate communication percentage and waiting percentage
- show when a larger batch stops improving `images/s`
- explain whether a larger batch improved the local-work-to-synchronization ratio or merely increased waiting, communication percentage, or memory pressure

---

## Deliverables

1. A GitHub repo containing:

- all relevant Python scripts
- a short `README.md`
- optional stretch-goal controller script if implemented

The README should contain:

- exact setup commands
- exact data-preparation commands
- exact baseline run command
- exact profiler run command
- a short explanation of the loss calculation
- a short explanation of the loss gradient approximation
- a short explanation of the shard split and communication-group structure
- a short explanation of the bottleneck categories
- optional controller run command if implemented

2. Output artifacts, for example:

- `run_config.json`
- `metrics.csv`
- profiler trace JSON files, one per rank
- manual batch-size sweep summary table
- a short diagnosis summary
- optional controller decision log

3. A **Short video (5-10 min)** showing:

- **Code walkthrough**, specifically:
  - the stage-0 / stage-1 split in code
  - the boundary activation handoff and returned boundary gradient path
  - the communication groups used for point-to-point and collective operations
  - the loss calculation
  - the loss gradient approximation
- **Live demo execution** of a baseline run and one follow-up run
- **Output artifacts** analysis and discussion

---

## Required demo pattern

The live demo must include all of the following:

1. Fresh distributed training run

- launch a baseline profiled run with `torchrun`
- show that the training job runs correctly on the ImageNet-like synthetic workload

2. Trace walkthrough

- open at least one profiler trace
- identify the local compute, stage transfer, synchronization, `gather_embeddings`, `loss_calculation`, and any waiting-heavy evidence

3. Diagnosis

- explain the observed bottleneck using the Unit 2 / Unit 3 vocabulary

4. Manual tuning decision

- show the next batch size chosen to improve `images/s` from the trace and metric evidence

5. Follow-up run

- rerun with the updated configuration
- compare the old and new traces

6. Result explanation

- explain whether the run became better balanced
- support the explanation with trace and metric evidence

---

## Stretch goals (optional)

### Stretch A - async overlap

Extend the mainline runtime so the even ranks overlap forward-side and backward-side work instead of waiting synchronously for odd ranks to finish their share of the workload.

- after warmup, the even rank may alternate between forward-side progress and backward-side progress: hand off activations for step `t` to the stage-1 rank, then begin the backward pass for step `t-1` as soon as its boundary gradient returns and immediately afterwards begin the forward pass of step `t+1`
- keep at most two in-flight steps so the overlap is visible without turning the project into a full pipeline scheduler
- keep a small per-step state table so the even rank can save stage-0 boundary activations and any needed metadata until the matching gradient returns
- use explicit step ids in the handoff protocol so the returned boundary gradient for step `t` is matched with the saved forward state for step `t`

### Stretch B - load-balancing controller

Write an optional module that wraps the mainline training script and:

- runs the distributed training job with a chosen batch size and shard boundary layer
- analyzes the exported traces and summary metrics automatically
- compares the run length of the stage-0 compute region on even ranks against the stage-1-plus-loss region on odd ranks
- estimates what fraction of each step is spent in activation transfer, `gather_embeddings`, other communication, and waiting
- optimizes primarily for `images/s`
- uses communication-heavy percentage and stage imbalance as secondary heuristics when choosing a new batch size and/or shard boundary
- prefers shard boundaries that leave the odd ranks somewhat lighter, because the odd ranks also own the contrastive-loss path
- reruns the job and repeats the analysis
