# Distributed DL Unit 4 - DDP on a Cloud GPU Instance

**So far:**
1. We've been using low-level primitives (e.g. `all_reduce`).
1. We ran our scripts on CPU-only machines.

**In this unit:**
1. We'll use higher-level abstractions from `torch.nn.parallel` that implement (and hide) the actual collective communication patterns. Specifically:
   - `DistributedDataParallel`
   - `DistributedSampler`
2. Adapt the script to run on GPUs.
3. Run and profile the script on:
   - free Colab single-GPU instances
   - paid multi-GPU cloud instances (instructor demo)
4. Analyze GPU-enabled run traces.

---

## Setup

**Free path:**
1. Upload `colab_launcher.ipynb` to Colab.
2. Start a GPU runtime and make sure `torch.cuda.is_available()` returns `True`.
3. Run the notebook.
4. Download and analyze the generated traces per the following instructions.

**Note:** in `4_ddp_on_cloud_gpus/colab_output` you will find the artifacts generated in such a run. You can skip running it yourself and go straight to the analysis.

**Paid path (optional):**
1. Choose a cloud provider.
2. Start a 2-GPU runtime with PyTorch (any CUDA-compatible GPU will do).
3. Connect with SSH or JupyterLab (usually available).
4. Run these checks in a terminal:
   ```bash
   nvidia-smi
   ```

   ```bash
   python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count()); print('distributed_available', torch.distributed.is_available())"
   ```
   What to look for:
   - CUDA is available
   - `gpu_count` is at least `2`
   - `torchvision` imports cleanly
5. Download the script to the `cwd`:
   ```bash
   curl -L https://raw.githubusercontent.com/Idan-Alter/OU-22971-Toolbox/main/Distributed_DL/4_ddp_on_cloud_gpus/profile_ddp_gpu.py -o profile_ddp_gpu.py
   ```
6. Run the commands that follow in the cloud runtime.
   
**Note:** if your repo copy includes example files under `4_ddp_on_cloud_gpus/runpod_output`, you can skip generating a multi-GPU run yourself and go straight to the analysis.

---

## Optional local CPU validation

The script in this unit also accepts a `--cpu` flag. That switches the backend to `gloo` and runs the same high-level DDP flow on CPU.

This is useful for smoke testing scripts locally before spinning up costly cloud instances.

---

## What changes from unit 3

The training-step story is still the same:

- get a sharded batch
- run forward
- compute loss
- run backward
- synchronize gradients
- apply the optimizer step

What changes is **where** the compute happens and **how** PyTorch handles the synchronization:

- Previously we called `all_reduce` on gradients manually.
- Now, `DistributedDataParallel` handles gradient synchronization for us during backward.
- `DistributedSampler` replaces our earlier toy sampler which scatters disjoint indices to all ranks.
- If a GPU is available, the NCCL distributed communication backend is used and local computation happens on the accelerator.

So Unit 4 is not a new training-step story.
It is the same sharded batch -> forward -> loss -> backward -> sync -> step flow from Units 2 and 3, now expressed through higher-level PyTorch APIs on faster hardware.

In the profiler trace that means:

- DDP gradient sync shows up during `backward()`, because DDP launches bucketed `all_reduce` work from autograd hooks.
- `DistributedSampler` keeps each rank on a different shard of a dataset.

---

## Analyzing a GPU training run trace

1. Run 
   ```bash
   torchrun --standalone --nproc_per_node=1 profile_ddp_gpu.py --trace-name gpu_initial --num-workers 0
   ``` 
   And open the trace at [https://ui.perfetto.dev/](https://ui.perfetto.dev/).

   Alternatively, use `4_ddp_on_cloud_gpus/colab_output/colab_gpu_initial_rank0.json`.

2. Open the main Python thread; this is the CPU trace.
3. The main GPU trace is usually named "Stream `N`", (`N=7` in the colab trace).
4. Observe:
   - At the bottom of some CPU call stacks there are `cudaLaunchKernel` invocations; these are the handoffs from the CPU to the GPU.
      - Click such a slice and see the execution of the relevant CUDA kernel.
   - GPU execution is async: the CPU schedules kernel execution in a burst and then waits.
   - The second `train_step` slice is significantly shorter than the first. This is a profiling artifact, not an actual speedup.
      - Look closely and see that the second step launches CUDA kernels on the GPU that finish execution outside the `train_step` slice.
   - The GPU stream is empty during `next_batch` slices: the GPU idles while the CPU loads the next batch.
   - This is an antipattern since GPU time is expensive hardware.
      - Easy fix: load the next batch in a separate CPU subprocess asynchronously.
   - Even though this is a single-process run, there is still slight overhead from using the PyTorch DDP abstraction, but that is not actual cross-rank communication.

## Optimizing GPU utilization: async dataloader

1. Run
   ```bash
   torchrun --standalone --nproc_per_node=1 profile_ddp_gpu.py --trace-name gpu_baseline --num-workers 1
   ```
      - `num-workers=1` is passed as an argument to the dataloader and makes it load data in a subprocess.

2. Open the trace at [https://ui.perfetto.dev/](https://ui.perfetto.dev/).

   Alternatively, use `4_ddp_on_cloud_gpus/colab_output/colab_gpu_baseline_rank0.json`.
3. Observe and compare to the previous trace:
   - `next_batch` is very short.
   - By the time the main thread executes it, the batch is already materialized in CPU RAM by the subprocess.
      - The main thread still copies the batch to the GPU RAM.
      - Important interpretation: this slice measures how long the main thread waits at `next_batch`, not the full CPU cost of producing the batch.
   - The GPU stream is full (~97% utilization).
   - This is about as good as it gets.

---

## Analyzing a distributed GPU training run trace

1. Run
   ```bash
   torchrun --standalone --nproc_per_node=2 profile_ddp_gpu.py --trace-name gpu_baseline --num-workers 1
   ```
   - `--nproc_per_node=2` tells torchrun to run a distributed job on two ranks.
   - `--num-workers 1` is the default value, so we don't need to send it explicitly.
2. Open the trace at [https://ui.perfetto.dev/](https://ui.perfetto.dev/).

   Alternatively, use the included A40 baseline artifacts:

   ```text
   4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_baseline_rank0.json
   4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_baseline_rank1.json
   4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_baseline_stdout.log
   ```

3. Read the baseline trace as a bottleneck story:
   - The run uses two NVIDIA A40 GPUs, `resnet18`, `batch_size=64`, and `num_workers=1`.
   - The log reports `188.4 images/s` globally.
   - Both ranks are mapped to separate GPUs, but they do not spend the profiled window in the same place.
   - In the actual A40 baseline traces, `next_batch` takes most of the CPU launch path: about `94%` of `train_step` on rank 0 and `77%` on rank 1.
   - The faster rank can reach NCCL while the slower rank is still preparing input, so some apparent communication time is really waiting for the peer rank.
   - The GPU streams have large empty spaces between compute bursts.

   In other words: DDP is working, but the two expensive GPUs are waiting for data. The first tuning knob is not NCCL, model architecture, or optimizer math. It is the input pipeline.

4. The direct next step is to increase `--num-workers`.

### `profile_ddp_gpu.py` highlights

1. Select one device per local rank and initialize distributed execution:
   ```python
   torch.cuda.set_device(local_rank)
   backend = "nccl"
   device = torch.device("cuda", local_rank)
   dist.init_process_group(backend=backend)
   ```
   In `--cpu` mode the script switches to `gloo` instead.

2. Build one logical dataset, then shard its indices across ranks:
   ```python
   sampler = DistributedSampler(
       dataset,
       num_replicas=world_size,
       rank=rank,
       shuffle=True,
       drop_last=True,
   )
   loader = DataLoader(..., sampler=sampler, num_workers=args.num_workers, ...)
   sampler.set_epoch(0)
   ```
   `DistributedSampler` decides which examples this rank sees, and `DataLoader` turns that shard into local batches. Setting `--num-workers 0` keeps that loading work on the training process; the default baseline uses `--num-workers 1`.

3. Wrap the model in DDP:
   ```python
   ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
   ```
   After that, gradient synchronization is triggered during `backward()` instead of by our manual `all_reduce` calls from unit 2.

4. Keep the same high-level trace structure from unit 3, including `next_batch`:
   ```python
   with record_function("next_batch"):
       images, targets = next(data_iter)
       images = images.to(device, non_blocking=device.type == "cuda")
       targets = targets.to(device, non_blocking=device.type == "cuda")
   ```
   In this unit, `next_batch` includes both DataLoader fetch and host-to-device transfer.

5. Profile only the last two steps:
   ```python
   warmup_steps = max(args.steps - 2, 0)
   profiled_steps = min(args.steps, 2)
   ```
   That keeps startup noise out of the exported trace.

6. Fence the profiled window when timing CUDA work:
   ```python
   torch.cuda.synchronize(device)
   profiled_start = time.perf_counter()
   ...
   torch.cuda.synchronize(device)
   profiled_seconds = time.perf_counter() - profiled_start
   ```
   The synchronizations are there to make the wall-clock throughput timing reflect actual work done since CUDA kernels execute async.

7. Export one trace per rank and gather one summary row per rank onto rank 0:
   ```python
   prof.export_chrome_trace(str(trace_path))
   gathered_rows = gather_summaries_on_rank_zero(...)
   ```
   Rank 0 then prints the compact summary and saves it to a `.log` file.

By default, the script runs a small baseline:

- `model=resnet18`
- `batch_size=64`
- `num_workers=1`
- `steps=5`
- `dataset_size=4096`, so the documented runs still fit comfortably within one pass of the sharded loader

## Tuning ladder: make the GPUs busier

The tuning ladder is deliberately mechanical:

1. Fix the most obvious idle time.
2. Re-open the trace.
3. Only then decide what the next bottleneck is.

The included A40 traces show that story clearly.

### 1. More DataLoader workers

Run:

```bash
torchrun --standalone --nproc_per_node=2 profile_ddp_gpu.py --trace-name gpu_workers4 --num-workers 4
```

Open:

```text
4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_workers4_rank0.json
4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_workers4_rank1.json
4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_workers4_stdout.log
```

Compared with the A40 baseline, only one thing changed:

- same `resnet18`
- same `batch_size=64`
- same two A40 GPUs
- `num_workers=1` -> `num_workers=4`

This is the expected first win:

- `next_batch` becomes much shorter.
- In the A40 traces, CPU `next_batch` drops from most of the `train_step` launch path to roughly `6%`.
- The GPU streams are much denser.
- Global throughput rises from `188.4 images/s` to `533.0 images/s`.

So the baseline diagnosis was right: the original 2-GPU run was dominated by dataloading.

At `num_workers=4`, the trace is already pretty good. The A40 workers4 traces show roughly `98%` GPU activity across the profiled CUDA span. There is still a small visible gap on the compute stream near the end of backward, but now it is a different kind of gap: look inside `backward` for `nccl:all_reduce`, and on the NCCL stream for `ncclDevKernel_AllReduce_Sum_f32_RING_LL(...)`.

That gap is the DDP communication tail, not another DataLoader stall. Earlier gradient buckets can overlap with remaining backward compute, but the last gradient bucket has no later backward computation to hide behind. Once the input pipeline is no longer starving the GPUs, that last NCCL all-reduce becomes visible.

### 2. More model compute

Run a heavier model after keeping the dataloader fix:

```bash
torchrun --standalone --nproc_per_node=2 profile_ddp_gpu.py --trace-name gpu_resnet50_batch128_workers4 --model resnet50 --batch-size 128 --num-workers 4
```

Open:

```text
4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_resnet50_batch128_workers4_rank0.json
4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_resnet50_batch128_workers4_rank1.json
4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_resnet50_batch128_workers4_stdout.log
```

This run changes the workload, so compare the trace shape more than the raw images/sec:

- `resnet50` does more computation per image than `resnet18`.
- Forward and backward regions become much larger.
- The NCCL all-reduce still happens, and `resnet50` also has more gradients to reduce, so model size alone is not a free communication win.
- The log reports `330.3 images/s`, lower than the previous `resnet18` run because each image now costs more compute.


### 3. Larger batch

```bash
torchrun --standalone --nproc_per_node=2 profile_ddp_gpu.py --trace-name gpu_resnet50_batch256_workers4 --model resnet50 --batch-size 256 --num-workers 4
```

Open:

```text
4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_resnet50_batch256_workers4_rank0.json
4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_resnet50_batch256_workers4_rank1.json
4_ddp_on_cloud_gpus/runpod_output/A40_X_2/gpu_resnet50_batch256_workers4_stdout.log
```

compared with the previous run:

- same two A40 GPUs
- same `resnet50`
- same `num_workers=4`
- `batch_size=128` -> `batch_size=256`

The bigger batch improves global throughput from `330.3 images/s` to `512.9 images/s`.

Why? A larger local batch gives each rank more compute per training step.


## Hardware comparison: A40 vs A100

Now compare hardware only with the same software configuration.

Assume the A40 costs 1$/h and the A100 costs  3$/h. The A100 should therefore be more than `3x` faster to win on throughput per dollar.

For the baseline run:

```text
A40_X_2/gpu_baseline_stdout.log:  resnet18, batch_size=64, num_workers=1, 188.4 images/s
A100_X_2/gpu_baseline_stdout.log: resnet18, batch_size=64, num_workers=1, 490.1 images/s
```

The A100 is `2.6x` faster, but it costs `3x` as much. That is faster wall-clock time, but worse throughput per dollar.

For the tuned worker run:

```text
A40_X_2/gpu_workers4_stdout.log:  resnet18, batch_size=64, num_workers=4, 533.0 images/s
A100_X_2/gpu_workers4_stdout.log: resnet18, batch_size=64, num_workers=4, 899.6 images/s
```

The A100 is only `1.7x` faster here. At `3x` the cost, the A40 is the better value for this workload.

So the answer is: the A100 is absolutely faster, but for these apples-to-apples traces it is not enough faster to justify a `3x` price if the goal is images/sec per dollar. 

---

## Exercise

1. Push the tuning ladder until the hardware pushes back.
   Increase batch size until you hit out-of-memory on the GPU.
   Analyze the trace of the last successful configuration:
   - What got denser?
   - What idle gaps shrank?
   - Did throughput improve?
   - Did synchronization become a smaller fraction of the full step?
