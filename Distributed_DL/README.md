# Distributed Deep Learning with PyTorch Distributed

This folder contains Part 3 of Course 22971: a sequence on how distributed training moves tensors, where it gets expensive, and how to reason about scaling before adding GPU and model-parallel complexity.

## Start here

- Unit 0: [Dev Container Setup](0_devcontainer_setup/0_devcontainer_setup.md)
- Unit 1: [Collective Communication in PyTorch](1_collective_communication/0_collective_communication.md)
- Unit 2: [Parallel Training Challenges](2_training_challenges/0_training_challenges.md)
- Unit 3: [PyTorch Profiler: Probe the Toy Script with Traces](3_profiler_cpu_traces/0_profiler_cpu_traces.md)
- Unit 4: [DDP on a Cloud GPU Instance](4_ddp_on_cloud_gpus/0_ddp_on_cloud_gpus.md)
- Unit 5: [Parallelism Strategies Beyond DDP](5_parallelism_strategies/README.md)
- Unit 6: [Capstone Project Design Doc](6_torch_dist_capstone_project/design_doc.md)

## Setup

In this part, we run everything inside the dev container or on cloud instances.

The official Part 3 environment is `22971-td`, defined in [environment.yml](environment.yml).

The dev container uses CPU-only PyTorch so the local path stays lightweight and works on laptops.

For the later GPU sections we will use a managed cloud instance with GPU-enabled PyTorch already installed.
