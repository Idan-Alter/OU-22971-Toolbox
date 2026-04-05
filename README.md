# 22971 - Toolbox for Machine Learning and Big Data

This repository contains course materials for **Course 22971** at the **Open University of Israel**.

## Course goals

The course is designed to give students hands-on control of modern technical tools for common ML engineering tasks:

- experiment tracking and hyperparameter optimization
- model deployment
- distributed data processing
- distributed training on CPU and GPU clusters

The emphasis is practical engineering ability, with supporting theoretical understanding through system architecture concepts.

## Course content (parts)

### Part 1: [MLOps with MLflow](MLOps/README.md)

- model registry management
- experiment tracking (parameters, metrics, artifacts)
- hyperparameter optimization with Optuna integration
- deployment and monitoring

### Part 2: [Distributed computing with Ray](Ray/README.md)

- async execution, scalability, load management, fault tolerance
- remote functions and distributed state
- Ray architecture (scheduler, distributed object store, failure handling)
- efficient compute patterns (including MapReduce) and anti-patterns
- distributed model training
- Ray Data for parallel and sharded data processing

### Part 3: [Distributed deep learning with PyTorch Distributed](Distributed_DL/README.md)

- dev container setup for torchrun
- collective communication: broadcast, reduce, gather, scatter
- parallel training challenges: compute, memory, communication
- performance analysis with PyTorch Profiler and Perfetto or Chrome trace viewing
- distributed GPU training on managed cloud images
- five parallelism dimensions: data, tensor, pipeline, context, expert
- capstone project: SimCLR-like distributed training with manual ResNet sharding, per-rank traces, and manual scaling analysis

## Extra resources

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/)
- [Distributed Systems lecture series](https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [The Missing Semester](https://missing.csail.mit.edu/)

## Maintenance

Repo-wide maintenance scripts and helper runners live under [repo_maintenance/README.md](repo_maintenance/README.md).
