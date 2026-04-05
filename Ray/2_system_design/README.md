# Ray Unit 2 - Distributed Systems Design Through Classical Examples

This unit uses familiar workloads to teach distributed systems design.

The goal is to look at classical problems and ask the systems questions behind them:

- How should we split work across a cluster?
- Where should coordination live: the driver, tasks, or actors?
- What changes when work can fail and retry?

We use two case studies:

- [MapReduce word count](2_0_map_reduce/0_map_reduce.ipynb): the classical fan-out / shuffle / reduce pattern.
- [Distributed hyperparameter optimization](2_1_distributed_HPO/0_distributed_hpo.ipynb): a nested, stateful workload with retries and coordination.

The throughline is system design. Each example is a vehicle for reasoning about decomposition, communication, failure handling, and control flow on a cluster.
