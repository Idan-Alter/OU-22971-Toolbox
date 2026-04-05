# Ray

This folder contains Part 2 of Course 22971: a hands-on Ray sequence covering core execution primitives, local and Docker-backed clusters, system-design patterns, Ray Data, and a final capstone project.

## Start here

- Unit 0: [Core Primitives](0_core_primitives)
- Unit 1: [Docker Cluster Setup](1_cluster_setup/0_docker_cluster_setup.md)
- Unit 2: [Distributed Systems Design Through Classical Examples](2_system_design/README.md)
- Unit 3: [Sharded Data](3_ray_data/ray_data.ipynb)
- Unit 4: [Capstone Project Design Doc](4_ray_capstone_project/design_doc.md)

## Setup

This part keeps its own Conda spec in [environment.yml](environment.yml).
Create it:

```powershell
conda env create -f environment.yml
```

Most local notebooks and `ray` CLI commands assume the `22971-ray` environment:

```powershell
conda activate 22971-ray
```
