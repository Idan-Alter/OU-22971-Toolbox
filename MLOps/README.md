# MLOps

This folder contains Part 1 of Course 22971: a hands-on MLOps sequence built around reproducibility, MLflow, Optuna, monitoring, registry-based promotion, and a final capstone workflow.

## Start here

- Unit 0: [Colab Reproducibility Anti-Pattern Demo](0_colab_antipatterns/colab_antipatterns.ipynb)
- Unit 1: [Reproducible ML Pipeline Setup (Conda)](1_conda_environments/0_conda_environments.md)
- Unit 2: [Experiment Logging and Artifact Persistence](2_logging_persistence/0_logging_persistence.md)
- Unit 3: [MLflow Setup](3_mlflow_setup/0_mlflow_setup.md)
- Unit 4: [Logging](4_mlflow_logging/0_mlflow_logging.md)
- Unit 5: [Hyperparameter Tuning with Optuna](5_xgboost_tuning/2_optuna_hyperparameter_tuning.md)
- Unit 6: [Model Evaluation, Monitoring, and Temporal Data Drift (Case Study)](6_monitoring_data_drift/1_monitoring_data_drift.md)
- Unit 7: [Model Registry, Promotion, and Deployment](7_model_registry_deployment/0_model_registry_deployment.md)
- Unit 8: [Capstone Project Design Doc](8_mlops_capstone_project/design_doc.md)

## Taxi case study

Units 6-8 build a single storyline around NYC Green Taxi data: monitor a baseline model, move models through the registry, and then design a fuller monitoring and retraining loop in the capstone project.

## Setup

If you are starting fresh, begin with Unit 1 to create and validate the `22971-mlflow` environment. Later units assume that environment already exists.
