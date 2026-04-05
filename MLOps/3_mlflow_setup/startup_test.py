import json
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # Which server to use for logging?
mlflow.set_experiment("3_mlflow_setup")           # Which experiment to log into?

with mlflow.start_run():
    mlflow.log_param("smoke", "test")
    mlflow.log_metric("ok", 1.0)

    # create a tiny artifact
    artifact = {
        "status": "ok",
        "note": "smoke test run"
    }

    with open("artifact.json", "w") as f:
        json.dump(artifact, f, indent=2)

    mlflow.log_artifact("artifact.json")

    print("logged")
