import mlflow
import mlflow.sklearn
import pickle

class MLFlowTracker:
    def __init__(self, experiment_name="DefaultExperiment"):
        """
        Initialize the MLFlowTracker.
        
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name=None):
        """
        Start a new MLflow run.
        
        """
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        """
        Log model parameters.
        
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        """
        Log model metrics.
       
        """
        mlflow.log_metrics(metrics)

    def log_model(self, model, model_name):
        """
        Log a trained model in MLflow.
        
        """
        mlflow.sklearn.log_model(model, model_name)

    def log_artifact(self, file_path, artifact_path=None):
        """
        Log additional artifacts (like vectorizers, preprocessed data, etc.).
        
        """
        mlflow.log_artifact(file_path, artifact_path)

    def end_run(self):
        """
        End the current MLflow run.
        """
        mlflow.end_run()

    def save_model_version(self, model_uri, model_name):
        """
        Register a model version in MLflow Model Registry.
        
        """
        client = mlflow.tracking.MlflowClient()
        mlflow.register_model(model_uri, model_name)

    def transition_model(self, model_name, version, stage):
        """
        Transition model to different stages (e.g., Staging, Production).
        
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
