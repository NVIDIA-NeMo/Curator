from __future__ import annotations

import os

from loguru import logger


class MlflowSink:
    def __init__(self, config: dict | None = None) -> None:
        import mlflow

        self.mlflow = mlflow

        # Set tracking URI from config or env var
        if config and "tracking_uri" in config:
            tracking_uri = config["tracking_uri"]
            self.mlflow.set_tracking_uri(tracking_uri)
        else:
            msg = "tracking_uri is not defined in the config (for mlflow sink)"
            raise ValueError(msg)

        # Set experiment from config, env var, or default
        if config and "experiment" in config:
            experiment = config["experiment"]
        else:
            msg = "experiment is not defined in the config (for mlflow sink)"
            raise ValueError(msg)

        self.mlflow.set_experiment(experiment)

        self._parent_run_id = None
        self._child_run_id = None
        self._parent_run = None

    def start_session(self, session_name: str) -> None:
        """Start a parent run for the session."""
        try:
            self._parent_run = self.mlflow.start_run(run_name=session_name)
            self._parent_run_id = self._parent_run.info.run_id
            logger.info(f"🧪 MLflow parent run started: {self._parent_run_id}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to start MLflow parent run: {e}")

    def end_session(self, success: bool = True) -> None:
        """End the parent run with specified success status."""
        try:
            if self._parent_run_id is not None:
                # First end any active child run
                if self._child_run_id is not None:
                    self.mlflow.end_run()
                    self._child_run_id = None
                # Parent run should already be active after child ends, just end it
                status = "FINISHED" if success else "FAILED"
                self.mlflow.end_run(status=status)
                self._parent_run_id = None
                self._parent_run = None
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to end MLflow parent run: {e}")

    def start_run(self, run_name: str) -> None:
        """Start a child run for the entry."""
        try:
            # Create nested run under the parent
            if self._parent_run_id is not None:
                child_run = self.mlflow.start_run(
                    run_name=run_name, nested=True, tags={"mlflow.parentRunId": self._parent_run_id}
                )
            else:
                child_run = self.mlflow.start_run(run_name=run_name)
            self._child_run_id = child_run.info.run_id
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to start MLflow child run: {e}")

    def log_params(self, params: dict) -> None:
        """Log parameters to the child run."""
        try:
            if self._child_run_id is not None:
                # Child run is already active, just log directly
                self.mlflow.log_params(params)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to log params to MLflow child run: {e}")

    def log_metrics(self, metrics: dict) -> None:
        """Log metrics to the child run."""
        try:
            if self._child_run_id is not None:
                # Child run is already active, just log directly
                self.mlflow.log_metrics(metrics)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to log metrics to MLflow child run: {e}")

    def log_artifact(self, path: str) -> None:
        """Log artifact to the child run."""
        try:
            if self._child_run_id is not None and os.path.exists(path):
                # Child run is already active, just log directly
                # self.mlflow.log_artifact(path)  # noqa: ERA001
                logger.warning(f"Skipping artifact logging to MLflow: {path}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to log artifact to MLflow child run: {e}")

    def end_run(self, success: bool = True) -> None:
        """End the child run with specified success status."""
        try:
            if self._child_run_id is not None:
                # Child run should already be active, just end it
                status = "FINISHED" if success else "FAILED"
                self.mlflow.end_run(status=status)
                self._child_run_id = None
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to end MLflow child run: {e}")
