import os

from loguru import logger


class WandbSink:
    def __init__(self) -> None:
        import wandb  # pyright: ignore[reportMissingImports]

        self.wandb = wandb
        self._project = os.getenv("WANDB_PROJECT", "ray-curator")
        self._session_name = None
        self._run = None

    def start_session(self, session_name: str) -> None:
        """Set the session name for grouping runs."""
        self._session_name = session_name

    def end_session(self, success: bool = True) -> None:
        """End session - WandB groups don't need explicit cleanup."""

    def start_run(self, run_name: str) -> None:
        """Start a new run in the session group."""
        try:
            if self._run:
                self._run.finish()

            self._run = self.wandb.init(
                project=self._project,
                name=run_name,
                group=self._session_name,  # Group runs by session name
                tags=[f"session:{self._session_name}"] if self._session_name else [],
                reinit=True,  # Allow multiple runs in same process
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to start WandB run: {e}")

    def log_params(self, params: dict) -> None:
        """Log parameters to the current run."""
        try:
            if self._run:
                self.wandb.config.update(params, allow_val_change=True)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to log params to WandB run: {e}")

    def log_metrics(self, metrics: dict) -> None:
        """Log metrics to the current run."""
        try:
            if self._run:
                self.wandb.log(metrics)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to log metrics to WandB run: {e}")

    def log_artifact(self, path: str) -> None:
        """Log artifact to the current run."""
        try:
            if self._run and os.path.exists(path):
                art = self.wandb.Artifact("benchmark-artifacts", type="benchmark")
                art.add_file(path)
                self._run.log_artifact(art)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to log artifact to WandB run: {e}")

    def end_run(self, success: bool = True) -> None:
        """End the current run."""
        try:
            if self._run:
                # WandB doesn't have explicit status like MLflow, but we can log it as metadata
                self.wandb.log({"run_success": success, "run_status": "FINISHED" if success else "FAILED"})
                self._run.finish()
                self._run = None
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to end WandB run: {e}")
