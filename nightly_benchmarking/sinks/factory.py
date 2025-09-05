from typing import Protocol

from loguru import logger


class Sink(Protocol):
    def start_session(self, session_name: str) -> None: ...
    def end_session(self, success: bool = True) -> None: ...
    def start_run(self, run_name: str) -> None: ...
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def end_run(self, success: bool = True) -> None: ...


class NoopSink:
    def start_session(self, session_name: str) -> None:
        pass

    def end_session(self, success: bool = True) -> None:
        pass

    def start_run(self, run_name: str) -> None:
        pass

    def log_params(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def end_run(self, success: bool = True) -> None:
        pass


def build_sink(kind: str, config: dict | None = None) -> Sink:
    if kind == "mlflow":
        try:
            from .mlflow_sink import MlflowSink

            return MlflowSink(config=config.get("mlflow", {}) if config else {})
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to build MLflow sink: {e}")
            return NoopSink()
    if kind == "wandb":
        try:
            from .wandb_sink import WandbSink

            return WandbSink(config=config.get("wandb", {}) if config else {})
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to build WandB sink: {e}")
            return NoopSink()
    return NoopSink()
