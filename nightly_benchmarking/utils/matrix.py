import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nightly_benchmarking.utils.datasets import DatasetResolver


@dataclass
class MatrixEntry:
    name: str
    script: str | None = None
    args: str | None = None
    script_base_dir: str | None = None
    timeout_s: int | None = None
    ray: dict[str, Any] = field(default_factory=dict)  # supports only single node: num_cpus,num_gpus,object_store_gb
    # If set, overrides the session-level delete_scratch setting for this entry
    delete_scratch: bool | None = None

    def get_command_to_run(self) -> str:
        if self.script:
            script_path = Path(self.script_base_dir or "nightly_benchmarking/scripts") / self.script
            cmd = f"python {script_path} {self.args or ''} --benchmark-results-path" + " {session}/benchmark_results"
        else:
            msg = f"Entry {self.name} must specify either cmd or script"
            raise ValueError(msg)

        return cmd

    @staticmethod
    def substitute_datasets_in_cmd(cmd: str, resolver: DatasetResolver) -> str:
        pattern = re.compile(r"\{dataset:([^,}]+),([^}]+)\}")

        def _replace(match: re.Match[str]) -> str:
            dataset_name = match.group(1).strip()
            dataset_format = match.group(2).strip()
            return resolver.resolve(dataset_name, dataset_format)

        return pattern.sub(_replace, cmd)

    @staticmethod
    def substitute_template_placeholders(cmd: str, entry_dir: str) -> str:
        """Substitute template placeholders in command.

        Supports {session}/path patterns where anything after {session}/ becomes
        a path within the entry directory.

        Examples:
        - {session}/results.json -> /path/to/session/entry/results.json
        - {session}/tempdir/output -> /path/to/session/entry/tempdir/output
        - {session}/logs -> /path/to/session/entry/logs
        """
        entry_path = Path(entry_dir)

        # Handle {session}/path patterns

        session_pattern = re.compile(r"\{session\}/([^}\s]+)")

        def replace_session_path(match: re.Match[str]) -> str:
            subpath = match.group(1)
            return str(entry_path / subpath)

        return session_pattern.sub(replace_session_path, cmd)


@dataclass
class MatrixConfig:
    results_dir: str
    entries: list[MatrixEntry]
    default_timeout_s: int = 7200
    mlflow: dict[str, Any] = field(default_factory=dict)
    wandb: dict[str, Any] = field(default_factory=dict)
    slack: dict[str, Any] = field(default_factory=dict)
    # Whether to delete the entry's scratch directory after completion by default
    delete_scratch: bool = True

    @classmethod
    def load_yaml(cls, path: str) -> "MatrixConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        if "results_dir" not in data:
            msg = "results_dir is required"
            raise ValueError(msg)

        if "entries" not in data:
            msg = "entries is required"
            raise ValueError(msg)

        # Resolve environment variables in the loaded data
        data = _resolve_env_vars(data)
        # Resolve entries
        entries = [MatrixEntry(**e) for e in data["entries"]]

        # assert none of the entries have the same name
        if len(entries) != len({entry.name for entry in entries}):
            msg = "Entries must have unique names"
            raise ValueError(msg)

        return MatrixConfig(
            results_dir=data["results_dir"],
            entries=entries,
            default_timeout_s=data.get("default_timeout_s", 7200),
            mlflow=data.get("mlflow", {}),
            wandb=data.get("wandb", {}),
            slack=data.get("slack", {}),
            delete_scratch=data.get("delete_scratch", True),
        )


def _resolve_env_vars(data: dict | list | str) -> dict | list | str:
    """Recursively resolve environment variables in YAML data.

    Supports ${VAR_NAME} syntax. If the environment variable is not found,
    the original string is left unchanged.
    """
    import os
    import re

    if isinstance(data, dict):
        return {key: _resolve_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Pattern to match ${VAR_NAME}
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replace_env_var(match: re.Match[str]) -> str:
            env_var_name = match.group(1)
            env_value = os.getenv(env_var_name)
            if env_value is not None:
                return env_value
            else:
                msg = f"Environment variable {env_var_name} not found in the environment"
                raise ValueError(msg)

        return pattern.sub(replace_env_var, data)
    else:
        return data
