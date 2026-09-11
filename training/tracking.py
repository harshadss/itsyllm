"""Optional experiment tracking kept outside individual trainer loops."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Protocol


class Tracker(Protocol):
    def log(self, metrics: dict[str, int | float]) -> None: ...

    def finish(self) -> None: ...


class NullTracker:
    """Default tracker that makes experiment tracking entirely optional."""

    def log(self, metrics: dict[str, int | float]) -> None:
        del metrics

    def finish(self) -> None:
        pass


class WandbTracker:
    def __init__(
        self,
        *,
        project: str,
        entity: str | None,
        run_name: str | None,
        mode: str,
        output_dir: Path,
        stage: str,
        run_config: dict[str, Any],
    ) -> None:
        try:
            import wandb
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "W&B tracking requires the optional dependency. Run: uv sync --extra wandb"
            ) from error

        output_dir.mkdir(parents=True, exist_ok=True)
        run_id_path = output_dir / ".wandb-run-id"
        run_id = run_id_path.read_text(encoding="utf-8").strip() if run_id_path.is_file() else None
        init_kwargs: dict[str, Any] = {
            "project": project,
            "entity": entity,
            "name": run_name,
            "mode": mode,
            "dir": str(output_dir),
            "config": run_config,
            "tags": [f"stage:{stage}"],
        }
        if run_id:
            init_kwargs.update(id=run_id, resume="allow")
        self._run = wandb.init(**init_kwargs)
        if not run_id:
            run_id_path.write_text(self._run.id, encoding="utf-8")
        self._run.define_metric("train/*", step_metric="train/tokens")
        self._run.define_metric("system/*", step_metric="train/tokens")
        self._logging_failed = False

    def log(self, metrics: dict[str, int | float]) -> None:
        if self._logging_failed:
            return
        try:
            self._run.log(metrics)
        except Exception as error:
            self._logging_failed = True
            print(f"W&B logging disabled after an error: {error}", file=sys.stderr)

    def finish(self) -> None:
        self._run.finish()


def create_tracker(
    *,
    mode: str,
    project: str | None,
    entity: str | None,
    run_name: str | None,
    output_dir: Path,
    stage: str,
    run_config: dict[str, Any],
) -> Tracker:
    if mode == "disabled":
        return NullTracker()
    if not project:
        raise ValueError("--wandb-project is required unless --wandb-mode disabled")
    return WandbTracker(
        project=project,
        entity=entity,
        run_name=run_name,
        mode=mode,
        output_dir=output_dir,
        stage=stage,
        run_config=run_config,
    )
