"""Dashboard paths and lightweight runtime configuration."""
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DashboardPaths:
    root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts" / "models"

    @property
    def metrics(self) -> Path:
        return self.root / "reports" / "metrics"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

