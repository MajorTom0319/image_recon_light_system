"""Renderer-independent light types used by the hybrid bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class MeshAreaLight:
    """An ILE lamp represented by an emissive triangle mesh."""

    id: int
    light_type: str
    center: np.ndarray
    rgb: np.ndarray
    geometry_path: Path
    visible: bool
    geometry_scale: float = 1.0
    confidence: float = 1.0
    mask_path: Path | None = None

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.center = np.asarray(self.center, dtype=np.float32).reshape(3)
        self.rgb = np.asarray(self.rgb, dtype=np.float32).reshape(3)
        self.geometry_path = Path(self.geometry_path)
        self.geometry_scale = float(self.geometry_scale)
        self.confidence = float(self.confidence)
        if self.mask_path is not None:
            self.mask_path = Path(self.mask_path)

        if self.light_type not in {"visible_lamp", "invisible_lamp"}:
            raise ValueError(f"Unsupported light type: {self.light_type}")
        if not np.isfinite(self.center).all():
            raise ValueError(f"Light {self.name} has a non-finite center")
        if not np.isfinite(self.rgb).all() or np.any(self.rgb < 0):
            raise ValueError(f"Light {self.name} has invalid RGB radiance")
        if not np.isfinite(self.geometry_scale) or self.geometry_scale <= 0:
            raise ValueError("geometry_scale must be finite and positive")
        if not self.geometry_path.is_file():
            raise FileNotFoundError(self.geometry_path)

    @property
    def name(self) -> str:
        return f"{self.light_type}_{self.id}"

    @property
    def scaled_center(self) -> np.ndarray:
        return self.center * self.geometry_scale


@dataclass(slots=True)
class ILELightSet:
    """Validated lamp proposals loaded from one ILE JSON export."""

    source_path: Path
    lights: list[MeshAreaLight]
    image_path: Path | None = None
    depth_path: Path | None = None
    schema_version: int | str = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_path = Path(self.source_path)
        if self.image_path is not None:
            self.image_path = Path(self.image_path)
        if self.depth_path is not None:
            self.depth_path = Path(self.depth_path)
