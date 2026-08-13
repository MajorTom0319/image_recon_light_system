"""Renderer-independent light types used by the hybrid bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class MeshAreaLight:
    """An ILE lamp or window represented by an emissive triangle mesh."""

    id: int
    light_type: str
    center: np.ndarray
    rgb: np.ndarray
    geometry_path: Path
    visible: bool
    geometry_scale: float = 1.0
    confidence: float = 1.0
    mask_path: Path | None = None
    window_lobes: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.center = np.asarray(self.center, dtype=np.float32).reshape(3)
        self.rgb = np.asarray(self.rgb, dtype=np.float32).reshape(3)
        self.geometry_path = Path(self.geometry_path)
        self.geometry_scale = float(self.geometry_scale)
        self.confidence = float(self.confidence)
        if self.mask_path is not None:
            self.mask_path = Path(self.mask_path)
        if self.window_lobes is not None:
            self.window_lobes = np.asarray(self.window_lobes, dtype=np.float32)

        if self.light_type not in {
            "visible_lamp",
            "invisible_lamp",
            "visible_window",
            "invisible_window",
        }:
            raise ValueError(f"Unsupported light type: {self.light_type}")
        if not np.isfinite(self.center).all():
            raise ValueError(f"Light {self.name} has a non-finite center")
        if not np.isfinite(self.rgb).all() or np.any(self.rgb < 0):
            raise ValueError(f"Light {self.name} has invalid RGB radiance")
        if not np.isfinite(self.geometry_scale) or self.geometry_scale <= 0:
            raise ValueError("geometry_scale must be finite and positive")
        if not self.geometry_path.is_file():
            raise FileNotFoundError(self.geometry_path)
        if self.is_window:
            if self.window_lobes is None or self.window_lobes.shape != (3, 7):
                raise ValueError(
                    f"Window {self.name} needs sun/sky/ground lobes with shape (3, 7)"
                )
            if not np.isfinite(self.window_lobes).all():
                raise ValueError(f"Window {self.name} has non-finite SG parameters")
            if np.any(self.window_lobes[:, :3] < 0):
                raise ValueError(f"Window {self.name} has negative SG radiance")
            direction_norms = np.linalg.norm(self.window_lobes[:, 3:6], axis=1)
            if np.any(direction_norms <= 1e-8):
                raise ValueError(f"Window {self.name} has a zero SG direction")
            self.window_lobes[:, 3:6] /= direction_norms[:, None]
            if np.any(self.window_lobes[:, 6] < 0):
                raise ValueError(f"Window {self.name} has negative SG concentration")
        elif self.window_lobes is not None:
            raise ValueError(f"Lamp {self.name} cannot contain window SG parameters")

    @property
    def name(self) -> str:
        return f"{self.light_type}_{self.id}"

    @property
    def scaled_center(self) -> np.ndarray:
        return self.center * self.geometry_scale

    @property
    def is_window(self) -> bool:
        return self.light_type.endswith("_window")


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
