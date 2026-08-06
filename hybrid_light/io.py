"""Read IndoorLightEditing light exports without importing either model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .light_types import ILELightSet, MeshAreaLight


def _vector(value: Any, size: int, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size < size:
        raise ValueError(f"{field} needs at least {size} values, got {result.size}")
    result = result[:size]
    if not np.isfinite(result).all():
        raise ValueError(f"{field} contains non-finite values")
    return result


def _optional_path(value: Any, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _resolve_geometry(
    filename: str,
    *,
    json_dir: Path,
    prediction_dir: Path | None,
) -> Path:
    path = Path(filename).expanduser()
    candidates = [path] if path.is_absolute() else [json_dir / path]
    if prediction_dir is not None and not path.is_absolute():
        candidates.append(prediction_dir / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve light geometry {filename!r}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _document_geometry_scale(document: dict[str, Any]) -> float:
    camera = document.get("camera") or {}
    if camera.get("depth_normalized", False):
        scale = float(camera.get("depth_scale", 0.0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Normalized ILE depth requires a positive camera.depth_scale")
        return scale
    return 1.0


def load_ile_lights(
    json_path: str | Path,
    *,
    include_visible: bool = True,
    include_invisible: bool = True,
    geometry_scale: float | None = None,
) -> ILELightSet:
    """Load lamp meshes and radiance from an ILE ``light_predictions.json``.

    Windows are intentionally excluded from this first prototype because ILE's
    directional sun/sky/ground window representation is not a uniform emitter.
    """
    json_path = Path(json_path).expanduser().resolve()
    with json_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)

    if not isinstance(document.get("lights"), dict):
        raise ValueError("ILE JSON must contain a 'lights' object")

    json_dir = json_path.parent
    prediction_dir = _optional_path(document.get("prediction_dir"), json_dir)
    scale = _document_geometry_scale(document) if geometry_scale is None else float(geometry_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("geometry_scale must be finite and positive")

    inputs = document.get("inputs") or {}
    lamp_masks = inputs.get("lamp_masks") or []
    lights: list[MeshAreaLight] = []

    def append_record(record: dict[str, Any], *, visible: bool, default_id: int) -> None:
        geometry_name = record.get("geometry_file")
        if not geometry_name:
            raise ValueError(f"Lamp {default_id} does not provide geometry_file")
        light_id = int(record.get("id", default_id))
        mask_path = None
        if visible and 0 <= light_id < len(lamp_masks):
            mask_path = _optional_path(lamp_masks[light_id], json_dir)
        lights.append(
            MeshAreaLight(
                id=light_id,
                light_type="visible_lamp" if visible else "invisible_lamp",
                center=_vector(record.get("center"), 3, "center"),
                rgb=_vector(record.get("src"), 3, "src"),
                geometry_path=_resolve_geometry(
                    str(geometry_name),
                    json_dir=json_dir,
                    prediction_dir=prediction_dir,
                ),
                visible=visible,
                geometry_scale=scale,
                confidence=float(record.get("confidence", 1.0)),
                mask_path=mask_path,
            )
        )

    raw_lights = document["lights"]
    if include_visible:
        for index, record in enumerate(raw_lights.get("visible_lamps") or []):
            append_record(record, visible=True, default_id=index)
    if include_invisible:
        invisible = raw_lights.get("invisible_lamp")
        if invisible:
            records = invisible if isinstance(invisible, list) else [invisible]
            for index, record in enumerate(records):
                append_record(record, visible=False, default_id=index)

    if not lights:
        raise ValueError("No supported visible/invisible lamps were found")

    return ILELightSet(
        source_path=json_path,
        lights=lights,
        image_path=_optional_path(inputs.get("image"), json_dir),
        depth_path=_optional_path(inputs.get("depth"), json_dir),
        schema_version=document.get("schema_version", 1),
        metadata={
            "prediction_dir": str(prediction_dir) if prediction_dir else None,
            "geometry_scale": scale,
            "windows_ignored": bool(
                raw_lights.get("visible_windows") or raw_lights.get("invisible_window")
            ),
        },
    )
