"""Mitsuba emitter for IndoorLightEditing directional window lights."""

from __future__ import annotations

import os
from typing import Any


_REGISTERED_VARIANTS: dict[str, type] = {}
_SAMPLE_WEIGHT_CLAMP = float(
    os.environ.get("MATERIALIST_WINDOW_SAMPLE_WEIGHT_CLAMP", "100.0")
)


def register_ile_window_emitter(mi: Any) -> None:
    """Register the emitter for the active Mitsuba variant once.

    ILE models each window as a finite rectangle whose radiance is the sum of
    three spherical Gaussians (sun, sky, and ground). Mitsuba's ordinary area
    emitter is direction independent, so the SG evaluation must live in the
    emitter rather than in a spatial texture.
    """
    import drjit as dr

    variant = mi.variant()
    if variant in _REGISTERED_VARIANTS:
        return

    class ILEWindowEmitter(mi.Emitter):
        def __init__(self, props):
            super().__init__(props)
            self.m_flags = +mi.EmitterFlags.Surface
            self.m_needs_sample_2 = True
            self.m_needs_sample_3 = True
            self._rgb = []
            self._direction = []
            self._concentration = []
            for name in ("sun", "sky", "ground"):
                self._rgb.append(mi.Color3f(props.get(f"{name}_rgb")))
                direction = mi.Vector3f(props.get(f"{name}_direction"))
                self._direction.append(dr.normalize(direction))
                self._concentration.append(
                    mi.Float(props.get(f"{name}_concentration"))
                )

        def _radiance(self, direction):
            result = mi.Color3f(0.0)
            for rgb, axis, concentration in zip(
                self._rgb, self._direction, self._concentration
            ):
                exponent = concentration * dr.minimum(
                    dr.dot(axis, direction) - 1.0, 0.0
                )
                result += rgb * dr.exp(exponent)
            return result

        def sample_direction(self, it, sample, active=True):
            shape = self.get_shape()
            ds = shape.sample_direction(it, sample, active)
            valid = active & (dr.dot(ds.d, ds.n) < 0.0) & (ds.pdf != 0.0)
            ds.emitter = (
                mi.EmitterPtr(self)
                if hasattr(mi, "EmitterPtr")
                else self
            )
            weight = dr.select(valid, self._radiance(ds.d) / ds.pdf, 0.0)
            # OBJ/PLY shape plugins do not expose per-shape ray intersection
            # on the CUDA backend, so exact SG/aperture mixture sampling cannot
            # be implemented inside this Python emitter. Bound the remaining
            # heavy tail from very concentrated lobes (lambda > 500 in
            # Example1). Set the environment variable to 0 to disable it.
            if _SAMPLE_WEIGHT_CLAMP > 0:
                weight = dr.minimum(
                    weight,
                    mi.Color3f(_SAMPLE_WEIGHT_CLAMP),
                )
            return ds, weight

        def pdf_direction(self, it, ds, active=True):
            active &= dr.dot(ds.d, ds.n) < 0.0
            pdf = self.get_shape().pdf_direction(it, ds, active)
            return dr.select(active, pdf, 0.0)

        def eval_direction(self, it, ds, active=True):
            active &= dr.dot(ds.d, ds.n) < 0.0
            return dr.select(active, self._radiance(ds.d), 0.0)

        def eval(self, si, active=True):
            active &= mi.Frame3f.cos_theta(si.wi) > 0.0
            # si.wi points from the window back to the preceding path vertex;
            # ILE's SG convention uses the reverse, point-to-window direction.
            direction_to_window = -si.to_world(si.wi)
            return dr.select(active, self._radiance(direction_to_window), 0.0)

        def sample_position(self, time, sample, active=True):
            ps = self.get_shape().sample_position(time, sample, active)
            weight = dr.select(active & (ps.pdf > 0.0), dr.rcp(ps.pdf), 0.0)
            return ps, weight

        def pdf_position(self, ps, active=True):
            return self.get_shape().pdf_position(ps, active)

        def sample_ray(
            self, time, wavelength_sample, position_sample, direction_sample, active=True
        ):
            ps, position_weight = self.sample_position(
                time, position_sample, active
            )
            si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
            local_direction = mi.warp.square_to_cosine_hemisphere(direction_sample)
            world_direction = si.to_world(local_direction)
            weight = (
                position_weight
                * dr.pi
                * self._radiance(-world_direction)
            )
            return si.spawn_ray(world_direction), weight

        def bbox(self):
            return self.get_shape().bbox()

        def to_string(self):
            return "ILEWindowEmitter[sun + sky + ground SG]"

    mi.register_emitter("ile_window", lambda props: ILEWindowEmitter(props))
    _REGISTERED_VARIANTS[variant] = ILEWindowEmitter
