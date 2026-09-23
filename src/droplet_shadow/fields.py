"""Electrostatic models in SI units, including ion-conserving spherical PB."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
from scipy.integrate import solve_bvp
try:
    from scipy.special import sph_harm_y as _sph_harm_y

    def spherical_harmonic(m: int, ell: int, azimuth: np.ndarray, polar: np.ndarray) -> np.ndarray:
        return _sph_harm_y(ell, m, polar, azimuth)
except ImportError:
    from scipy.special import sph_harm as _sph_harm

    def spherical_harmonic(m: int, ell: int, azimuth: np.ndarray, polar: np.ndarray) -> np.ndarray:
        return _sph_harm(m, ell, azimuth, polar)

from .config import SimulationConfig
from .constants import COULOMB_K, E_CHARGE, EPS0, K_B


def _points(points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    a = np.asarray(points, dtype=float)
    if a.shape[-1] != 3:
        raise ValueError("Points must have a final axis of length 3")
    return a.reshape(-1, 3), a.shape[:-1]


def rayleigh_charge_e(radius_m: float, surface_tension_n_m: float) -> float:
    return 8 * math.pi * math.sqrt(EPS0 * surface_tension_n_m * radius_m**3) / E_CHARGE


class Field(Protocol):
    def potential(self, points: np.ndarray) -> np.ndarray: ...
    def electric_field(self, points: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RadialShells:
    radius_m: float
    shell_radii_m: tuple[float, ...]
    shell_charges_c: tuple[float, ...]
    epsilon_inside: float = 78.0
    epsilon_outside: float = 1.0

    def __post_init__(self) -> None:
        if not self.shell_radii_m or len(self.shell_radii_m) != len(self.shell_charges_c):
            raise ValueError("Shell radii and charges must be nonempty and equally sized")
        if any(r <= 0 or r > self.radius_m for r in self.shell_radii_m):
            raise ValueError("All shells must lie inside or on the droplet")
        if list(self.shell_radii_m) != sorted(self.shell_radii_m):
            raise ValueError("Shell radii must be sorted")

    @property
    def total_charge_c(self) -> float:
        return sum(self.shell_charges_c)

    def _radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.zeros_like(r)
        # Potential is integrated inward from infinity with the dielectric
        # interfaces held at the shell positions. This also handles neutral pairs.
        phi = np.full_like(r, COULOMB_K * self.total_charge_c / (self.epsilon_outside * self.radius_m))
        for i in range(len(self.shell_radii_m) - 1, -1, -1):
            outer = self.radius_m if i == len(self.shell_radii_m) - 1 else self.shell_radii_m[i + 1]
            inner = self.shell_radii_m[i]
            enclosed = sum(self.shell_charges_c[: i + 1])
            r_clip = np.clip(r, inner, outer)
            phi += COULOMB_K * enclosed / self.epsilon_inside * (1 / r_clip - 1 / outer)
        outside = r >= self.radius_m
        phi[outside] = COULOMB_K * self.total_charge_c / (self.epsilon_outside * r[outside])
        for radius, charge in zip(self.shell_radii_m, self.shell_charges_c):
            q += charge * (r >= radius)
        eps = np.where(outside, self.epsilon_outside, self.epsilon_inside)
        e_rad = np.divide(COULOMB_K * q, eps * r**2, out=np.zeros_like(r), where=r > 0)
        return phi, e_rad

    def potential(self, points: np.ndarray) -> np.ndarray:
        p, shape = _points(points)
        return self._radial(np.linalg.norm(p, axis=1))[0].reshape(shape)

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        p, shape = _points(points)
        r = np.linalg.norm(p, axis=1)
        er = self._radial(r)[1]
        out = p * np.divide(er, r, out=np.zeros_like(r), where=r > 0)[:, None]
        return out.reshape((*shape, 3))


@dataclass(frozen=True)
class UniformVolume:
    radius_m: float
    charge_c: float
    epsilon_inside: float = 78.0
    epsilon_outside: float = 1.0

    def _radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = self.radius_m
        inside = r < R
        phi = COULOMB_K * self.charge_c / (self.epsilon_outside * np.maximum(r, R))
        phi[inside] = COULOMB_K * self.charge_c * (
            1 / (self.epsilon_outside * R) + (R**2 - r[inside] ** 2) / (2 * self.epsilon_inside * R**3)
        )
        er = COULOMB_K * self.charge_c / (
            self.epsilon_outside * np.maximum(r, R) ** 2
        )
        er[inside] = COULOMB_K * self.charge_c * r[inside] / (self.epsilon_inside * R**3)
        return phi, er

    def potential(self, points: np.ndarray) -> np.ndarray:
        p, shape = _points(points)
        return self._radial(np.linalg.norm(p, axis=1))[0].reshape(shape)

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        p, shape = _points(points)
        r = np.linalg.norm(p, axis=1)
        er = self._radial(r)[1]
        return (p * np.divide(er, r, out=np.zeros_like(r), where=r > 0)[:, None]).reshape((*shape, 3))


@dataclass(frozen=True)
class TabulatedRadial:
    radius_m: float
    radii_m: np.ndarray
    potential_v: np.ndarray
    field_v_m: np.ndarray
    charge_c: float
    epsilon_outside: float = 1.0

    def _radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inside = r <= self.radius_m
        phi = np.empty_like(r)
        er = np.empty_like(r)
        phi[inside] = np.interp(r[inside], self.radii_m, self.potential_v)
        er[inside] = np.interp(r[inside], self.radii_m, self.field_v_m)
        phi[~inside] = COULOMB_K * self.charge_c / (self.epsilon_outside * r[~inside])
        er[~inside] = COULOMB_K * self.charge_c / (self.epsilon_outside * r[~inside] ** 2)
        return phi, er

    def potential(self, points: np.ndarray) -> np.ndarray:
        p, shape = _points(points)
        return self._radial(np.linalg.norm(p, axis=1))[0].reshape(shape)

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        p, shape = _points(points)
        r = np.linalg.norm(p, axis=1)
        er = self._radial(r)[1]
        return (p * np.divide(er, r, out=np.zeros_like(r), where=r > 0)[:, None]).reshape((*shape, 3))


def poisson_boltzmann(config: SimulationConfig, nodes: int = 320) -> TabulatedRadial:
    """Spherical fixed-number PB solution, with a pinned surface monolayer."""
    g, ch = config.geometry, config.charge
    R = g.radius_m
    species = [(1, ch.ion_positive_count), (-1, ch.ion_negative_count)]
    species = [(z, n) for z, n in species if n > 0]
    expected_qe = ch.surface_fixed_e + ch.ion_positive_count - ch.ion_negative_count
    if abs(expected_qe - ch.q_e) > 1e-7 * max(1.0, abs(ch.q_e)):
        raise ValueError("PB q_e must equal surface_fixed_e + positive ions - negative ions")
    if not species:
        return TabulatedRadial(R, np.array([0, R]),
                               np.full(2, COULOMB_K * ch.q_e * E_CHARGE / R),
                               np.zeros(2), ch.q_e * E_CHARGE)
    nref = max(n for _, n in species)
    u_edge = E_CHARGE * COULOMB_K * ch.q_e * E_CHARGE / (K_B * g.temperature_k * R)
    beta = 3 * E_CHARGE**2 * nref / (4 * math.pi * EPS0 * g.epsilon_water * K_B * g.temperature_k * R)
    x = np.unique(np.r_[np.linspace(1e-6, 0.95, nodes // 2),
                          1 - np.geomspace(1e-7, 0.05, nodes // 2)[::-1]])
    charges = np.array([z for z, _ in species])
    targets = np.array([n / nref for _, n in species])

    def fun(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        dens = np.exp(np.clip(p[:, None] - charges[:, None] * y[0], -100, 100))
        return np.vstack((y[1] / t**2,
                          -beta * t**2 * (charges[:, None] * dens).sum(axis=0),
                          3 * t[None, :] ** 2 * dens))

    def bc(ya: np.ndarray, yb: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.r_[ya[1], ya[2:], yb[0] - u_edge, yb[2:] - targets]

    y0 = np.zeros((2 + len(species), x.size))
    y0[0] = u_edge
    y0[2:] = targets[:, None] * x[None, :] ** 3
    p0 = np.log(targets) + charges * u_edge
    sol = solve_bvp(fun, bc, x, y0, p=p0, tol=2e-4,
                    max_nodes=20000, verbose=0)
    if not sol.success:
        raise RuntimeError(f"Poisson-Boltzmann solver failed: {sol.message}")
    radii = np.r_[0.0, sol.x * R]
    u = np.r_[sol.y[0, 0], sol.y[0]]
    # h/x² equals du/dx. The center field is zero by spherical symmetry.
    er = -K_B * g.temperature_k / (E_CHARGE * R) * np.r_[0.0, sol.y[1] / sol.x**2]
    return TabulatedRadial(R, radii, u * K_B * g.temperature_k / E_CHARGE,
                           er, ch.q_e * E_CHARGE)


class SumField:
    def __init__(self, *fields: Field):
        self.fields = tuple(fields)

    def potential(self, points: np.ndarray) -> np.ndarray:
        return sum((f.potential(points) for f in self.fields), start=np.zeros(np.asarray(points).shape[:-1]))

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        return sum((f.electric_field(points) for f in self.fields), start=np.zeros_like(np.asarray(points, dtype=float)))


def build_field(config: SimulationConfig) -> Field:
    g, c = config.geometry, config.charge
    R = g.radius_m
    if c.model == "surface":
        base: Field = RadialShells(R, (R,), (c.q_e * E_CHARGE,), g.epsilon_water)
    elif c.model == "volume":
        base = UniformVolume(R, c.q_e * E_CHARGE, g.epsilon_water)
    elif c.model == "neutral_double_layer":
        inner = R - c.double_layer_thickness_nm * 1e-9
        if inner <= 0:
            raise ValueError("Double layer is thicker than the droplet radius")
        base = RadialShells(R, (inner, R),
                            (c.double_layer_q_e * E_CHARGE, -c.double_layer_q_e * E_CHARGE),
                            g.epsilon_water)
        if c.q_e:
            base = SumField(base, RadialShells(R, (R,), (c.q_e * E_CHARGE,), g.epsilon_water))
    else:
        base = poisson_boltzmann(config)
    fields: list[Field] = [base]
    if c.model in {"surface", "volume"} and c.double_layer_q_e:
        inner = R - c.double_layer_thickness_nm * 1e-9
        if inner <= 0:
            raise ValueError("Double layer is thicker than the droplet radius")
        fields.append(RadialShells(R, (inner, R),
                                   (c.double_layer_q_e * E_CHARGE,
                                    -c.double_layer_q_e * E_CHARGE),
                                   g.epsilon_water))
    if c.dipole_potential_v:
        inner = R - c.dipole_thickness_nm * 1e-9
        if inner <= 0:
            raise ValueError("Dipole layer is thicker than the droplet radius")
        q = c.dipole_potential_v * c.dipole_epsilon / (COULOMB_K * (1 / inner - 1 / R))
        fields.append(RadialShells(R, (inner, R), (q, -q), c.dipole_epsilon))
    if c.dipole_fraction or c.quadrupole_fraction or c.patch_fraction:
        fields.append(SurfaceMultipoles(config))
    return fields[0] if len(fields) == 1 else SumField(*fields)


class SurfaceMultipoles:
    """Frozen asymmetric surface charge with a zero monopole component."""

    def __init__(self, config: SimulationConfig):
        g, c = config.geometry, config.charge
        self.radius_m = g.radius_m
        self.epsilon_inside = g.epsilon_water * EPS0
        self.epsilon_outside = EPS0
        order = c.multipole_order
        u, w = np.polynomial.legendre.leggauss(max(2 * order + 8, 32))
        phi = np.linspace(0, 2 * np.pi, max(4 * order + 16, 64), endpoint=False)
        uu, pp = np.meshgrid(u, phi, indexing="ij")
        nx = np.sqrt(1 - uu**2) * np.cos(pp)
        sigma0 = c.q_e * E_CHARGE / (4 * np.pi * self.radius_m**2)
        patch_center = np.cos(np.arccos(np.clip(nx, -1, 1)))
        patch = np.exp((patch_center - 1) / (2 * np.sin(np.deg2rad(c.patch_width_deg) / 2) ** 2))
        patch -= np.average(patch, weights=np.broadcast_to(w[:, None], patch.shape))
        patch_base = (abs(c.q_e) if c.q_e else 1e6) * E_CHARGE / (4 * np.pi * self.radius_m**2)
        sigma = sigma0 * (c.dipole_fraction * nx + c.quadrupole_fraction * (3 * nx**2 - 1) / 2)
        sigma += c.patch_fraction * patch_base * patch
        self.coefficients: dict[tuple[int, int], complex] = {}
        weights = w[:, None] * (2 * np.pi / phi.size)
        theta = np.arccos(uu)
        for ell in range(1, order + 1):
            for m in range(-ell, ell + 1):
                value = np.sum(sigma * np.conj(spherical_harmonic(m, ell, pp, theta)) * weights)
                if abs(value) > 1e-16 * max(abs(sigma0), abs(patch_base)):
                    self.coefficients[ell, m] = value * self.radius_m ** (ell + 2) / (
                        self.epsilon_outside * (ell + 1) + self.epsilon_inside * ell)

    def potential(self, points: np.ndarray) -> np.ndarray:
        xyz, shape = _points(points)
        r = np.linalg.norm(xyz, axis=1)
        theta = np.arccos(np.divide(xyz[:, 2], r, out=np.ones_like(r), where=r > 0))
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        result = np.zeros(len(r), dtype=complex)
        inside = r < self.radius_m
        for (ell, m), a in self.coefficients.items():
            angular = spherical_harmonic(m, ell, azimuth, theta)
            radial = np.empty_like(r, dtype=complex)
            radial[inside] = a * (r[inside] / self.radius_m) ** ell / self.radius_m ** (ell + 1)
            radial[~inside] = a / r[~inside] ** (ell + 1)
            result += radial * angular
        return result.real.reshape(shape)

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        xyz, shape = _points(points)
        h = self.radius_m * 2e-5
        central_inside = np.linalg.norm(xyz, axis=1) < self.radius_m
        phi0 = self.potential(xyz)
        grads = []
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = h
            plus = xyz + delta
            minus = xyz - delta
            phi_plus = self.potential(plus)
            phi_minus = self.potential(minus)
            derivative = (phi_plus - phi_minus) / (2 * h)
            plus_crosses = (np.linalg.norm(plus, axis=1) < self.radius_m) != central_inside
            minus_crosses = (np.linalg.norm(minus, axis=1) < self.radius_m) != central_inside
            derivative = np.where(plus_crosses & ~minus_crosses,
                                  (phi0 - phi_minus) / h, derivative)
            derivative = np.where(minus_crosses & ~plus_crosses,
                                  (phi_plus - phi0) / h, derivative)
            grads.append(derivative)
        return -np.stack(grads, axis=-1).reshape((*shape, 3))


def multipole_convergence_error(config: SimulationConfig) -> float | None:
    """Relative exterior-field RMS change on refining the spherical harmonics."""
    charge = config.charge
    if not (charge.dipole_fraction or charge.quadrupole_fraction or charge.patch_fraction):
        return None
    lower = charge.multipole_order if charge.multipole_order < 16 else 12
    upper = min(lower + 4, 16)
    index = np.arange(64) + 0.5
    z = 1 - 2 * index / 64
    azimuth = np.pi * (3 - np.sqrt(5)) * index
    xy = np.sqrt(1 - z*z)
    unit = np.column_stack((xy * np.cos(azimuth), xy * np.sin(azimuth), z))
    points = np.concatenate((unit * 1.02 * config.geometry.radius_m,
                             unit * 2.0 * config.geometry.radius_m))
    coarse = SurfaceMultipoles(config.with_updates(charge={"multipole_order": lower}))
    fine = SurfaceMultipoles(config.with_updates(charge={"multipole_order": upper}))
    a = coarse.electric_field(points)
    b = fine.electric_field(points)
    return float(np.sqrt(np.mean((a - b)**2)) / max(np.sqrt(np.mean(b**2)), 1e-30))
