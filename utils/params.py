from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ParamSpec:
    key: str
    label: str
    kind: str  # int|float|choice
    minimum: float | int
    maximum: float | int
    step: float | int
    default: Any
    choices: list[str] | None = None


def _extract_constants(py_file: str) -> Dict[str, Any]:
    with open(py_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=py_file)

    values: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                try:
                    values[name] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return values


def load_param_specs(classical_file: str, quantum_file: str) -> Dict[str, List[ParamSpec]]:
    c = _extract_constants(classical_file)
    q = _extract_constants(quantum_file)

    classical_specs = [
        ParamSpec("potential", "Potential", "choice", 0, 0, 1, c.get("POTENTIAL", "yukawa"), ["yukawa", "soft_sphere", "coulomb", "lennard_jones"]),
        ParamSpec("n_particles", "Particles", "int", 10, 300, 1, int(c.get("N_PARTICLES", 90))),
        ParamSpec("b_max", "Impact parameter b", "float", 0.2, 10.0, 0.05, float(c.get("B_MAX", 4.5))),
        ParamSpec("v0", "Initial velocity", "float", 0.1, 10.0, 0.1, float(c.get("V0", 1.0))),
        ParamSpec("mass", "Mass", "float", 0.1, 20.0, 0.1, 1.0),
        ParamSpec("beta", "Potential strength beta", "float", -50.0, 50.0, 0.1, float(c.get("BETA", 3.0))),
        ParamSpec("mu", "Yukawa mu", "float", 0.01, 10.0, 0.01, float(c.get("MU", 0.6))),
        ParamSpec("sphere_v0", "Soft-sphere V0", "float", -100.0, 200.0, 0.1, float(c.get("SPHERE_V0", 8.0))),
        ParamSpec("sphere_a", "Soft-sphere radius a", "float", 0.1, 10.0, 0.05, float(c.get("SPHERE_A", 1.2))),
        ParamSpec("lj_eps", "LJ epsilon", "float", 0.01, 30.0, 0.01, float(c.get("LJ_EPS", 1.5))),
        ParamSpec("lj_sigma", "LJ sigma", "float", 0.01, 10.0, 0.01, float(c.get("LJ_SIGMA", 1.0))),
        ParamSpec("dt", "Time step dt", "float", 0.001, 0.1, 0.001, float(c.get("DT", 0.015))),
        ParamSpec("total_time", "Total simulation time", "float", 1.0, 100.0, 0.1, float(c.get("N_STEPS", 900)) * float(c.get("DT", 0.015))),
    ]

    quantum_specs = [
        ParamSpec("potential_type", "Potential type", "choice", 0, 0, 1, "barrier", ["barrier", "well"]),
        ParamSpec("sigma_x", "Wave packet width sigma_x", "float", 0.1, 5.0, 0.05, float(q.get("SIGMA_X", 0.9))),
        ParamSpec("sigma_z", "Wave packet width sigma_z", "float", 0.1, 5.0, 0.05, float(q.get("SIGMA_Z", 1.1))),
        ParamSpec("k0", "Initial momentum k", "float", 0.1, 20.0, 0.1, float(q.get("K0", 5.0))),
        ParamSpec("soft_v0", "Potential strength", "float", 0.1, 200.0, 0.1, float(q.get("SOFT_V0", 30.0))),
        ParamSpec("nx", "Grid size NX", "int", 64, 1024, 1, int(q.get("NX", 384))),
        ParamSpec("nz", "Grid size NZ", "int", 64, 1024, 1, int(q.get("NZ", 512))),
        ParamSpec("dt", "Time step dt", "float", 0.0005, 0.05, 0.0005, float(q.get("DT", 0.004))),
        ParamSpec("total_time", "Total time", "float", 0.2, 30.0, 0.1, float(q.get("N_FRAMES", 210)) * float(q.get("STEPS_PER_FRAME", 10)) * float(q.get("DT", 0.004))),
    ]

    return {
        "Classical Scattering": classical_specs,
        "Quantum Scattering": quantum_specs,
    }
