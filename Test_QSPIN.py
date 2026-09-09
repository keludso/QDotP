# Example test file using the refactored kdotP package structure.
# Strain sweep version: saves strain vs Rabi frequency.

import numpy as np
import traceback

from kdotP.base import Device
from kdotP.base.materials import Si, SiO2
from kdotP.poisson_solver import NonlinearPoissonSolver
from kdotP.Schrodinger_Solver import Schrodinger_Solver
from savedat import savedat
from kdotP.Rabi import Rabi_solver


# ============================================================
# User settings
# ============================================================

filename = "Sven_device_v1.msh2"

B = [0.0, 0.0, 0.3]
Vac = 1e-3

results_file = "rabi_vs_strain.txt"


# ============================================================
# Read mesh and create device
# ============================================================

device = Device(filename, verbose=True)

logger = savedat("test_rabi_strain_sweep.dat")
device.set_logger(logger)


# ============================================================
# Set material parameters by named region
# ============================================================

materials_map = {
    "SiO2_cap": SiO2,
    "Si_substrate": Si,
    "Si_substrate_QD": Si,
    "BOX": SiO2,
    "BOX_QD": SiO2,
    "Epi_Si": Si,
    "Epi_Si_QD": Si,
    "Thermal_SiO2": SiO2,
    "Thermal_SiO2_QD": SiO2,
}

device.set_materials(materials_map)


# ============================================================
# Nonlinear Poisson solve
# This is done once because the electrostatic potential is
# assumed independent of strain.
# ============================================================

psolver = NonlinearPoissonSolver(
    device,
    0.0,
    verbose=True,
    log_writer=logger,
)

Gate_voltages = {
    "Barrier_left": -0.5,
    "Plunger": -0.1,
    "Barrier_right": -0.5,
}

Ohmic_gate = ["bottom_surface"]

phi = psolver.nonlin_solve(
    Gate_voltages,
    Ohmic_gate,
    save_file=True,
    output_file="poisson.vtu",
)


# ============================================================
# Create submesh for dot region
# ============================================================

dot_submesh = [
    "Epi_Si_QD",
    "BOX_QD",
    "Thermal_SiO2_QD",
]

dot_materials = {
    "Epi_Si_QD": Si,
    "BOX_QD": SiO2,
    "Thermal_SiO2_QD": SiO2,
}

submesh, vertex_map = device.create_submesh(dot_submesh, dot_materials)

phi_sub = device.submesh_parameters(
    phi,
    submesh,
    vertex_map,
)


ssolver = Schrodinger_Solver(
    device,
    submesh,
    phi_sub,
    log_writer=logger,
    verbose=True,
)

eigenvalues, eigenvectors = ssolver.solve_schrodinger()
ssolver.save_probability_density_ground_state(
    eigenvectors, eigenvalues, output_file="ground_state_6band.vtu"
)


