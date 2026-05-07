# Example test file using the refactored kdotP package structure.
# The overall flow is kept close to your original Test_qd_2025.py.

from kdotP.base import Device
from kdotP.base.materials import Ge, SiGe80, Al2O3, GeSn15, GeSn08
from kdotP.poisson_solver import NonlinearPoissonSolver
from kdotP.luttinger_solver import LK4BandHamiltonian
from kdotP.luttinger_solver import LK_hamiltonian_8band
from kdotP.luttinger_solver import LK_hamiltonian_6band
import numpy as np
#from materials import Ge, SiGe80, Al2O3
from savedat import savedat

filename = "qd_2025.msh2"

# Reading mesh - domain, cell_tags, facets
device = Device(filename, verbose=True)

# Set logger
logger = savedat("test_GeSn.dat")
device.set_logger(logger)

# Set material parameters by named region
materials_map = {
    "cap": Al2O3,
    "barrier_dot": Ge,
    "barrier": Ge,
    "two_deg_dot": GeSn15,
    "two_deg": GeSn15,
    "relaxed_barrier": Ge,
    "relaxed_dot": Ge,
    "substrate": GeSn08,
}
device.set_materials(materials_map)

# Nonlinear Poisson solve
psolver = NonlinearPoissonSolver(device, 0.0, verbose=True, log_writer=logger)
Gate_voltages = {
    "top_gate_1": 0.0,
    "top_gate_2": 0.0,
    "bottom_gate": 0.0,
    "plunger_gate": 0.0,
}
Ohmic_gate = ["back_gate"]
phi = psolver.nonlin_solve(Gate_voltages, Ohmic_gate)

# Create a submesh for the dot region
# This part is shared through the Device class in kdotP.base.
dot_submesh = ["barrier_dot", "two_deg_dot", "relaxed_dot"]
dot_materials = {
    "barrier_dot": Ge,
    "two_deg_dot": GeSn15,
    "relaxed_dot": Ge,
}
submesh, vertex_map = device.create_submesh(dot_submesh, dot_materials)
phi_sub = device.submesh_parameters(phi, submesh, vertex_map)


lk_solver = LK_hamiltonian_6band(
    device,
    submesh,
    phi_sub,
    log_writer=logger,
    verbose=True,
)



def lattice_GeSnx(x):
    a_Ge = 5.657956
    a_Sn = 6.489417
    bowling = -0.083
    return a_Ge*(1-x) + a_Sn*x - bowling*x*(1-x)

def strain_GeSn_to_Ge(strain_GeSn, x_GeSn):
    GeSn_lattice = lattice_GeSnx(x_GeSn)
    a_parallel = (strain_GeSn+1)*GeSn_lattice
    return (a_parallel - 5.657956)/5.657956


# Writing for all the strains 
R = np.array([0.01,0.015])

for r in R:

    lk_solver.set_region_strain(
        "two_deg_dot",
        [[r, 0.0, 0.0],
        [0.0, r, 0.0],
        [0.0, 0.0, 0.682505*r]]
    )

    r_Ge = strain_GeSn_to_Ge(r, 0.15)

    ezz = -2*r*0.37

    lk_solver.set_region_strain(
        "barrier_dot",
        [[r, 0.0, 0.0],
        [0.0, r, 0.0],
        [0.0, 0.0, ezz]]
    )


    #B = [0, 0, 0.2]
    #lk_solver.set_BField(B)


    eigenvalues, eigenvectors = lk_solver.solve_LK()
    lk_solver.save_probability_density_ground_state(
        eigenvectors, eigenvalues, output_file="ground_state_6band_mmstrain.vtu"
    )

