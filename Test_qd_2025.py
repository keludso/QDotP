# this is to test the functions 

from mesh_read import Mesh_read
from materials import Ge, SiGe80, Al2O3
from Poisson_Solver import nonlinear_solver
from Luttinger_Solver import LK_hamiltonian_4band


filename = "qd_2025.msh2"

# Reading mesh - domain, cell_tags, facets 
device = Mesh_read(filename)

# setting region to material parameters 

materials_map = {
    "cap": Al2O3,
    "barrier_dot": SiGe80,
    "barrier": SiGe80,
    "two_deg_dot": Ge,
    "two_deg": Ge,
    "relaxed_barrier": SiGe80,
    "relaxed_dot": SiGe80,
    "substrate": SiGe80,
}

device.set_materials(materials_map)

# --- Non-Linear Solver ---- 

psolver = nonlinear_solver(device.domain,device, 0.0)



# Only dirichlet conditions for now 
Gate_voltages = {
        "top_gate_1": 1.0,
        "top_gate_2": 1.0,
        "bottom_gate": 1.0,
        "plunger_gate": 1.0
    }

Ohmic_gate = ["back_gate"]

# Creating a submesh 
dot_submesh = [ "barrier_dot","two_deg_dot","relaxed_dot"]
dot_materials = {
    "barrier_dot": SiGe80,
    "two_deg_dot": Ge,
    "relaxed_dot": SiGe80
}

phi = psolver.nonlin_solve(Gate_voltages, Ohmic_gate)

submesh,vertex_map = device.create_submesh(dot_submesh, dot_materials)



phi_sub = device.submesh_parameters(phi,submesh,vertex_map)


# Solving the Luttinger Kohn Equations 
LK_solver = LK_hamiltonian_4band(submesh,phi_sub,device.chi_sub,device.Eg_sub, device.gamma1_sub, device.gamma2_sub, device.gamma3_sub, device.delta_sub)


# After solving eigenvalue problem:
eigenvalues, eigenvectors = LK_solver.solve_LK()

#LK_solver.save_eigenvectors_xdmf(eigenvectors,eigenvalues)

#LK_solver.save_prob_densities_xdmf(eigenvectors,eigenvalues)
#LK_solver.save_ground_state_vtu(eigenvectors,eigenvalues)
LK_solver.save_probability_density_ground_state(eigenvectors,eigenvalues)
