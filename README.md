# kdotP: FEM k·p / Poisson Solver for Quantum Dot Heterostructures

## 1. Introduction

`kdotP` is a Python-based finite-element simulation package for semiconductor quantum-dot heterostructures. The package is designed to solve the electrostatic potential of realistic gate-defined devices and use this potential as an input to multiband k·p Hamiltonian calculations.

The main workflow begins with a Gmsh-generated device mesh. The mesh is read into a `Device` object, where physical regions are assigned material properties such as Ge, SiGe, or Al₂O₃. The nonlinear Poisson equation is then solved on the full device geometry using gate voltage boundary conditions. After the electrostatic potential is obtained, the active quantum-dot region is extracted as a submesh. The electrostatic potential is mapped onto this submesh and used in the 4-band, 6-band, or 8-band k·p Hamiltonian solver.

This package is intended for research-level simulations of Ge/SiGe and related strained semiconductor quantum-dot systems. It can be used to study electrostatic confinement, strain effects, magnetic-field dependence, heavy-hole/light-hole mixing, eigenenergies, and quantum-dot probability densities.

---

## 2. Requirements

The code requires a scientific Python environment with finite-element and eigenvalue solver support.

Typical requirements are:

```bash
python >= 3.10
numpy
mpi4py
petsc4py
slepc4py
dolfinx
basix
ufl
gmsh
pyvista
```

For serial execution:

```bash
python test_6band.py
```

For MPI execution:

```bash
mpirun -np 4 python test_6band.py
```

The input mesh should be generated using Gmsh and saved in `.msh2` format. The physical region names and gate boundary names in the mesh must match the names used in the Python script.

Example physical region names include:

```text
cap
barrier_dot
barrier
two_deg_dot
two_deg
relaxed_barrier
relaxed_dot
substrate
```

Example gate or contact names include:

```text
top_gate_1
top_gate_2
bottom_gate
plunger_gate
back_gate
```

---

## 3. Example File

The following example demonstrates the basic workflow for solving the nonlinear Poisson equation, extracting the active dot region, solving the 6-band k·p Hamiltonian, and saving the ground-state probability density.

```python
from kdotP.base import Device
from kdotP.base.materials import Ge, SiGe80, Al2O3
from kdotP.poisson_solver import NonlinearPoissonSolver
from kdotP.luttinger_solver import LK_hamiltonian_6band
import numpy as np
from savedat import savedat


filename = "qd_2025.msh2"

# Read mesh: domain, cell tags, and facet tags
device = Device(filename, verbose=True)

# Set logger
logger = savedat("test_6band_strain_magnetic_field.dat")
device.set_logger(logger)

# Assign materials by Gmsh physical region name
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

# Create nonlinear Poisson solver
psolver = NonlinearPoissonSolver(
    device,
    0.0,
    verbose=True,
    log_writer=logger,
)

# Define gate voltages
Gate_voltages = {
    "top_gate_1": 0.0,
    "top_gate_2": 0.0,
    "bottom_gate": 0.0,
    "plunger_gate": 0.0,
}

Ohmic_gate = ["back_gate"]

# Solve nonlinear Poisson equation
phi = psolver.nonlin_solve(
    Gate_voltages,
    Ohmic_gate,
    save_file=True,
    output_file="poisson.vtu",
)

# Create a submesh for the active quantum-dot region
dot_submesh = ["barrier_dot", "two_deg_dot", "relaxed_dot"]

dot_materials = {
    "barrier_dot": SiGe80,
    "two_deg_dot": Ge,
    "relaxed_dot": SiGe80,
}

submesh, vertex_map = device.create_submesh(dot_submesh, dot_materials)

# Map Poisson potential from full mesh to dot submesh
phi_sub = device.submesh_parameters(phi, submesh, vertex_map)

# Create 6-band k·p Hamiltonian solver
lk_solver = LK_hamiltonian_6band(
    device,
    submesh,
    phi_sub,
    log_writer=logger,
    verbose=True,
)

# Solve 6-band Hamiltonian
eigenvalues, eigenvectors = lk_solver.solve_LK()

# Save ground-state probability density
lk_solver.save_probability_density_ground_state(
    eigenvectors,
    eigenvalues,
    output_file="ground_state_6band.vtu",
)

# Example strain sweep
R = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])

for r in R:

    ezz = -2 * r * 0.37

    lk_solver.set_region_strain(
        "two_deg_dot",
        [
            [r,   0.0, 0.0],
            [0.0, r,   0.0],
            [0.0, 0.0, ezz],
        ],
    )

    B = [0.0, 0.0, 0.2]
    lk_solver.set_BField(B)

    eigenvalues, eigenvectors = lk_solver.solve_LK()

    lk_solver.save_probability_density_ground_state(
        eigenvectors,
        eigenvalues,
        output_file=f"ground_state_6band_strain_{r:+.4f}.vtu",
    )
```

---
## 4. Description of the Nonlinear Poisson and kdotP Method
## 4. Description of the Nonlinear Poisson and kdotP Method

The `kdotP` simulation workflow combines electrostatic device modeling with multiband k·p quantum calculations. The calculation starts from a full three-dimensional device mesh that includes the dielectric layers, semiconductor regions, electrostatic gates, ohmic contacts, and the active quantum-well or quantum-dot region.

The nonlinear Poisson solver computes the electrostatic potential over the full device geometry. The solver uses the material properties assigned through the `Device` object and applies voltage boundary conditions to the named gate and contact regions.

The nonlinear Poisson equation is written as:

```text
div[ ε(r) grad φ(r) ] = -ρ(φ, r)
```

where:

```text
ε(r)      = spatially dependent dielectric constant
φ(r)      = electrostatic potential
ρ(φ, r)   = nonlinear charge density
```

The charge density depends on the local electrostatic potential and material parameters, which makes the electrostatic problem nonlinear.

After the electrostatic potential is obtained, the active quantum-dot region is extracted from the full device mesh. This step reduces the size of the quantum-mechanical problem because the Poisson equation must be solved over the entire device, while the k·p Hamiltonian only needs to be solved in the region where the confined carrier states are located. The electrostatic potential from the full mesh is then mapped onto the quantum-dot submesh and used as an input to the k·p Hamiltonian solver.

The k·p method describes the interaction and coupling between semiconductor energy bands. The number of bands included in the Hamiltonian determines the level of physical accuracy and computational cost. The 4-band model includes the heavy-hole and light-hole valence bands. The 6-band model extends this by including the split-off valence bands. The 8-band model further includes conduction-band states and their coupling to the valence bands. Therefore, the appropriate model can be selected based on the level of accuracy required for the simulation.

The valence-band basis functions are written in terms of total angular momentum states as follows:

```text
|3/2,  3/2> =  (1/sqrt(2)) ( |X + iY> ↑ )

|3/2, -3/2> =  (1/sqrt(2)) ( |X - iY> ↓ )

|3/2,  1/2> =  (1/sqrt(6)) ( |X + iY> ↓ ) 
               - sqrt(2/3) |Z ↑>

|3/2, -1/2> = -(1/sqrt(6)) ( |X - iY> ↑ ) 
               - sqrt(2/3) |Z ↓>

|1/2,  1/2> =  (1/sqrt(3)) ( |X + iY> ↓ ) 
               + (1/sqrt(3)) |Z ↑>

|1/2, -1/2> = -(1/sqrt(3)) ( |X - iY> ↑ ) 
               + (1/sqrt(3)) |Z ↓>
```

### 4.1 Four-Band k·p Hamiltonian

The 4-band model is written in the heavy-hole and light-hole basis:

```text
Basis = {
  |3/2,  3/2>,
  |3/2, -3/2>,
  |3/2,  1/2>,
  |3/2, -1/2>
}
```

The 4-band Hamiltonian is:

```text
H_4x4 =

[  P + Q        0             -S_-             R        ]
[    0        P + Q        -R†              -S_+       ]
[ -S_-†       -R             P - Q            C        ]
[   R†        -S_+†           C†             P - Q      ]
```

where:

```text
P = E_V(r) + (hbar^2 / 2m_e) [ γ1 k_x^2 + γ1 k_y^2 + γ1 k_z^2 ]

Q = (hbar^2 / 2m_e) [ γ2 k_x^2 + γ2 k_y^2 - 2γ2 k_z^2 ]

R = -(hbar^2 sqrt(3) / 2m_e) k_- γ_bar k_-
    +(hbar^2 sqrt(3) / 2m_e) k_+ μ k_+

S_± = (hbar^2 sqrt(3) / m_e) [ k_±(σ - δ)k_z + k_zπk_± ]

C = (hbar^2 / m_e) [ k_z(σ - δ - π)k_- - k_-(σ - δ - π)k_z ]
```

The wave-vector and material-dependent parameters are:

```text
k_± = k_x ± i k_y

k_parallel^2 = k_x^2 + k_y^2

γ_bar = (1/2)(γ3 + γ2)

μ = (1/2)(γ3 - γ2)

σ = γ_bar - (1/2)δ

π = μ + (3/2)δ

δ = (1/9)(1 + γ1 + γ2 - 3γ3)
```

The 4-band Hamiltonian is useful when the split-off bands and conduction bands are sufficiently far away in energy and the low-energy physics is dominated by heavy-hole and light-hole states.

### 4.2 Six-Band k·p Hamiltonian

The 6-band Hamiltonian extends the 4-band model by including the split-off valence-band states. This model is useful when the split-off band contributes to the low-energy hole states or when stronger band mixing is expected.

The 6-band Hamiltonian is:

```text
H_6x6 =

[  P+Q       0          -S_-          R          (1/sqrt(2))S_-        sqrt(2)R              ]

[   0       P+Q         -R†          -S_+       -sqrt(2)R†             (1/sqrt(2))S_+        ]

[ -S_-†     -R          P-Q           C          sqrt(2)Q              sqrt(3/2)Σ_-          ]

[  R†       -S_+†        C†           P-Q       -sqrt(3/2)Σ_+          sqrt(2)Q              ]

[ (1/sqrt(2))S_-†   -sqrt(2)R     sqrt(2)Q   -sqrt(3/2)Σ_+†     P+Δ        -C              ]

[ sqrt(2)R†          (1/sqrt(2))S_+†  sqrt(3/2)Σ_-   sqrt(2)Q   -C†        P+Δ             ]
```

where:

```text
P = E_v(z) + (1/2)[ γ1 k_parallel^2 + k_z γ1 k_z ]

Q = ζ(z) + (1/2)[ γ2 k_parallel^2 - 2k_z γ2 k_z ]

R = -(sqrt(3)/2) γ_bar k_-^2 + (sqrt(3)/2) μ k_+^2

S_± = sqrt(3) k_± [ (σ - δ)k_z + k_zπ ]

Σ_± = sqrt(3) k_± { 
        [ (1/3)(σ - δ) + (2/3)π ] k_z
        + k_z [ (2/3)(σ - δ) + (1/3)π ]
      }

C = k_- [ k_z(σ - δ - π) - (σ - δ - π)k_z ]
```

with:

```text
k_parallel^2 = k_x^2 + k_y^2

k_+ = k_x + i k_y

k_- = k_x - i k_y
```

Here, `Δ` is the spin-orbit split-off energy, and `ζ(z)` represents the strain-dependent or band-edge contribution included in the model. In the finite-element implementation, the electrostatic potential from the nonlinear Poisson solver enters through the spatially dependent band-edge term.

### 4.3 Eight-Band k·p Hamiltonian

The 8-band Hamiltonian includes conduction-band states in addition to the heavy-hole, light-hole, and split-off valence-band states. This model is useful when conduction-valence coupling, narrow-gap effects, or strong confinement effects are important.

The 8-band Hamiltonian is:

```text
H_8x8 =

[ Ev+E0+(hbar^2 k^2/2m')      0              -(1/sqrt(2))P0k_+     sqrt(3/2)P0k_z       0              0              -(1/sqrt(2))P0k_-     sqrt(3/2)P0k_z ]

[ 0                            Ev+E0+(hbar^2 k^2/2m')   0          -(1/sqrt(6))P0k_+   -(1/sqrt(6))P0k_-   0          sqrt(3/2)P0k_z      -(1/sqrt(6))P0k_+ ]

[ -(1/sqrt(2))P0k_-           0              P+Q                   0                    -S_-           R              (1/sqrt(2))S_-       sqrt(2)R ]

[ sqrt(3/2)P0k_z              -(1/sqrt(6))P0k_+   0                 P+Q                  -R†            -S_+           -sqrt(2)R†            (1/sqrt(2))S_+ ]

[ 0                            -(1/sqrt(6))P0k_-   -S_-†            -R                   P-Q            C              sqrt(2)Q             sqrt(3)Σ_+ ]

[ 0                             0              R†                   -S_+†                C†             P-Q            -sqrt(3)Σ_+          sqrt(2)Q ]

[ -(1/sqrt(2))P0k_-            sqrt(3/2)P0k_z   (1/sqrt(2))S_-†     -sqrt(2)R            sqrt(2)Q       -sqrt(3)Σ_+†    P+Δ                  -C ]

[ sqrt(3/2)P0k_z               -(1/sqrt(6))P0k_+   sqrt(2)R†        (1/sqrt(2))S_+†       sqrt(3)Σ_-     sqrt(2)Q       -C†                  P+Δ ]
```

In this expression:

```text
P0 = Kane momentum matrix element
E0 = conduction-band energy offset relative to the valence band
m' = modified conduction-band effective mass parameter
```

The Kane energy is defined as:

```text
E_P = 2m0P0^2 / hbar^2
```

The Luttinger parameters can be corrected to account for conduction-band coupling as:

```text
γ1 = γ1^L - E_P / (3E_g + Δ)

γ2 = γ2^L - (1/2) E_P / (3E_g + Δ)

γ3 = γ3^L - (1/2) E_P / (3E_g + Δ)
```

where:

```text
E_g       = band gap
Δ         = spin-orbit split-off energy
γ1^L      = original γ1 Luttinger parameter
γ2^L      = original γ2 Luttinger parameter
γ3^L      = original γ3 Luttinger parameter
```

### 4.4 Finite-Element Implementation

In the finite-element implementation, the k·p Hamiltonian is assembled on the active quantum-dot submesh. The electrostatic potential obtained from the nonlinear Poisson solver enters the Hamiltonian through the position-dependent band-edge potential. This allows gate voltages and device geometry to directly affect the confined quantum states.

The Hamiltonian can also include strain and magnetic-field effects. Strain is included through region-dependent strain tensors and the corresponding Bir-Pikus deformation-potential terms. Magnetic fields can be included through Zeeman terms and, when implemented, orbital magnetic-field coupling.

The resulting eigenvalue problem has the generalized form:

```text
Aψ = E Bψ
```

where:

```text
A = assembled k·p Hamiltonian matrix
B = finite-element mass matrix
E = eigenenergy
ψ = multicomponent envelope-function eigenstate
```

For a multiband model, the physical probability density is obtained by summing over all band components:

```text
ρ(r) = Σ_i |ψ_i(r)|^2
```

For the 6-band model, this sum is taken over the six coupled envelope-function components. The probability density can then be saved as a `.vtu` file and visualized in ParaView or PyVista. This allows the user to directly compare the electrostatic potential, confined wavefunction, eigenenergies, strain response, and magnetic-field response of the simulated quantum-dot device.


## 5. Results

The simulation produces electrostatic potential maps, quantum-dot submesh results, eigenenergies, and ground-state probability-density distributions.
ith quantum-mechanical k·p modeling. This allows the user to study how gate voltage, device geometry, strain, and magnetic field affect quantum-dot confinement and low-energy hole states. The bandstructure plot across the center of the quantum dot shows a band-engineered SiGe/GE heterostructure.

<img width="1321" height="645" alt="NonLinear_poisson" src="https://github.com/user-attachments/assets/09a4ac73-649f-4873-ab27-8367b74bf8fb" />


The wavefunction calculated using the 6-band model across the Y axis showing the quantum dot residing in the Ge structure. 

<img width="970" height="757" alt="ground_statewavefunction_Yaxis" src="https://github.com/user-attachments/assets/46af5f45-a790-4a22-a1b0-872bfd7359bb" />


The wavefunction was calculated using the 6-band across the Z axis.
<img width="1119" height="770" alt="ground_statewavefunction_Zaxis" src="https://github.com/user-attachments/assets/209b70d1-dc96-4b6e-a534-ff76af7db74b" />

