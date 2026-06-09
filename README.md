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

The `kdotP` simulation workflow combines electrostatic device modeling with multiband k·p quantum calculations. The calculation starts with a full three-dimensional device mesh that contains the dielectric layers, semiconductor regions, gates, ohmic contacts, and active quantum-well or quantum-dot region.

The nonlinear Poisson solver computes the electrostatic potential over the full device geometry. It uses the material properties assigned through the `Device` object and applies voltage boundary conditions to the named gates and ohmic contacts. The nonlinear Poisson equation has the general form:

[
\nabla \cdot \left[\epsilon(\mathbf{r}) \nabla \phi(\mathbf{r}) \right] = -\rho(\phi,\mathbf{r}),
]

where (\epsilon(\mathbf{r})) is the spatially dependent dielectric constant, (\phi(\mathbf{r})) is the electrostatic potential, and (\rho(\phi,\mathbf{r})) is the nonlinear charge density. The charge density depends on the local electrostatic potential and material parameters, which makes the problem nonlinear.

After the electrostatic potential is obtained, the active quantum-dot region is extracted from the full device mesh. This is done because the electrostatic problem must be solved over the entire device, while the quantum Hamiltonian only needs to be solved in the active region where the confined carrier states are located. The potential from the full mesh is then mapped onto the quantum-dot submesh and used as an input to the k·p Hamiltonian solver.

The k·p method is based on the interaction and coupling between semiconductor energy bands. The number of bands included in the Hamiltonian determines the level of physical accuracy and computational cost. A lower-dimensional model, such as the 4-band Hamiltonian, captures the heavy-hole and light-hole valence bands. A 6-band model also includes the split-off bands. An 8-band model includes both valence-band and conduction-band states and is therefore useful when conduction-valence coupling is important.

The basis functions used in the valence-band model are

[
\left| \frac{3}{2}, \frac{3}{2} \right\rangle =
\frac{1}{\sqrt{2}} \left( |X+iY\rangle \uparrow \right),
]

[
\left| \frac{3}{2}, -\frac{3}{2} \right\rangle =
\frac{1}{\sqrt{2}} \left( |X-iY\rangle \downarrow \right),
]

[
\left| \frac{3}{2}, \frac{1}{2} \right\rangle =
\frac{1}{\sqrt{6}} \left( |X+iY\rangle \downarrow \right)
-\sqrt{\frac{2}{3}} |Z\uparrow\rangle,
]

[
\left| \frac{3}{2}, -\frac{1}{2} \right\rangle =
-\frac{1}{\sqrt{6}} \left( |X-iY\rangle \uparrow \right)
-\sqrt{\frac{2}{3}} |Z\downarrow\rangle,
]

[
\left| \frac{1}{2}, \frac{1}{2} \right\rangle =
\frac{1}{\sqrt{3}} \left( |X+iY\rangle \downarrow \right)
+\frac{1}{\sqrt{3}} |Z\uparrow\rangle,
]

[
\left| \frac{1}{2}, -\frac{1}{2} \right\rangle =
-\frac{1}{\sqrt{3}} \left( |X-iY\rangle \uparrow \right)
+\frac{1}{\sqrt{3}} |Z\downarrow\rangle.
]

The 4-band model is written in the heavy-hole and light-hole basis,

[
\left{
\left| \frac{3}{2},\frac{3}{2} \right\rangle,
\left| \frac{3}{2},-\frac{3}{2} \right\rangle,
\left| \frac{3}{2},\frac{1}{2} \right\rangle,
\left| \frac{3}{2},-\frac{1}{2} \right\rangle
\right},
]

and is given by \cite{foreman1993}

[
H_{4\times4} =
\begin{pmatrix}
P + Q & 0 & -S_{-} & R \
0 & P + Q & -R^{\dagger} & -S_{+} \
-S_{-}^{\dagger} & -R & P - Q & C \
R^{\dagger} & -S_{+}^{\dagger} & C^{\dagger} & P - Q
\end{pmatrix}.
]

The matrix elements are

[
P =
E_V(\mathbf{r})+
\frac{\hbar^2}{2m_e}
\left(
\gamma_1 k_x^2+\gamma_1 k_y^2+\gamma_1 k_z^2
\right),
]

[
Q =
\frac{\hbar^2}{2m_e}
\left(
\gamma_2 k_x^2+\gamma_2 k_y^2-2\gamma_2 k_z^2
\right),
]

[
R =
-\frac{\hbar^2\sqrt{3}}{2m_e} k_- \bar{\gamma} k_-
+
\frac{\hbar^2\sqrt{3}}{2m_e} k_+ \mu k_+,
]

[
S_{\pm} =
\frac{\hbar^2\sqrt{3}}{m_e}
\left[
k_{\pm}(\sigma-\delta)k_z+k_z\pi k_{\pm}
\right],
]

[
C =
\frac{\hbar^2}{m_e}
\left[
k_z(\sigma-\delta-\pi)k_-
-------------------------

k_-(\sigma-\delta-\pi)k_z
\right].
]

The auxiliary parameters are

[
k_{\pm}=k_x\pm i k_y,
\qquad
k_{\parallel}^2=k_x^2+k_y^2,
]

[
\bar{\gamma}=\frac{1}{2}(\gamma_3+\gamma_2),
\qquad
\mu=\frac{1}{2}(\gamma_3-\gamma_2),
]

[
\sigma=\bar{\gamma}-\frac{1}{2}\delta,
\qquad
\pi=\mu+\frac{3}{2}\delta,
]

[
\delta=\frac{1}{9}
\left(
1+\gamma_1+\gamma_2-3\gamma_3
\right).
]

The 6-band Hamiltonian extends the 4-band model by including the split-off valence-band states. It is written as

[
H_{6\times6} =
\begin{pmatrix}
P + Q & 0 & -S_{-} & R & \frac{1}{\sqrt{2}}S_{-} & \sqrt{2}R \
0 & P + Q & -R^{\dagger} & -S_{+} & -\sqrt{2}R^{\dagger} & \frac{1}{\sqrt{2}}S_{+} \
-S_{-}^{\dagger} & -R & P - Q & C & \sqrt{2}Q & \sqrt{\frac{3}{2}}\Sigma_{-} \
R^{\dagger} & -S_{+}^{\dagger} & C^{\dagger} & P - Q & -\sqrt{\frac{3}{2}}\Sigma_{+} & \sqrt{2}Q \
\frac{1}{\sqrt{2}}S_{-}^{\dagger} & -\sqrt{2}R & \sqrt{2}Q & -\sqrt{\frac{3}{2}}\Sigma_{+}^{\dagger} & P + \Delta & -C \
\sqrt{2}R^{\dagger} & \frac{1}{\sqrt{2}}S_{+}^{\dagger} & \sqrt{\frac{3}{2}}\Sigma_{-} & \sqrt{2}Q & -C^{\dagger} & P + \Delta
\end{pmatrix}.
]

Here,

[
P =
E_v(z)+
\frac{1}{2}
\left(
\gamma_1 k_{\parallel}^2+k_z\gamma_1 k_z
\right),
]

[
Q =
\zeta(z)+
\frac{1}{2}
\left(
\gamma_2 k_{\parallel}^2-2k_z\gamma_2 k_z
\right),
]

[
R =
-\frac{\sqrt{3}}{2}\bar{\gamma} k_-^2
+
\frac{\sqrt{3}}{2}\mu k_+^2,
]

[
S_{\pm} =
\sqrt{3}k_{\pm}
\left[
(\sigma-\delta)k_z+k_z\pi
\right],
]

[
\Sigma_{\pm} =
\sqrt{3}k_{\pm}
\left{
\left[
\frac{1}{3}(\sigma-\delta)+\frac{2}{3}\pi
\right]k_z
+
k_z
\left[
\frac{2}{3}(\sigma-\delta)+\frac{1}{3}\pi
\right]
\right},
]

[
C =
k_-
\left[
k_z(\sigma-\delta-\pi)
----------------------

(\sigma-\delta-\pi)k_z
\right],
]

with

[
k_{\parallel}^2=k_x^2+k_y^2,
\qquad
k_+=k_x+i k_y,
\qquad
k_- = k_x-i k_y.
]

The 8-band model further extends the Hamiltonian by including conduction-band states and their coupling to the valence bands. This model is useful when conduction-valence mixing, narrow-gap effects, or strong confinement effects are important. The 8-band Hamiltonian can be written as

[
H_{8\times8} =
\begin{pmatrix}
E_v + E_0 + \dfrac{\hbar^2 k^2}{2m'} & 0 &
-\dfrac{1}{\sqrt{2}} P_0 k_+ &
\dfrac{\sqrt{3}}{\sqrt{2}} P_0 k_z &
0 & 0 &
-\dfrac{1}{\sqrt{2}} P_0 k_- &
\sqrt{\dfrac{3}{2}} P_0 k_z [8pt]

0 & E_v + E_0 + \dfrac{\hbar^2 k^2}{2m'} &
0 &
-\dfrac{1}{\sqrt{6}} P_0 k_+ &
-\dfrac{1}{\sqrt{6}} P_0 k_- &
0 &
\sqrt{\dfrac{3}{2}} P_0 k_z &
-\dfrac{1}{\sqrt{6}} P_0 k_+ [8pt]

-\dfrac{1}{\sqrt{2}} P_0 k_- & 0 &
P + Q & 0 & -S_- & R &
\dfrac{1}{\sqrt{2}} S_- & \sqrt{2}R [8pt]

\sqrt{\dfrac{3}{2}} P_0 k_z & -\dfrac{1}{\sqrt{6}} P_0 k_+ &
0 & P + Q & -R^\dagger & -S_+ &
-\sqrt{2}R^\dagger & \dfrac{1}{\sqrt{2}}S_+ [8pt]

0 & -\dfrac{1}{\sqrt{6}} P_0 k_- &
-S_-^\dagger & -R & P - Q & C &
\sqrt{2}Q & \sqrt{3}\Sigma_+ [8pt]

0 & 0 &
R^\dagger & -S_+^\dagger & C^\dagger & P - Q &
-\sqrt{3}\Sigma_+ & \sqrt{2}Q [8pt]

-\dfrac{1}{\sqrt{2}}P_0 k_- & \sqrt{\dfrac{3}{2}}P_0 k_z &
\dfrac{1}{\sqrt{2}}S_-^\dagger & -\sqrt{2}R & \sqrt{2}Q & -\sqrt{3}\Sigma_+^\dagger &
P+\Delta & -C [8pt]

\sqrt{\dfrac{3}{2}}P_0 k_z & -\dfrac{1}{\sqrt{6}}P_0 k_+ &
\sqrt{2}R^\dagger & \dfrac{1}{\sqrt{2}}S_+^\dagger & \sqrt{3}\Sigma_- & \sqrt{2}Q &
-C^\dagger & P+\Delta
\end{pmatrix}.
]

The Luttinger parameters used in the model can be corrected to account for coupling to the conduction band. The Kane energy is

[
E_P =
\frac{2m_0P_0^2}{\hbar^2}.
]

The modified Luttinger parameters are

[
\gamma_1 =
\gamma_1^L -
\frac{E_P}{3E_g+\Delta},
]

[
\gamma_2 =
\gamma_2^L -
\frac{1}{2}
\frac{E_P}{3E_g+\Delta},
]

[
\gamma_3 =
\gamma_3^L -
\frac{1}{2}
\frac{E_P}{3E_g+\Delta}.
]

In the finite-element implementation, the k·p Hamiltonian is assembled on the active quantum-dot submesh. The electrostatic confinement potential from the nonlinear Poisson solver enters through (E_v(\mathbf{r})) or the corresponding band-edge potential. Strain can be included through Bir-Pikus deformation-potential terms, and magnetic-field effects can be included through Zeeman and orbital coupling terms depending on the selected Hamiltonian model.

The final eigenvalue problem has the generalized form

[
A\psi = EB\psi,
]

where (A) is the assembled k·p Hamiltonian matrix, (B) is the finite-element mass matrix, (E) is the eigenenergy, and (\psi) is the multicomponent envelope-function eigenstate.

For multiband models, the physical probability density is obtained by summing over all band components:

[
\rho(\mathbf{r}) =
\sum_i |\psi_i(\mathbf{r})|^2.
]

For the 6-band model, this sum is taken over the six coupled envelope-function components. The resulting probability density can be saved as a `.vtu` file and visualized in ParaView or PyVista.


## 5. Results

The simulation produces electrostatic potential maps, quantum-dot submesh results, eigenenergies, and ground-state probability-density distributions.
ith quantum-mechanical k·p modeling. This allows the user to study how gate voltage, device geometry, strain, and magnetic field affect quantum-dot confinement and low-energy hole states. The bandstructure plot across the center of the quantum dot shows a band-engineered SiGe/GE heterostructure.

[NonLinear_poisson.tif](https://github.com/user-attachments/files/28761430/NonLinear_poisson.tif)

The wavefunction calculated using the 6-band model across the Y axis showing the quantum dot residing in the Ge structure. 

<img width="970" height="757" alt="ground_statewavefunction_Yaxis" src="https://github.com/user-attachments/assets/46af5f45-a790-4a22-a1b0-872bfd7359bb" />


The wavefunction was calculated using the 6-band across the Z axis.
<img width="1119" height="770" alt="ground_statewavefunction_Zaxis" src="https://github.com/user-attachments/assets/209b70d1-dc96-4b6e-a534-ff76af7db74b" />

