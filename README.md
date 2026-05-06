# kdotP
# kdotP: FEM k·p / Poisson Solver for Quantum Dot Heterostructures

`kdotP` is a Python package for solving electrostatics and multiband k·p Hamiltonians in semiconductor quantum-dot heterostructures.  
The code is designed around a `Device` object that reads a Gmsh mesh, assigns materials to physical regions, solves the nonlinear Poisson equation, extracts a submesh for the active quantum-dot region, and solves 4-band, 6-band, or 8-band Luttinger/Kane Hamiltonians.

This README describes the main workflow, the purpose of each function/class used in the example script, and common help/troubleshooting notes.

---

## 1. Main Features

- Read `.msh2` device meshes generated from Gmsh.
- Assign semiconductor and dielectric materials by physical region name.
- Solve the nonlinear Poisson equation with gate voltage boundary conditions.
- Extract active quantum-dot submeshes from the full electrostatic device.
- Map electrostatic potential from the full mesh to a submesh.
- Solve 4-band, 6-band, or 8-band k·p Hamiltonians.
- Include strain tensors region-by-region.
- Include external magnetic fields.
- Save eigenstate probability densities to `.vtu` files for visualization in ParaView or PyVista.
- Save log output to a text file using `savedat`.

---

## 2. Requirements

The code assumes a scientific Python environment with FEM and eigenvalue solver support.

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

If using MPI:

```bash
mpirun -np 4 python test_6band.py
```

For serial testing:

```bash
python test_6band.py
```

---

## 3. Package Structure

A typical project layout is:

```text
kdotP/
│
├── base/
│   ├── __init__.py
│   ├── device.py
│   └── materials.py
│
├── poisson_solver.py
├── luttinger_solver.py
├── savedat.py
│
examples/
└── test_6band.py
```

The example imports:

```python
from kdotP.base import Device
from kdotP.base.materials import Ge, SiGe80, Al2O3
from kdotP.poisson_solver import NonlinearPoissonSolver
from kdotP.luttinger_solver import LK4BandHamiltonian
from kdotP.luttinger_solver import LK_hamiltonian_8band
from kdotP.luttinger_solver import LK_hamiltonian_6band
from savedat import savedat
```

---

## 4. Example: Testing the 6-Band Model

The following script solves the nonlinear Poisson problem, extracts the dot submesh, solves the 6-band k·p Hamiltonian, and then repeats the calculation for several strain values.

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

# Solve nonlinear Poisson equation 
#

psolver = NonlinearPoissonSolver(
    device,
    0.0,
    verbose=True,
    log_writer=logger,
)

Gate_voltages = {
    "top_gate_1": 0.0,
    "top_gate_2": 0.0,
    "bottom_gate": 0.0,
    "plunger_gate": 0.0,
}

Ohmic_gate = ["back_gate"]

phi = psolver.nonlin_solve(
    Gate_voltages,
    Ohmic_gate,
    save_file=True,
    output_file="poisson.vtu",
)

# Create a submesh for the dot region
dot_submesh = ["barrier_dot", "two_deg_dot", "relaxed_dot"]

dot_materials = {
    "barrier_dot": SiGe80,
    "two_deg_dot": Ge,
    "relaxed_dot": SiGe80,
}

submesh, vertex_map = device.create_submesh(dot_submesh, dot_materials)

# Interpolate/map Poisson potential from full mesh to dot submesh
phi_sub = device.submesh_parameters(phi, submesh, vertex_map)

# Create 6-band Luttinger/Kane Hamiltonian solver
lk_solver = LK_hamiltonian_6band(
    device,
    submesh,
    phi_sub,
    log_writer=logger,
    verbose=True,
)

# Solve the 6-band Hamiltonian
eigenvalues, eigenvectors = lk_solver.solve_LK()

# Save the ground-state probability density
lk_solver.save_probability_density_ground_state(
    eigenvectors,
    eigenvalues,
    output_file="ground_state_6band.vtu",
)

# Sweep strain values
R = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])

for r in R:

    # Out-of-plane strain assuming elastic relaxation
    ezz = -2 * r * 0.37

    lk_solver.set_region_strain(
        "two_deg_dot",
        [
            [r,   0.0, 0.0],
            [0.0, r,   0.0],
            [0.0, 0.0, ezz],
        ],
    )

    # Magnetic field in Tesla
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

## 5. Workflow Description

The calculation follows this sequence:

```text
Gmsh mesh
   ↓
Device(filename)
   ↓
Assign materials
   ↓
Solve nonlinear Poisson equation
   ↓
Extract quantum-dot submesh
   ↓
Map electrostatic potential to submesh
   ↓
Build 6-band k·p Hamiltonian
   ↓
Apply strain and magnetic field
   ↓
Solve eigenvalue problem
   ↓
Save probability density
```

---

## 6. Class and Function Descriptions

### `Device`

```python
device = Device(filename, verbose=True)
```

The `Device` class is the central object that stores the mesh, region labels, facet labels, material assignments, and helper methods for submesh creation.

#### Inputs

| Argument | Type | Description |
|---|---|---|
| `filename` | `str` | Name of the `.msh2` mesh file. |
| `verbose` | `bool` | If `True`, prints mesh and region information. |

#### Main responsibilities

- Reads the Gmsh mesh.
- Stores the computational domain.
- Stores cell tags and facet tags.
- Maps physical region names to mesh markers.
- Assigns material parameters.
- Creates submeshes for selected regions.
- Maps fields from the full mesh to submeshes.
- Stores a logger if provided.

---

### `device.set_logger`

```python
device.set_logger(logger)
```

Attaches a log writer to the `Device` object.

#### Inputs

| Argument | Type | Description |
|---|---|---|
| `logger` | object | A log-writer object, such as one created by `savedat`. |

#### Purpose

Use this function to save diagnostic information, solver messages, material assignments, and eigenvalue results to a text file.

---

### `device.set_materials`

```python
device.set_materials(materials_map)
```

Assigns materials to named physical regions in the mesh.

#### Inputs

| Argument | Type | Description |
|---|---|---|
| `materials_map` | `dict` | Dictionary mapping region names to material objects. |

#### Example

```python
materials_map = {
    "two_deg_dot": Ge,
    "barrier_dot": SiGe80,
    "cap": Al2O3,
}
device.set_materials(materials_map)
```

#### Notes

The keys must match the physical names defined in the Gmsh mesh.  
If a region name is misspelled or missing from the mesh, material assignment may fail or produce incorrect device parameters.

---

### `NonlinearPoissonSolver`

```python
class NonlinearPoissonSolver:
    """
    Nonlinear Poisson solver that works directly with a Device object.

    The solver reuses the material functions stored in the Device class and
    takes the mesh from device.domain, so it can be shared cleanly across the
    package.
    """

    def __init__(self, device, fermi, verbose=False, log_writer=None):
        ...
```

Creates a nonlinear Poisson solver for the full device mesh.

#### Constructor

```python
psolver = NonlinearPoissonSolver(
    device,
    fermi,
    verbose=False,
    log_writer=None,
)
```

#### Constructor arguments

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `device` | `Device` | required | Main device object. The solver uses `device.domain` as the mesh and reuses the material fields, region labels, facet labels, and material functions already stored in the `Device` object. |
| `fermi` | `float` | required | Fermi-level reference used in the nonlinear charge-density model. In the example, this is set to `0.0`. The exact physical meaning depends on how the carrier density is implemented in `poisson_solver.py`, but it is usually the electrochemical potential or energy reference for evaluating carrier occupation. |
| `verbose` | `bool` | `False` | If `True`, prints detailed solver information, such as mesh information, nonlinear iteration progress, residuals, and diagnostic messages. |
| `log_writer` | object or `None` | `None` | Optional logger object, such as one created by `savedat`. If provided, solver messages and diagnostic outputs are written to the log file. |

#### Purpose

The `NonlinearPoissonSolver` computes the electrostatic potential `phi` over the full device geometry. It uses the material parameters assigned to the `Device` object and applies voltage boundary conditions to named gates and ohmic contacts.

The nonlinear Poisson equation usually has the form:

```text
∇ · (ε(r) ∇φ(r)) = -ρ(φ, r)
```

where:

- `ε(r)` is the position-dependent dielectric constant.
- `φ(r)` is the electrostatic potential.
- `ρ(φ, r)` is the nonlinear charge density.

---

### `psolver.nonlin_solve`

```python
def nonlin_solve(
    self,
    gate_voltages,
    ohmic_gate_name,
    save_file=True,
    output_file="poisson.vtu",
):
    ...
```

Solves the nonlinear Poisson equation using the gate voltages and ohmic boundary names provided by the user.

#### Usage

```python
phi = psolver.nonlin_solve(
    gate_voltages=Gate_voltages,
    ohmic_gate_name=Ohmic_gate,
    save_file=True,
    output_file="poisson.vtu",
)
```

or, using positional arguments:

```python
phi = psolver.nonlin_solve(Gate_voltages, Ohmic_gate)
```

#### Arguments

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `gate_voltages` | `dict[str, float]` | required | Dictionary mapping gate names to applied voltage values. The keys must match physical boundary names in the Gmsh mesh. Example: `{"top_gate_1": 0.0, "plunger_gate": -0.2}`. |
| `ohmic_gate_name` | `list[str]` or `str` | required | Name or list of names of ohmic contacts/reference gates. These boundaries are usually treated as contacts or reference electrostatic boundaries. In the example, `Ohmic_gate = ["back_gate"]`. |
| `save_file` | `bool` | `True` | If `True`, saves the computed electrostatic potential to a `.vtu` file. If `False`, the potential is returned but not written to disk. |
| `output_file` | `str` | `"poisson.vtu"` | Name of the output file used when `save_file=True`. The file is usually written in VTK/VTU format for visualization in ParaView or PyVista. |

#### Returns

| Output | Type | Description |
|---|---|---|
| `phi` | FEM function | Electrostatic potential solved on the full device mesh. This field is later mapped to the quantum-dot submesh using `device.submesh_parameters`. |

#### Example

```python
Gate_voltages = {
    "top_gate_1": 0.0,
    "top_gate_2": 0.0,
    "bottom_gate": 0.0,
    "plunger_gate": 0.0,
}

Ohmic_gate = ["back_gate"]

phi = psolver.nonlin_solve(
    Gate_voltages,
    Ohmic_gate,
    save_file=True,
    output_file="poisson.vtu",
)
```

#### Notes

- Every key in `gate_voltages` must correspond to a physical boundary name in the mesh.
- Every name in `ohmic_gate_name` must also exist as a boundary/facet label in the mesh.
- The returned `phi` is defined on the full device mesh.
- The Luttinger/Kane solver normally uses a restricted potential `phi_sub`, obtained by mapping `phi` to the active quantum-dot submesh.
- If the file output is not needed, use `save_file=False` to avoid writing `poisson.vtu`.

---

### `device.create_submesh`

```python
submesh, vertex_map = device.create_submesh(dot_submesh, dot_materials)
```

Creates a smaller mesh containing only selected physical regions.

#### Inputs

| Argument | Type | Description |
|---|---|---|
| `dot_submesh` | `list[str]` | Region names to include in the submesh. |
| `dot_materials` | `dict` | Material map for the selected submesh regions. |

#### Returns

| Output | Type | Description |
|---|---|---|
| `submesh` | FEM mesh | Mesh containing only the selected dot regions. |
| `vertex_map` | array-like | Map between submesh vertices and original full-mesh vertices. |

#### Example

```python
dot_submesh = ["barrier_dot", "two_deg_dot", "relaxed_dot"]

dot_materials = {
    "barrier_dot": SiGe80,
    "two_deg_dot": Ge,
    "relaxed_dot": SiGe80,
}

submesh, vertex_map = device.create_submesh(dot_submesh, dot_materials)
```

#### Purpose

The electrostatic problem is solved on the full device, but the k·p Hamiltonian is usually solved only in the active quantum-dot region to reduce computational cost.

---

### `device.submesh_parameters`

```python
phi_sub = device.submesh_parameters(phi, submesh, vertex_map)
```

Maps a field from the full device mesh to the submesh.

#### Inputs

| Argument | Type | Description |
|---|---|---|
| `phi` | FEM function | Field defined on the full mesh. |
| `submesh` | FEM mesh | Target submesh. |
| `vertex_map` | array-like | Mapping from submesh vertices to full-mesh vertices. |

#### Returns

| Output | Type | Description |
|---|---|---|
| `phi_sub` | FEM function | Electrostatic potential mapped onto the submesh. |

#### Purpose

The Luttinger/Kane Hamiltonian needs the electrostatic potential in the active quantum-dot region. This function transfers the Poisson solution from the full mesh to the submesh.

---

### `LK_hamiltonian_6band`

```python
def __init__(
    self,
    device,
    submesh,
    phi,
    log_writer=None,
    verbose=False,
):
    ...
```

Creates the 6-band k·p Hamiltonian solver on the active quantum-dot submesh.

#### Constructor

```python
lk_solver = LK_hamiltonian_6band(
    device,
    submesh,
    phi,
    log_writer=None,
    verbose=False,
)
```

In the example:

```python
lk_solver = LK_hamiltonian_6band(
    device,
    submesh,
    phi_sub,
    log_writer=logger,
    verbose=True,
)
```

#### Constructor arguments

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `device` | `Device` | required | Full `Device` object. The solver uses it to access material parameters, region labels, material maps, logging utilities, and device-level metadata. Even though the Hamiltonian is solved on a submesh, the full device object is still needed for region/material bookkeeping. |
| `submesh` | FEM mesh | required | Active-region mesh on which the 6-band Hamiltonian is assembled. This is usually produced by `device.create_submesh(...)` and contains only the quantum-dot and nearby barrier regions. |
| `phi` | FEM function | required | Electrostatic potential on the same mesh used by the Hamiltonian solver. In the example, this is `phi_sub`, obtained from `device.submesh_parameters(phi, submesh, vertex_map)`. |
| `log_writer` | object or `None` | `None` | Optional logger, such as `savedat`. If provided, Hamiltonian assembly information, eigenvalues, solver diagnostics, strain settings, and magnetic-field settings can be written to the log file. |
| `verbose` | `bool` | `False` | If `True`, prints detailed information about Hamiltonian setup, matrix assembly, eigenvalue solver settings, and diagnostics. |

#### Purpose

The 6-band Hamiltonian solver builds the finite-element form of the Luttinger/Kane Hamiltonian for valence-band states. It is typically used for hole states in Ge/SiGe quantum wells or quantum dots.

The 6-band model typically includes:

- Heavy-hole states.
- Light-hole states.
- Split-off hole states.
- Electrostatic confinement through `phi`.
- Strain coupling through Bir-Pikus terms.
- Magnetic-field coupling through Zeeman terms and any additional field terms implemented in the code.

#### Important consistency requirement

`submesh` and `phi` must be defined on the same mesh. Do not pass the full-device Poisson solution directly into `LK_hamiltonian_6band` unless the Hamiltonian is also being solved on the full device mesh. For the standard workflow, use:

```python
phi_sub = device.submesh_parameters(phi, submesh, vertex_map)
```

then pass `phi_sub` into the Hamiltonian solver.

---

### `lk_solver.solve_LK`

```python
eigenvalues, eigenvectors = lk_solver.solve_LK()
```

Assembles and solves the generalized eigenvalue problem for the k·p Hamiltonian.

#### Returns

| Output | Type | Description |
|---|---|---|
| `eigenvalues` | array-like | Computed energy eigenvalues. |
| `eigenvectors` | array-like | Corresponding multicomponent eigenvectors. |

#### Purpose

Solves:

```text
A ψ = E B ψ
```

where:

- `A` is the assembled k·p Hamiltonian matrix.
- `B` is the FEM mass matrix.
- `E` is the eigenenergy.
- `ψ` is the multiband envelope-function eigenstate.

#### Notes

The eigenvectors are multicomponent wavefunctions.  
For the 6-band model, each eigenvector contains six coupled envelope-function components.

---

### `lk_solver.save_probability_density_ground_state`

```python
def save_probability_density_ground_state(
    self,
    eigenvectors,
    eigenvalues,
    output_file="ground_state_probability.vtu",
):
    ...
```

Saves the probability density of the ground-state eigenvector to a `.vtu` file.

#### Usage

```python
lk_solver.save_probability_density_ground_state(
    eigenvectors,
    eigenvalues,
    output_file="ground_state_6band.vtu",
)
```

#### Arguments

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `eigenvectors` | array-like / PETSc vector collection | required | Eigenvectors returned by `lk_solver.solve_LK()`. For the 6-band model, each eigenvector contains six coupled envelope-function components. |
| `eigenvalues` | array-like | required | Eigenvalues returned by `lk_solver.solve_LK()`. These are used to identify/order the ground state and may also be written as metadata or logged depending on implementation. |
| `output_file` | `str` | `"ground_state_probability.vtu"` | Name of the output VTK file. Use a unique filename inside parameter sweeps to avoid overwriting previous results. |

#### What it saves

The function computes the total multiband probability density of the lowest-energy state:

```text
ρ(r) = Σ_i |ψ_i(r)|²
```

where `i` runs over all band components. For a 6-band Hamiltonian, this sum is over six envelope-function components.

#### Returns

This function is usually used for file output and may not return a value. The main output is the `.vtu` file.

#### Example

```python
lk_solver.save_probability_density_ground_state(
    eigenvectors,
    eigenvalues,
    output_file="ground_state_6band.vtu",
)
```

For a strain sweep, use unique filenames:

```python
output_file = f"ground_state_6band_strain_{r:+.4f}.vtu"

lk_solver.save_probability_density_ground_state(
    eigenvectors,
    eigenvalues,
    output_file=output_file,
)
```

#### Notes

- The output can be opened in ParaView or PyVista.
- The probability density should be checked for proper normalization.
- For multiband models, the physically meaningful density is the sum of the squared magnitudes of all components.
- If all sweep iterations use the same `output_file`, only the last result will be kept.

---

### `lk_solver.set_region_strain`

```python
lk_solver.set_region_strain(
    region_name,
    strain_tensor,
)
```

Sets the strain tensor for a specific material region.

#### Inputs

| Argument | Type | Description |
|---|---|---|
| `region_name` | `str` | Name of the region where strain is applied. |
| `strain_tensor` | `3 x 3 list` or `numpy.ndarray` | Strain tensor. |

#### Example

```python
r = 0.005
ezz = -2 * r * 0.37

lk_solver.set_region_strain(
    "two_deg_dot",
    [
        [r,   0.0, 0.0],
        [0.0, r,   0.0],
        [0.0, 0.0, ezz],
    ],
)
```

#### Notes

The strain tensor is:

```text
ε = [[εxx, εxy, εxz],
     [εyx, εyy, εyz],
     [εzx, εzy, εzz]]
```

For biaxial in-plane strain:

```text
εxx = εyy = r
εzz = -2 C12/C11 r
```

In the example, `0.37` is used as an approximate elastic ratio.

---

### `lk_solver.set_BField`

```python
lk_solver.set_BField(B)
```

Sets the external magnetic field.

#### Inputs

| Argument | Type | Description |
|---|---|---|
| `B` | `list[float]` or `numpy.ndarray` | Magnetic field vector `[Bx, By, Bz]` in Tesla. |

#### Example

```python
B = [0.0, 0.0, 0.2]
lk_solver.set_BField(B)
```

#### Purpose

Adds magnetic-field terms to the Hamiltonian, such as Zeeman splitting and other field-dependent contributions implemented in the solver.

---

### `LK4BandHamiltonian`

```python
from kdotP.luttinger_solver import LK4BandHamiltonian
```

The 4-band Hamiltonian solver is intended for valence-band calculations involving only heavy-hole and light-hole states.

#### Typical model content

- Heavy-hole bands.
- Light-hole bands.
- Luttinger parameters.
- Electrostatic confinement.
- Optional strain and magnetic-field terms, depending on implementation.

#### When to use

Use the 4-band model when split-off bands are far away in energy and do not significantly affect the low-energy hole states.

---

### `LK_hamiltonian_8band`

```python
from kdotP.luttinger_solver import LK_hamiltonian_8band
```

The 8-band Hamiltonian solver is intended for coupled conduction-band and valence-band calculations.

#### Typical model content

- Conduction-band states.
- Heavy-hole states.
- Light-hole states.
- Split-off states.
- Kane coupling between conduction and valence bands.
- Strain and magnetic-field effects, depending on implementation.

#### When to use

Use the 8-band model when conduction-valence coupling, strong confinement, or narrow-gap effects are important.

---

### Material objects: `Ge`, `SiGe80`, `Al2O3`

```python
from kdotP.base.materials import Ge, SiGe80, Al2O3
```

These objects define material parameters used by the Poisson and k·p solvers.

Typical parameters may include:

- Band gap.
- Electron affinity.
- Relative dielectric constant.
- Luttinger parameters.
- Deformation potentials.
- Effective masses.
- Spin-orbit splitting.
- Elastic constants.
- Kane parameters.

#### Example

```python
materials_map = {
    "two_deg_dot": Ge,
    "barrier_dot": SiGe80,
    "cap": Al2O3,
}
```

---

### `savedat`

```python
logger = savedat("output_log.dat")
```

Creates a simple log writer.

#### Inputs

| Argument | Type | Description |
|---|---|---|
| `filename` | `str` | Name of the output log file. |

#### Purpose

Stores text output such as:

- Mesh information.
- Material assignments.
- Solver convergence.
- Matrix diagnostics.
- Eigenvalues.
- Strain and magnetic-field sweep results.

---

## 7. Input Mesh Requirements

The `.msh2` file must contain named physical regions and physical boundaries.

Example physical volume or surface names:

```text
cap
barrier_dot
barrier
two_deg_dot
two_deg
relaxed_barrier
relaxed_dot
substrate
top_gate_1
top_gate_2
bottom_gate
plunger_gate
back_gate
```

The names in the Python dictionaries must match the names in the mesh exactly.

For example, this will only work if `"two_deg_dot"` exists in the Gmsh file:

```python
materials_map = {
    "two_deg_dot": Ge,
}
```

---

## 8. Output Files

The script produces files such as:

```text
test_6band_strain_magnetic_field.dat
ground_state_6band.vtu
ground_state_6band_strain_-0.0100.vtu
ground_state_6band_strain_-0.0050.vtu
ground_state_6band_strain_+0.0000.vtu
ground_state_6band_strain_+0.0050.vtu
ground_state_6band_strain_+0.0100.vtu
```

### `.dat` file

Contains log output from the device, Poisson solver, and k·p solver.

### `.vtu` files

Contain ground-state probability density data for visualization.

To view in ParaView:

```bash
paraview ground_state_6band.vtu
```

---

## 9. Strain Sweep

The example applies a biaxial strain sweep:

```python
R = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
```

For each strain value:

```python
ezz = -2 * r * 0.37
```

Then the tensor is applied to the Ge quantum well region:

```python
lk_solver.set_region_strain(
    "two_deg_dot",
    [
        [r,   0.0, 0.0],
        [0.0, r,   0.0],
        [0.0, 0.0, ezz],
    ],
)
```

This allows the user to study how heavy-hole/light-hole mixing, confinement energy, and ground-state probability density change with strain.

---

## 10. Magnetic-Field Sweep

A magnetic field can be applied using:

```python
B = [0.0, 0.0, 0.2]
lk_solver.set_BField(B)
```

where the units are Tesla.

To sweep magnetic field:

```python
Bz_values = np.linspace(0.0, 1.0, 11)

for Bz in Bz_values:
    lk_solver.set_BField([0.0, 0.0, Bz])
    eigenvalues, eigenvectors = lk_solver.solve_LK()
```

---

## 11. Help and Troubleshooting

### Problem: Mesh region name not found

Check that the physical names in the mesh match the dictionary keys.

```python
materials_map = {
    "two_deg_dot": Ge,
}
```

The name `"two_deg_dot"` must exist in the Gmsh physical groups.

---

### Problem: Gate boundary condition is not applied

Check that the gate name exists as a physical surface/facet in the mesh.

```python
Gate_voltages = {
    "top_gate_1": 0.0,
}
```

The name `"top_gate_1"` must match the mesh boundary label.

---

### Problem: Poisson solver does not converge

Try:

- Reducing gate voltages.
- Checking material parameters.
- Checking boundary conditions.
- Making sure ohmic contacts are assigned correctly.
- Using a better initial guess for `phi`.
- Checking whether the carrier-density model is numerically stable.

---

### Problem: Eigenvalue solver does not converge

Try:

- Reducing the number of requested eigenvalues.
- Increasing the Krylov subspace size.
- Checking that the Hamiltonian matrix is Hermitian.
- Checking that the mass matrix is positive definite.
- Refining or simplifying the mesh.
- Checking for invalid material parameters.
- Checking strain values for unphysical inputs.

---

### Problem: Probability density output looks wrong

Check:

- Whether the eigenvector is normalized.
- Whether the correct submesh is used.
- Whether the `.vtu` file is opened on the correct mesh.
- Whether the probability density sums over all band components.
- Whether the eigenvalue ordering is correct.

---

### Problem: Strain has no effect

Check:

- Whether `set_region_strain` is called before `solve_LK`.
- Whether the region name is correct.
- Whether the 6-band Hamiltonian includes the Bir-Pikus strain terms.
- Whether the strain tensor is assigned to the active material region.

---

### Problem: Magnetic field has no effect

Check:

- Whether `set_BField` is called before `solve_LK`.
- Whether the Hamiltonian includes Zeeman and/or orbital magnetic-field terms.
- Whether the magnetic field is large enough to produce visible splitting.
- Whether the eigenvalues being compared correspond to the same states.

---

## 12. Recommended Script Improvements

For strain sweeps, use unique filenames:

```python
output_file = f"ground_state_6band_strain_{r:+.4f}.vtu"
```

Instead of overwriting the same file:

```python
output_file = "ground_state_6band_mmstrain.vtu"
```

Also save the eigenvalues in the log file or in a separate `.csv` file:

```python
np.savetxt("eigenvalues_strain_sweep.txt", eigenvalue_table)
```

---

## 13. Minimal Help Example

```python
help(Device)
help(NonlinearPoissonSolver)
help(LK_hamiltonian_6band)
```

To inspect available methods:

```python
dir(device)
dir(psolver)
dir(lk_solver)
```

To check material parameters:

```python
print(Ge)
print(SiGe80)
print(Al2O3)
```

---

## 14. Quick Function Argument Summary

| Function / Class | Argument | Required? | Default | Description |
|---|---|---:|---:|---|
| `NonlinearPoissonSolver.__init__` | `device` | yes | — | Full `Device` object containing mesh, materials, region tags, and boundary tags. |
| `NonlinearPoissonSolver.__init__` | `fermi` | yes | — | Fermi-level/reference energy used in the nonlinear carrier-density model. |
| `NonlinearPoissonSolver.__init__` | `verbose` | no | `False` | Prints detailed diagnostic output when enabled. |
| `NonlinearPoissonSolver.__init__` | `log_writer` | no | `None` | Optional logger for writing solver output to a file. |
| `nonlin_solve` | `gate_voltages` | yes | — | Dictionary mapping gate boundary names to voltages. |
| `nonlin_solve` | `ohmic_gate_name` | yes | — | Ohmic/reference contact name or list of names. |
| `nonlin_solve` | `save_file` | no | `True` | Controls whether the Poisson result is saved to disk. |
| `nonlin_solve` | `output_file` | no | `"poisson.vtu"` | File name for the saved electrostatic potential. |
| `LK_hamiltonian_6band.__init__` | `device` | yes | — | Full `Device` object used for material and region information. |
| `LK_hamiltonian_6band.__init__` | `submesh` | yes | — | Active-region mesh for the k·p Hamiltonian. |
| `LK_hamiltonian_6band.__init__` | `phi` | yes | — | Electrostatic potential defined on `submesh`. |
| `LK_hamiltonian_6band.__init__` | `log_writer` | no | `None` | Optional logger for Hamiltonian/eigenvalue information. |
| `LK_hamiltonian_6band.__init__` | `verbose` | no | `False` | Prints Hamiltonian setup and eigensolver diagnostics. |
| `save_probability_density_ground_state` | `eigenvectors` | yes | — | Eigenvectors returned by `solve_LK()`. |
| `save_probability_density_ground_state` | `eigenvalues` | yes | — | Eigenvalues returned by `solve_LK()`. |
| `save_probability_density_ground_state` | `output_file` | no | `"ground_state_probability.vtu"` | Output file for the ground-state probability density. |

---

## 15. Full Calculation Checklist

Before running the solver, verify:

- The mesh file exists.
- Physical region names are correct.
- Physical gate names are correct.
- Every semiconductor/dielectric region has a material assigned.
- Gate voltages are physically reasonable.
- Ohmic contacts are defined.
- The submesh regions are included in the full mesh.
- The Poisson solution is successfully computed.
- The electrostatic potential is mapped to the submesh.
- Strain tensors are assigned to the intended regions.
- Magnetic field is set before solving the Hamiltonian.
- Output filenames are unique for parameter sweeps.

---

## 16. Citation / Acknowledgment Placeholder

If this code is used in a publication or report, cite the relevant k·p, Luttinger-Kohn, Bir-Pikus, finite-element, PETSc/SLEPc, and FEniCSx references.

Suggested placeholder:

```text
This calculation used a custom finite-element k·p solver based on the Luttinger-Kohn/Bir-Pikus Hamiltonian and nonlinear Poisson electrostatics.
```

---

## 17. Author Notes

This code is intended for research-level simulations of semiconductor quantum-dot heterostructures, especially Ge/SiGe and related strained quantum-well systems.  
The current example focuses on testing the 6-band model with strain and magnetic field.
