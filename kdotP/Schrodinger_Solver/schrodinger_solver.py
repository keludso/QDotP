"""
===============================================================================
Project: kdotP 
Description: Schrodinger Solver

Author: Kelvin Dsouza
Affiliation: North Carolina State University

License: MIT License

Copyright (c) 2026 Kelvin Dsouza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================================================
"""


import math
import numpy as np
import basix.ufl
import pyvista as pv
import ufl
from basix.ufl import element
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx import fem, plot
from dolfinx.mesh import exterior_facet_indices


class Schrodinger_Solver:
    """
    Schrodinger Solver

    This version is moved into the kdotP.luttinger_solver package and is
    structured to share the Device object from kdotP.base. The implementation
    stays close to your original code, with only light cleanup for reuse and a
    verbose flag for controlled prints.
    """

    def __init__(
        self,
        device,
        submesh,
        phi,
        log_writer=None,
        verbose=False,
    ):
        self.device = device
        self.domain = submesh
        self.verbose = verbose
        self.log_writer = log_writer if log_writer is not None else getattr(device, "log_writer", None)

        # choose between submesh or main mesh parameters
        self.gamma1 = device.gamma1_sub
        self.gamma2 = device.gamma2_sub
        self.gamma3 = device.gamma3_sub
        self.kappa = device.kappa_sub
        self.delta = device.delta_sub

        self.av = device.av_sub
        self.bv = device.bv_sub
        self.dv = device.dv_sub

        self.Me_x = device.Me_x_sub
        self.Me_y = device.Me_y_sub
        self.Me_z = device.Me_z_sub

        # Tags for strain 
        self.cell_tags = getattr(self.device, "submesh_cell_tags", None)
        self.region_map = getattr(self.device, "submesh_region_map", {})

        self.Bfield = False
        self.B = [0.0,0.0,0.0]

        self.strain = False

        # Keep original mesh rescaling logic.
        X = self.domain.geometry.x
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        extent = maxs - mins
        self._msg(f"[SS] BBox min: {mins}, max: {maxs}, extent: {extent}")

        if np.max(extent) > 10:
            self._msg("[SS] Mesh appears to already be in nm scale")
        else:
            self.domain.geometry.x[:] *= 1e9
            X = self.domain.geometry.x
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            extent = maxs - mins
            self._msg(f"[SS] Rescaled mesh to nm. BBox min: {mins}, max: {maxs}, extent: {extent}")

        self.dx = ufl.Measure("dx", domain=self.domain)

        self.phi = phi
        self.chi = device.chi_sub
        self.Eg = device.Eg_sub

        self._msg(f"[SS] phi min: {np.real(self.phi.x.array).min():.6e}")
        self._msg(f"[SS] phi max: {np.real(self.phi.x.array).max():.6e}")
        self._msg(f"[SS] chi min: {np.real(self.chi.x.array).min():.6e}")

        # hbar^2/(2m0) in the unit system used in the original code.
        self.h2m = 0.03809982116
        self._msg(f"[SS] h2m = {self.h2m}")

        V = phi.function_space
        Q = fem.functionspace(self.domain, ("DG", 0))

        self.Ec = fem.Function(Q)
        self.Ev = fem.Function(Q)

        phi_ref = fem.Constant(self.domain, PETSc.ScalarType(4.3963- 0j))

        Ec_expr =  -(phi -phi_ref + self.chi) # Use chii_P1, not chii
        Ev_expr = Ec_expr - self.Eg

        self.Ec.interpolate(fem.Expression(Ec_expr, Q.element.interpolation_points))
        self.Ev.interpolate(fem.Expression(Ev_expr, Q.element.interpolation_points))

        # Convert to VTK for visualization
        topology, cell_types, geometry = plot.vtk_mesh(self.domain, self.domain.topology.dim)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        grid.point_data["phi"] = np.real(phi.x.array)
        grid.cell_data["EC"] = np.real(self.Ec.x.array)
        grid.cell_data["Ev"] = np.real(self.Ev.x.array)
        grid.save("Poisson_subdomain.vtu")

        # Strain fields initialized to zero and later set per region.
        self.eps_xx = fem.Function(Q)
        self.eps_yy = fem.Function(Q)
        self.eps_zz = fem.Function(Q)
        self.eps_xy = fem.Function(Q)
        self.eps_xz = fem.Function(Q)
        self.eps_yz = fem.Function(Q)
        for f in [self.eps_xx, self.eps_yy, self.eps_zz, self.eps_xy, self.eps_xz, self.eps_yz]:
            f.x.array[:] = 0.0

        self._log("[SS], Schrodinger Solver Started")
        self._log(f"[SS], Initial Ev max: {np.real(self.Ev.x.array).max():.3e}")

    def _log(self, message):
        if self.log_writer is not None:
            self.log_writer.write(message)

    def _msg(self, message):
        if self.domain.comm.rank == 0 and self.verbose:
            print(message)
        self._log(message)

    def set_BField(self, B):
        # Tags for magnetic field 
        self.Bfield = True
        self.B = B if B is not None else [0.0, 0.0, 0.0]

    def set_region_strain(self, region_name, strain_matrix):
        strain = np.array(strain_matrix, dtype=float)
        if strain.shape != (3, 3):
            raise ValueError(f"strain_matrix must be 3x3, got {strain.shape}")

        if region_name not in self.region_map:
            raise ValueError(
                f"Region '{region_name}' not found. Available: {list(self.region_map.keys())}"
            )

        dofs = self.region_map[region_name]
        self.strain = True

        self.eps_xx.x.array[dofs] = strain[0, 0]
        self.eps_yy.x.array[dofs] = strain[1, 1]
        self.eps_zz.x.array[dofs] = strain[2, 2]
        self.eps_xy.x.array[dofs] = strain[0, 1]
        self.eps_xz.x.array[dofs] = strain[0, 2]
        self.eps_yz.x.array[dofs] = strain[1, 2]

        for f in [self.eps_xx, self.eps_yy, self.eps_zz, self.eps_xy, self.eps_xz, self.eps_yz]:
            f.x.scatter_forward()

        self._msg(f"[SS] Applied strain to region '{region_name}'")

    def A_vec(self):
        Bx, By, Bz = self.B
        x = ufl.SpatialCoordinate(self.domain)

        X = x[0]
        Y = x[1]
        Z = x[2]

        # Symmetric gauge: A = 1/2 B x r
        Ax = 0.5 * (By * Z - Bz * Y)
        Ay = 0.5 * (Bz * X - Bx * Z)
        Az = 0.5 * (Bx * Y - By * X)

        return [Ax, Ay, Az]


    def grad_B(self,u,i):
        if not self.Bfield or not getattr(self, "include_orbital", False):
            return PETSc.ScalarType(-1j) * ufl.grad(u)[i]
        
        A = self.A_vec()

        alpha = PETSc.ScalarType(self.orbital_sign * 1.519e-3)

        return (
            PETSc.ScalarType(-1j) * ufl.grad(u)[i]
            + alpha * A[i] * u
        )

    def sym_d(self, u, v, i, j):
        return 0.5 * (
            ufl.inner(self.grad_B(u,i), self.grad_B(v,j))
            + ufl.inner(self.grad_B(u,j), self.grad_B(v,i))
        )

    def inverse_mass_tensor(self):
        return ufl.as_matrix([
            [1.0 / self.Me_x, 0.0,               0.0],
            [0.0,                1.0 / self.Me_y, 0.0],
            [0.0,                0.0,               1.0 / self.Me_z],
        ])
    
    def assemble_components(self, u, v):
        Me_inv = self.inverse_mass_tensor()

        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)

        a_kinetic = (
            self.h2m
            * ufl.inner(ufl.dot(Me_inv, grad_u), grad_v)
            * self.dx
        )

        a_potential = ufl.inner(self.Ec * u, v) * self.dx

        return a_kinetic + a_potential

    def create_mass_matrix(self, u,v):
        return (
            ufl.inner(u, v) * self.dx
        )

    def solve_schrodinger(self, nev=10, target= -0.5, shift_offset=0.05):
        P1 = element(
            "Lagrange",
            self.domain.basix_cell(),
            degree=1
        )
        self.V = fem.functionspace(self.domain, P1)

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        a = self.assemble_components(u,v)

        self._msg(f"[SS] Number of integrals in form: {len(a.integrals())}")

        # Zero Dirichlet condition on the external boundary.
        self.domain.topology.create_connectivity(self.domain.topology.dim - 1, self.domain.topology.dim)
        bc_facets = exterior_facet_indices(self.domain.topology)
        bc_dofs = fem.locate_dofs_topological(self.V, self.domain.topology.dim - 1, bc_facets)
        u_bc = fem.Function(self.V)
        with u_bc.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        bc = fem.dirichletbc(u_bc, bc_dofs)

        A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
        A.assemble()

        b = self.create_mass_matrix(u,v)
        B = fem.petsc.assemble_matrix(fem.form(b), bcs=[bc])
        B.assemble()

        if self.verbose and self.domain.comm.rank == 0:
            info_A = A.getInfo()
            info_B = B.getInfo()
            print(f"[SS] Non-zeros in A: {info_A['nz_used']}")
            print(f"[SS] Non-zeros in B: {info_B['nz_used']}")
            print(f"[SS] ||A|| = {A.norm():.6e}")
            print(f"[SS] ||B|| = {B.norm():.6e}")

            AH = A.copy()
            AH.hermitianTranspose()
            D = A.copy()
            D.axpy(-1.0, AH)
            print(f"[SS] ||A-A^H||/||A|| = {D.norm() / A.norm():.6e}")

            BH = B.copy()
            BH.hermitianTranspose()
            M = B.copy()
            M.axpy(-1.0, BH)
            print(f"[SS] ||B-B^H||/||B|| = {M.norm() / B.norm():.6e}")

        eps = SLEPc.EPS().create(self.domain.comm)
        eps.setOperators(A, B)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(target)
        eps.setDimensions(nev=nev, ncv=max(80, 2 * nev + 20))
        eps.setTolerances(tol=1e-12, max_it=500)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(target + shift_offset)

        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("superlu_dist")

        st.setOptionsPrefix("st_")
        eps.setFromOptions()

        if self.verbose and self.domain.comm.rank == 0:
            eps.setMonitor(
                lambda eps_obj, it, nconv, eig, err: print(
                    f"Iter {it:3d}, converged {nconv}, eig={eig[0]:.4e}, err={err[0]:.2e}"
                )
            )

        eps.solve()

        num_converged = eps.getConverged()
        self._msg(f"[SS] Converged eigenvalues: {num_converged}")

        eigenvalues = []
        eigenvectors = []
        for i in range(min(num_converged, nev)):
            eigenval = eps.getEigenvalue(i)
            eigenvalues.append(eigenval.real)

            if self.verbose and self.domain.comm.rank == 0:
                rel = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
                absr = eps.computeError(i, SLEPc.EPS.ErrorType.ABSOLUTE)
                print(f"[SS] i={i} E={eigenval * 1:.6e} rel={rel:.3e} abs={absr:.3e}")

            eigenfunction = fem.Function(self.V)
            eps.getEigenpair(i, eigenfunction.x.petsc_vec)
            eigenfunction.x.scatter_forward()
            eigenvectors.append(eigenfunction)

        self._log("Eigenvalues for Schrodinger Equation (eV):")
        for i, E in enumerate(eigenvalues):
            self._msg(f"[SS] E_{i + 1} = {E:.8f}")
            self._log(f"  E_{i + 1} = {E:.8f}")

        eps.destroy()
        A.destroy()
        B.destroy()
        return eigenvalues, eigenvectors

    def save_probability_density_ground_state(
            self,
            eigenvectors,
            eigenvalues,
            output_file="ground_state_probability.vtu",
        ):
            """Save |psi|² for the lowest-energy scalar electron state."""

            if not eigenvectors:
                raise ValueError("No eigenvectors were provided.")

            psi = eigenvectors[0]
            energy = eigenvalues[0]

            topology, cell_types, geometry = plot.vtk_mesh(self.V)
            grid = pv.UnstructuredGrid(topology, cell_types, geometry)

            psi_values = psi.x.array
            probability = np.abs(psi_values) ** 2

            grid.point_data["psi_real"] = np.real(psi_values)
            grid.point_data["psi_imag"] = np.imag(psi_values)
            grid.point_data["probability_density"] = probability

            grid.save(output_file)

            self._msg(
                f"[SS] Ground-state energy = {np.real(energy):.8f} eV"
            )
            self._msg(
                f"[SS] Maximum |psi|^2 = {probability.max():.6e}"
            )
            self._msg(f"[SS] Saved probability density to {output_file}")
