"""
===============================================================================
Project: kdotP 
Description: 4-band Luttinger-Kohn Hamiltonian solver using FEM

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


class LK4BandHamiltonian:
    """
    4-band Luttinger-Kohn Hamiltonian solver.

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
        self._msg(f"[LK4] BBox min: {mins}, max: {maxs}, extent: {extent}")

        if np.max(extent) > 10:
            self._msg("[LK4] Mesh appears to already be in nm scale")
        else:
            self.domain.geometry.x[:] *= 1e9
            X = self.domain.geometry.x
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            extent = maxs - mins
            self._msg(f"[LK4] Rescaled mesh to nm. BBox min: {mins}, max: {maxs}, extent: {extent}")

        self.dx = ufl.Measure("dx", domain=self.domain)

        self.phi = phi
        self.chi = device.chi_sub
        self.Eg = device.Eg_sub

        self._msg(f"[LK4] phi min: {np.real(self.phi.x.array).min():.6e}")
        self._msg(f"[LK4] phi max: {np.real(self.phi.x.array).max():.6e}")
        self._msg(f"[LK4] chi min: {np.real(self.chi.x.array).min():.6e}")

        # hbar^2/(2m0) in the unit system used in the original code.
        self.h2m = 0.03809982116
        self._msg(f"[LK4] h2m = {self.h2m}")

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

        self._log("[LK4], Schrodinger Solver Started")
        self._log(f"[LK4], Initial Ev max: {np.real(self.Ev.x.array).max():.3e}")

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

        self._msg(f"[LK4] Applied strain to region '{region_name}'")




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


    def P_operator(self, u, v):
        return self.h2m * self.gamma1 * (
            ufl.inner(self.grad_B(u,0), self.grad_B(v,0))
            + ufl.inner(self.grad_B(u,1), self.grad_B(v,1))
            + ufl.inner(self.grad_B(u,2), self.grad_B(v,2))
        )

    def Q_operator(self, u, v):
        return self.h2m * self.gamma2 * (
            ufl.inner(self.grad_B(u,0), self.grad_B(v,0))
            + ufl.inner(self.grad_B(u,1), self.grad_B(v,1))
            - 2 * ufl.inner(self.grad_B(u,2), self.grad_B(v,2))
        )

    def R_operator(self, u, v):
        # R = -sqrt(3) gamma2 (kx^2 - ky^2) + 2 i sqrt(3) gamma3 {kx,ky}
        return (
            math.sqrt(3.0) * self.h2m * self.gamma2
            * (
                -ufl.inner(self.grad_B(u,0), self.grad_B(v,0))
                + ufl.inner(self.grad_B(u,1), self.grad_B(v,1))
            )
            + 2.0j * math.sqrt(3.0) * self.h2m * self.gamma3 * self.sym_d(u, v, 0, 1)
        )


    def R_operator_conj(self, u, v):
        # Conjugate partner of R
        return (
            math.sqrt(3.0) * self.h2m * self.gamma2
            * (
                -ufl.inner(self.grad_B(u,0), self.grad_B(v,0))
                + ufl.inner(self.grad_B(u,1), self.grad_B(v,1))
            )
            - 2.0j * math.sqrt(3.0) * self.h2m * self.gamma3 * self.sym_d(u, v, 0, 1)
        )

    def Splus_operator(self, u, v):
        # S_+ = 2 sqrt(3) gamma3 ({kx,kz} + i {ky,kz})
        return 2.0 * math.sqrt(3.0) * self.h2m * self.gamma3 * (
            self.sym_d(u, v, 0, 2) + 1.0j * self.sym_d(u, v, 1, 2)
            )
    

    def Sminus_operator(self, u, v):
        # S_- = 2 sqrt(3) gamma3 ({kx,kz} - i {ky,kz})
        return 2.0 * math.sqrt(3.0) * self.h2m * self.gamma3 * (
            self.sym_d(u, v, 0, 2) - 1.0j * self.sym_d(u, v, 1, 2)
            )
   

    def assemble_components(self, u1, u2, u3, u4, v1, v2, v3, v4):
        # Keep the same Hamiltonian layout as the original implementation.
        H11 = self.P_operator(u1, v1) * self.dx + self.Q_operator(u1, v1) * self.dx
        H22 = self.P_operator(u2, v2) * self.dx + self.Q_operator(u2, v2) * self.dx
        H33 = self.P_operator(u3, v3) * self.dx - self.Q_operator(u3, v3) * self.dx
        H44 = self.P_operator(u4, v4) * self.dx - self.Q_operator(u4, v4) * self.dx



        H13 = -self.Sminus_operator(u1, v3) * self.dx
        H31 = -self.Splus_operator(u3, v1) * self.dx

        H24 = -self.Splus_operator(u2, v4) * self.dx
        H42 = -self.Sminus_operator(u4, v2) * self.dx

        H14 =  self.R_operator(u1, v4) * self.dx
        H41 =  self.R_operator_conj(u4, v1) * self.dx

        H23 = -self.R_operator(u2, v3) * self.dx
        H32 = -self.R_operator_conj(u3, v2) * self.dx

        if self.phi is not None:
            self._msg(f"[LK4] Ev min: {np.real(self.Ev.x.array).min():.6e}")
            self._msg(f"[LK4] Ev max: {np.real(self.Ev.x.array).max():.6e}")
            H11 = H11 - self.Ev * ufl.inner(u1, v1) * self.dx
            H22 = H22 - self.Ev * ufl.inner(u2, v2) * self.dx
            H33 = H33 - self.Ev * ufl.inner(u3, v3) * self.dx
            H44 = H44 - self.Ev* ufl.inner(u4, v4) * self.dx

        if self.Bfield:
            Bx, By, Bz = self.B
            muB = 0.00005788
            self._msg(f"[LK4] Magnetic field: Bx={Bx}, By={By}, Bz={Bz}")

            H13 = H13 - muB * Bx * math.sqrt(3) * self.kappa * ufl.inner(u1, v3) * self.dx
            H24 = H24 + muB * Bx * math.sqrt(3) * self.kappa * ufl.inner(u2, v4) * self.dx
            H31 = H31 - muB * Bx * math.sqrt(3) * self.kappa * ufl.inner(u3, v1) * self.dx
            H42 = H42 + muB * Bx * math.sqrt(3) * self.kappa * ufl.inner(u4, v2) * self.dx

            H13 = H13 - 1j * muB * By * math.sqrt(3) * self.kappa * ufl.inner(u1, v3) * self.dx
            H24 = H24 + 1j * muB * By * math.sqrt(3) * self.kappa * ufl.inner(u2, v4) * self.dx
            H31 = H31 - (-1j) * muB * By * math.sqrt(3) * self.kappa * ufl.inner(u3, v1) * self.dx
            H42 = H42 + (-1j) * muB * By * math.sqrt(3) * self.kappa * ufl.inner(u4, v2) * self.dx

            H11 = H11 - 3 * muB * Bz * self.kappa * ufl.inner(u1, v1) * self.dx
            H22 = H22 + 3 * muB * Bz * self.kappa * ufl.inner(u2, v2) * self.dx
            H33 = H33 + muB * Bz * self.kappa * ufl.inner(u3, v3) * self.dx
            H44 = H44 - muB * Bz * self.kappa * ufl.inner(u4, v4) * self.dx

        if self.strain:
            P_e = -self.av* (self.eps_xx + self.eps_yy + self.eps_zz)
            Q_e = -self.bv / 2 * (self.eps_xx + self.eps_yy -2*self.eps_zz)
            R_e = math.sqrt(3) / 2 * self.bv * (self.eps_xx - self.eps_yy) - 1j * self.dv * self.eps_xy
            R_e_conj = math.sqrt(3) / 2 * self.bv * (self.eps_xx - self.eps_yy) + 1j * self.dv * self.eps_xy
            S_e = -self.dv * (self.eps_xz - 1j * self.eps_yz)
            S_e_conj = -self.dv * (self.eps_xz + 1j * self.eps_yz)

            H11 = H11 + P_e * ufl.inner(u1, v1) * self.dx + Q_e * ufl.inner(u1, v1) * self.dx
            H22 = H22 + P_e * ufl.inner(u2, v2) * self.dx + Q_e * ufl.inner(u2, v2) * self.dx
            H33 = H33 + P_e * ufl.inner(u3, v3) * self.dx - Q_e * ufl.inner(u3, v3) * self.dx
            H44 = H44 + P_e * ufl.inner(u4, v4) * self.dx - Q_e * ufl.inner(u4, v4) * self.dx

            H13 = H13 - S_e * ufl.inner(u1, v3) * self.dx
            H14 = H14 + R_e * ufl.inner(u1, v4) * self.dx
            H23 = H23 - R_e * ufl.inner(u2, v3) * self.dx
            H24 = H24 - S_e * ufl.inner(u2, v4) * self.dx
            H31 = H31 - S_e_conj * ufl.inner(u3, v1) * self.dx
            H32 = H32 - R_e_conj * ufl.inner(u3, v2) * self.dx
            H41 = H41 + R_e_conj * ufl.inner(u4, v1) * self.dx
            H42 = H42 - S_e * ufl.inner(u4, v2) * self.dx

        return (
            H11 + H13 + H14
            + H22 + H23 + H24
            + H31 + H32 + H33
            + H41 + H42 + H44
        )

    def create_mass_matrix(self, u1, u2, u3, u4, v1, v2, v3, v4):
        return (
            ufl.inner(u1, v1) * self.dx
            + ufl.inner(u2, v2) * self.dx
            + ufl.inner(u3, v3) * self.dx
            + ufl.inner(u4, v4) * self.dx
        )

    def solve_LK(self, nev=10, target=-1.0, shift_offset=0.05):
        P1 = element("Lagrange", self.domain.basix_cell(), 1)
        mixed_elem = basix.ufl.mixed_element([P1, P1, P1, P1])
        self.V = fem.functionspace(self.domain, mixed_elem)

        u1, u2, u3, u4 = ufl.TrialFunction(self.V)
        v1, v2, v3, v4 = ufl.TestFunction(self.V)
        a = self.assemble_components(u1, u2, u3, u4, v1, v2, v3, v4)

        self._msg(f"[LK4] Number of integrals in form: {len(a.integrals())}")

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

        b = self.create_mass_matrix(u1, u2, u3, u4, v1, v2, v3, v4)
        B = fem.petsc.assemble_matrix(fem.form(b), bcs=[bc])
        B.assemble()

        if self.verbose and self.domain.comm.rank == 0:
            info_A = A.getInfo()
            info_B = B.getInfo()
            print(f"[LK4] Non-zeros in A: {info_A['nz_used']}")
            print(f"[LK4] Non-zeros in B: {info_B['nz_used']}")
            print(f"[LK4] ||A|| = {A.norm():.6e}")
            print(f"[LK4] ||B|| = {B.norm():.6e}")

            AH = A.copy()
            AH.hermitianTranspose()
            D = A.copy()
            D.axpy(-1.0, AH)
            print(f"[LK4] ||A-A^H||/||A|| = {D.norm() / A.norm():.6e}")

            BH = B.copy()
            BH.hermitianTranspose()
            M = B.copy()
            M.axpy(-1.0, BH)
            print(f"[LK4] ||B-B^H||/||B|| = {M.norm() / B.norm():.6e}")

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
        self._msg(f"[LK4] Converged eigenvalues: {num_converged}")

        eigenvalues = []
        eigenvectors = []
        for i in range(min(num_converged, nev)):
            eigenval = eps.getEigenvalue(i)
            eigenvalues.append(eigenval.real)

            if self.verbose and self.domain.comm.rank == 0:
                rel = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
                absr = eps.computeError(i, SLEPc.EPS.ErrorType.ABSOLUTE)
                print(f"[LK4] i={i} E={eigenval * -1:.6e} rel={rel:.3e} abs={absr:.3e}")

            eigenfunction = fem.Function(self.V)
            eps.getEigenpair(i, eigenfunction.x.petsc_vec)
            eigenfunction.x.scatter_forward()
            eigenvectors.append(eigenfunction)

        self._log("Eigenvalues for 4-band model (eV):")
        for i, E in enumerate(eigenvalues):
            self._msg(f"[LK4] E_{i + 1} = {E:.8f}")
            self._log(f"  E_{i + 1} = {E:.8f}")

        eps.destroy()
        A.destroy()
        B.destroy()
        return eigenvalues, eigenvectors

    def save_probability_density_ground_state(self, eigenvectors, eigenvalues, output_file="ground_state_probability.vtu"):
        psi = eigenvectors[0]
        E = eigenvalues[0]

        num_comp = self.V.ufl_element().num_sub_elements
        band_names = ["HH_up", "HH_down", "LH_up", "LH_down"][:num_comp]

        V0, dof_map0 = self.V.sub(0).collapse()
        topology, cell_types, geometry = plot.vtk_mesh(V0)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        total_prob = np.zeros(len(dof_map0))
        self._msg(f"[LK4] Saving ground state probability density. E = {E:.6f} eV")

        for i, name in enumerate(band_names):
            _, dof_map = self.V.sub(i).collapse()
            vals = psi.x.array[dof_map]
            prob = np.abs(vals) ** 2
            total_prob += prob
            grid.point_data[f"{name}_abs2"] = prob
            self._msg(f"[LK4] {name}: max |psi|^2 = {prob.max():.3e}")

        grid.point_data["total_prob"] = total_prob
        grid.save(output_file)
        self._msg(f"[LK4] Saved to {output_file}")


# Backward-compatible alias
LK_hamiltonian_4band = LK4BandHamiltonian
