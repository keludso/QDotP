"""
===============================================================================
Project: kdotP 
Description: 8-band Luttinger-Kohn Hamiltonian solver using FEM

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


import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import math 
import basix.ufl
from basix.ufl import element, mixed_element
import dolfinx
from dolfinx import fem, mesh, io, plot
import ufl
from ufl import dx, grad, TestFunction, TrialFunction, inner, div
from dolfinx.mesh import CellType, exterior_facet_indices, locate_entities
import time
from dolfinx.io import XDMFFile
import pyvista as pv
from dolfinx.fem.petsc import assemble_matrix


class LK_hamiltonian_8band:

    """
    8-band Luttinger-Kohn Hamiltonian solver.

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
        self.Kane_Ep = device.Kane_Ep_sub

        self.av = device.av_sub
        self.bv = device.bv_sub
        self.dv = device.dv_sub

        # Tags for strain 
        self.cell_tags = getattr(self.device, "submesh_cell_tags", None)
        self.region_map = getattr(self.device, "submesh_region_map", {})

        self.Bfield = False

        self.strain = False

        # Keep original mesh rescaling logic.
        X = self.domain.geometry.x
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        extent = maxs - mins
        self._msg(f"[LK6] BBox min: {mins}, max: {maxs}, extent: {extent}")

        if np.max(extent) > 10:
            self._msg("[LK6] Mesh appears to already be in nm scale")
        else:
            self.domain.geometry.x[:] *= 1e9
            X = self.domain.geometry.x
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            extent = maxs - mins
            self._msg(f"[LK6] Rescaled mesh to nm. BBox min: {mins}, max: {maxs}, extent: {extent}")

        self.dx = ufl.Measure("dx", domain=self.domain)

        self.phi = phi
        self.chi = device.chi_sub
        self.Eg = device.Eg_sub

        self._msg(f"[LK8] phi min: {np.real(self.phi.x.array).min():.6e}")
        self._msg(f"[LK8] phi max: {np.real(self.phi.x.array).max():.6e}")
        self._msg(f"[LK8] chi min: {np.real(self.chi.x.array).min():.6e}")

        # hbar^2/(2m0) in the unit system used in the original code.
        self.h2m = 0.03809982
        self._msg(f"[LK8] h2m = {self.h2m}")

        self.P_coupling = math.sqrt(self.h2m)*ufl.sqrt(self.Kane_Ep)

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
        self.eps_xx = fem.Function(V)
        self.eps_yy = fem.Function(V)
        self.eps_zz = fem.Function(V)
        self.eps_xy = fem.Function(V)
        self.eps_xz = fem.Function(V)
        self.eps_yz = fem.Function(V)
        for f in [self.eps_xx, self.eps_yy, self.eps_zz, self.eps_xy, self.eps_xz, self.eps_yz]:
            f.x.array[:] = 0.0

        self._log("[LK6], Schrodinger Solver Started")
        self._log(f"[LK6], Initial Ev max: {np.real(self.Ev.x.array).max():.3e}")

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
        if not np.allclose(strain, strain.T, atol=1e-12):
            raise ValueError("strain_matrix must be symmetric")
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

        self._msg(f"[LK6] Applied strain to region '{region_name}'")
     

    def sym_d(self, u, v, i, j):
        return 0.5 * (
            ufl.inner(ufl.grad(u)[i], ufl.grad(v)[j])
            + ufl.inner(ufl.grad(u)[j], ufl.grad(v)[i])
        )


    def P_operator(self, u, v):
        return self.h2m * self.gamma1 * (
            ufl.inner(ufl.grad(u)[0], ufl.grad(v)[0])
            + ufl.inner(ufl.grad(u)[1], ufl.grad(v)[1])
            + ufl.inner(ufl.grad(u)[2], ufl.grad(v)[2])
        )


    def Q_operator(self, u, v):
        return self.h2m * self.gamma2 * (
            ufl.inner(ufl.grad(u)[0], ufl.grad(v)[0])
            + ufl.inner(ufl.grad(u)[1], ufl.grad(v)[1])
            - 2.0 * ufl.inner(ufl.grad(u)[2], ufl.grad(v)[2])
        )


    def R_operator(self, u, v):
        return (
            math.sqrt(3.0) * self.h2m * self.gamma2
            * (
                -ufl.inner(ufl.grad(u)[0], ufl.grad(v)[0])
                +  ufl.inner(ufl.grad(u)[1], ufl.grad(v)[1])
            )
            + 2.0j * math.sqrt(3.0) * self.h2m * self.gamma3 * self.sym_d(u, v, 0, 1)
        )


    def R_operator_conj(self, u, v):
        return (
            math.sqrt(3.0) * self.h2m * self.gamma2
            * (
                -ufl.inner(ufl.grad(u)[0], ufl.grad(v)[0])
                +  ufl.inner(ufl.grad(u)[1], ufl.grad(v)[1])
            )
            - 2.0j * math.sqrt(3.0) * self.h2m * self.gamma3 * self.sym_d(u, v, 0, 1)
        )


    def Splus_operator(self, u, v):
        return 2.0 * math.sqrt(3.0) * self.h2m * self.gamma3 * (
            self.sym_d(u, v, 0, 2) + 1.0j * self.sym_d(u, v, 1, 2)
        )


    def Sminus_operator(self, u, v):
        return 2.0 * math.sqrt(3.0) * self.h2m * self.gamma3 * (
            self.sym_d(u, v, 0, 2) - 1.0j * self.sym_d(u, v, 1, 2)
        )


    def Sigma_plus_operator(self, u, v):
        return math.sqrt(3.0) * self.h2m * self.gamma3 * (
            self.sym_d(u, v, 0, 2) + 1.0j * self.sym_d(u, v, 1, 2)
        )


    def Sigma_minus_operator(self, u, v):
        return math.sqrt(3.0) * self.h2m * self.gamma3 * (
            self.sym_d(u, v, 0, 2) - 1.0j * self.sym_d(u, v, 1, 2)
        )


    def CB_term(self, u, v, x):
        P = self.P_coupling
        return -0.5j * (
            P * ufl.Dx(u, x) * ufl.conj(v)
            + ufl.Dx(P * u, x) * ufl.conj(v)
        )


    def CB_term_adj(self, u, v, x):
        P = self.P_coupling
        return 0.5j * u * (
            ufl.conj(P * ufl.Dx(v, x))
            + ufl.conj(ufl.Dx(P * v, x))
        )


    def assemble_components(
        self,
        u1, u2, u3, u4, u5, u6, u7, u8,
        v1, v2, v3, v4, v5, v6, v7, v8
    ):
        rt12 = math.sqrt(1.0 / 2.0)
        rt16 = math.sqrt(1.0 / 6.0)
        rt13 = math.sqrt(1.0 / 3.0)
        rt23 = math.sqrt(2.0 / 3.0)
        rt2 = math.sqrt(2.0)
        rt32 = math.sqrt(3.0 / 2.0)

        # -----------------------------
        # Diagonal blocks
        # -----------------------------
        H11 = self.Ec * ufl.inner(u1, v1) * self.dx + self.h2m * (
            ufl.inner(ufl.grad(u1)[0], ufl.grad(v1)[0]) +
            ufl.inner(ufl.grad(u1)[1], ufl.grad(v1)[1]) +
            ufl.inner(ufl.grad(u1)[2], ufl.grad(v1)[2])
        ) * self.dx

        H22 = self.Ec * ufl.inner(u2, v2) * self.dx + self.h2m * (
            ufl.inner(ufl.grad(u2)[0], ufl.grad(v2)[0]) +
            ufl.inner(ufl.grad(u2)[1], ufl.grad(v2)[1]) +
            ufl.inner(ufl.grad(u2)[2], ufl.grad(v2)[2])
        ) * self.dx

        H33 = (self.P_operator(u3, v3) + self.Q_operator(u3, v3)) * self.dx
        H44 = (self.P_operator(u4, v4) + self.Q_operator(u4, v4)) * self.dx
        H55 = (self.P_operator(u5, v5) - self.Q_operator(u5, v5)) * self.dx
        H66 = (self.P_operator(u6, v6) - self.Q_operator(u6, v6)) * self.dx
        H77 = self.P_operator(u7, v7) * self.dx + self.delta * ufl.inner(u7, v7) * self.dx
        H88 = self.P_operator(u8, v8) * self.dx + self.delta * ufl.inner(u8, v8) * self.dx

        # -----------------------------
        # CB <-> VB couplings
        # -----------------------------
        H13 = -rt12 * self.P_coupling * (self.CB_term(u1, v3, 0) + 1.0j * self.CB_term(u1, v3, 1)) * self.dx
        H31 = -rt12 * self.P_coupling * (self.CB_term_adj(u3, v1, 0) - 1.0j * self.CB_term_adj(u3, v1, 1)) * self.dx

        H15 =  rt23 * self.P_coupling * self.CB_term(u1, v5, 2) * self.dx
        H51 =  rt23 * self.P_coupling * self.CB_term_adj(u5, v1, 2) * self.dx

        H16 =  rt16 * self.P_coupling * (self.CB_term(u1, v6, 0) - 1.0j * self.CB_term(u1, v6, 1)) * self.dx
        H61 =  rt16 * self.P_coupling * (self.CB_term_adj(u6, v1, 0) + 1.0j * self.CB_term_adj(u6, v1, 1)) * self.dx

        H17 = -rt13 * self.P_coupling * self.CB_term(u1, v7, 2) * self.dx
        H71 = -rt13 * self.P_coupling * self.CB_term_adj(u7, v1, 2) * self.dx

        H18 = -rt16 * self.P_coupling * (self.CB_term(u1, v8, 0) - 1.0j * self.CB_term(u1, v8, 1)) * self.dx
        H81 = -rt16 * self.P_coupling * (self.CB_term_adj(u8, v1, 0) - 1.0j * self.CB_term_adj(u8, v1, 1)) * self.dx

        H24 = -rt12 * self.P_coupling * (self.CB_term(u2, v4, 0) - 1.0j * self.CB_term(u2, v4, 1)) * self.dx
        H42 = -rt12 * self.P_coupling * (self.CB_term_adj(u4, v2, 0) + 1.0j * self.CB_term_adj(u4, v2, 1)) * self.dx

        H25 = -rt16 * self.P_coupling * (self.CB_term(u2, v5, 0) + 1.0j * self.CB_term(u2, v5, 1)) * self.dx
        H52 = -rt16 * self.P_coupling * (self.CB_term_adj(u5, v2, 0) - 1.0j * self.CB_term_adj(u5, v2, 1)) * self.dx

        H26 =  rt23 * self.P_coupling * self.CB_term(u2, v6, 2) * self.dx
        H62 =  rt23 * self.P_coupling * self.CB_term_adj(u6, v2, 2) * self.dx

        H27 = -rt13 * self.P_coupling * (self.CB_term(u2, v7, 0) + 1.0j * self.CB_term(u2, v7, 1)) * self.dx
        H72 = -rt13 * self.P_coupling * (self.CB_term_adj(u7, v2, 0) - 1.0j * self.CB_term_adj(u7, v2, 1)) * self.dx

        H28 =  rt13 * self.P_coupling * self.CB_term(u2, v8, 2) * self.dx
        H82 =  rt13 * self.P_coupling * self.CB_term_adj(u8, v2, 2) * self.dx

        # -----------------------------
        # Valence-valence couplings
        # -----------------------------
        H35 = -self.Sminus_operator(u3, v5) * self.dx
        H53 = -self.Splus_operator(u5, v3) * self.dx

        H36 =  self.R_operator(u3, v6) * self.dx
        H63 =  self.R_operator_conj(u6, v3) * self.dx

        H37 =  (1.0 / math.sqrt(2.0)) * self.Sminus_operator(u3, v7) * self.dx
        H73 =  (1.0 / math.sqrt(2.0)) * self.Splus_operator(u7, v3) * self.dx

        H38 =  rt2 * self.R_operator(u3, v8) * self.dx
        H83 =  rt2 * self.R_operator_conj(u8, v3) * self.dx

        H45 = -self.R_operator_conj(u4, v5) * self.dx
        H54 = -self.R_operator(u5, v4) * self.dx

        H46 = -self.Splus_operator(u4, v6) * self.dx
        H64 = -self.Sminus_operator(u6, v4) * self.dx

        H47 = -rt2 * self.R_operator_conj(u4, v7) * self.dx
        H74 = -rt2 * self.R_operator(u7, v4) * self.dx

        H48 =  (1.0 / math.sqrt(2.0)) * self.Splus_operator(u4, v8) * self.dx
        H84 =  (1.0 / math.sqrt(2.0)) * self.Sminus_operator(u8, v4) * self.dx

        H57 =  rt2 * self.Q_operator(u5, v7) * self.dx
        H75 =  rt2 * self.Q_operator(u7, v5) * self.dx

        H58 =  rt32 * self.Sigma_minus_operator(u5, v8) * self.dx
        H85 =  rt32 * self.Sigma_plus_operator(u8, v5) * self.dx

        H67 = -rt32 * self.Sigma_plus_operator(u6, v7) * self.dx
        H76 = -rt32 * self.Sigma_minus_operator(u7, v6) * self.dx

        H68 =  rt2 * self.Q_operator(u6, v8) * self.dx
        H86 =  rt2 * self.Q_operator(u8, v6) * self.dx

        # -----------------------------
        # Scalar valence band edge
        # -----------------------------
        if self.phi is not None:
            self._msg(f"[LK8] Ev min: {np.real(self.Ev.x.array).min():.6e}")
            self._msg(f"[LK8] Ev max: {np.real(self.Ev.x.array).max():.6e}")

            H33 += -self.Ev * ufl.inner(u3, v3) * self.dx
            H44 += -self.Ev * ufl.inner(u4, v4) * self.dx
            H55 += -self.Ev * ufl.inner(u5, v5) * self.dx
            H66 += -self.Ev * ufl.inner(u6, v6) * self.dx
            H77 += -self.Ev * ufl.inner(u7, v7) * self.dx
            H88 += -self.Ev * ufl.inner(u8, v8) * self.dx

        a = (
            H11 + H22 + H33 + H44 + H55 + H66 + H77 + H88
            + H13 + H31 + H15 + H51 + H16 + H61 + H17 + H71 + H18 + H81
            + H24 + H42 + H25 + H52 + H26 + H62 + H27 + H72 + H28 + H82
            + H35 + H53 + H36 + H63 + H37 + H73 + H38 + H83
            + H45 + H54 + H46 + H64 + H47 + H74 + H48 + H84
            + H57 + H75 + H58 + H85 + H67 + H76 + H68 + H86
        )

        return a
    

    def create_mass_matrix(self, u1, u2, u3, u4,u5,u6, u7, u8, v1, v2, v3, v4,v5,v6,v7, v8):

        b = (ufl.inner(u1,v1) * self.dx + 
             ufl.inner(u2,v2) * self.dx + 
             ufl.inner(u3,v3) * self.dx + 
             ufl.inner(u4,v4) * self.dx + 
             ufl.inner(u5,v5) * self.dx + 
             ufl.inner(u6,v6) * self.dx + 
             ufl.inner(u7,v7) * self.dx +
             ufl.inner(u8,v8) * self.dx )
        return b
    

    def solve_LK(self):
        scalar_type = PETSc.ScalarType  # complex128 if PETSc built with complex
        P1 = element("Lagrange", self.domain.basix_cell(), 1)
        mixed_elem = basix.ufl.mixed_element([P1, P1, P1, P1,P1,P1, P1, P1])
        self.V = fem.functionspace(self.domain, mixed_elem)

        u1, u2, u3, u4, u5, u6, u7, u8 = ufl.TrialFunction(self.V)
        v1, v2, v3, v4, v5, v6, v7, v8 = ufl.TestFunction(self.V)

        a = self.assemble_components(u1, u2, u3, u4, u5,u6, u7, u8, v1, v2, v3, v4,v5,v6, v7, v8)

        print("\nForm 'a' integrals:")
        print(f"Number of integrals in form: {len(a.integrals())}")

        # Boundary conditions
        print("Setting up boundary conditions...")
        self.domain.topology.create_connectivity(self.domain.topology.dim - 1, self.domain.topology.dim)

        bc_facets = exterior_facet_indices(self.domain.topology)
        bc_dofs = fem.locate_dofs_topological(self.V, self.domain.topology.dim - 1, bc_facets)
        u_bc = fem.Function(self.V)
        with u_bc.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        bc = fem.dirichletbc(u_bc, bc_dofs)

        # Assemble matrices
        A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
        A.assemble()

        b = self.create_mass_matrix(u1, u2, u3, u4, u5,u6, u7, u8, v1, v2, v3, v4,v5,v6, v7, v8)
        B = fem.petsc.assemble_matrix(fem.form(b), bcs=[bc])
        B.assemble()

   
        # Matrix diagnostics
        info_A = A.getInfo()
        info_B = B.getInfo()
        print(f"\nNon-zeros in A: {info_A['nz_used']}")
        print(f"Non-zeros in B: {info_B['nz_used']}")

        print(f"\n1. Matrix Norms:")
        print(f"   ||A|| = {A.norm():.6e}")
        print(f"   ||B|| = {B.norm():.6e}")

        #A.view(PETSc.Viewer().createASCII(comm=self.domain.comm))

        # Check symmetry
        AH = A.copy()
        AH.hermitianTranspose()  # Compute A^H = conj(A^T)
        D = A.copy()
        D.axpy(-1.0, AH)  # D = A - A^H
        hermiticity_error = D.norm() / A.norm()
        print(f"   ||A - A^H|| / ||A|| = {hermiticity_error:.6e}")
        print(f"||A||/||B|| = {A.norm()/B.norm():.4f}")  # rough spectral radius

        BH = B.copy()
        BH.hermitianTranspose()  # Compute A^H = conj(A^T)
        M = B.copy()
        M.axpy(-1.0, BH)  # D = A - A^H
        hermiticity_error = M.norm() / B.norm()
        print(f"   ||B - B^H|| / ||B|| = {hermiticity_error:.6e}")
     

        print("\n" + "="*70)
        print("SOLVING GENERALIZED EIGENVALUE PROBLEM")
        print("="*70)
        Ev_max = 0.0 #4.7
        # Calculating the target value 
        L = 150
        target = -1.0 #-self.h2m*13.38*math.pi**2/L**2 -Ev_max
        print(f"Taget ={target}")

        nev = 6 # Number of eigenvalues to compute

        eps = SLEPc.EPS().create(self.domain.comm)
        eps.setOperators(A, B)                                    #  MUST be before getST()
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(target)
        eps.setDimensions(nev=nev, ncv= 100)
        eps.setTolerances(tol=1e-8, max_it=500)

        # Step 2: Configure ST AFTER setOperators
        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(target+0.05)                                       #  use target directly, no offset

        # Step 3: Configure KSP/PC on the ST
        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")


        #ksp = st.getKSP()
        #ksp.setType("fgmres")
        #pc = ksp.getPC()
        #pc.setType("gamg")
  
        pc.setFactorSolverType("superlu_dist")

        # Step 4: MUMPS options must be set on the ST's KSP prefix
        st.setOptionsPrefix("st_")
        opts = PETSc.Options()

        # Step 5: Finalize and solve
        eps.setFromOptions()
        eps.setMonitor(lambda eps, it, nconv, eig, err: 
                       print(f"Iter {it:3d}, converged {nconv}, ?={eig[0]:.4e}, err={err[0]:.2e}"))
        
        eps.solve()
        eps.view()
        eps.errorView()

        
        print("Solver converged!")

        num_converged = eps.getConverged()
        print(f"Converged: {num_converged} eigenvalues\n")

        eigenvalues  = []
        eigenvectors = []

        for i in range(min(num_converged, nev)):
            eigenval = eps.getEigenvalue(i)
            eigenvalues.append(eigenval.real)

            rel  = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
            absr = eps.computeError(i, SLEPc.EPS.ErrorType.ABSOLUTE)
            print(f"i={i}  E={eigenval*-1:.6e}  rel={rel:.3e}  abs={absr:.3e}")

            # Create a NEW Function each iteration â€” reusing the same object
            # means all entries in eigenvectors point to the same data
            eigenfunction = fem.Function(self.V)
            eps.getEigenpair(i, eigenfunction.x.petsc_vec)
            eigenfunction.x.scatter_forward()

            eigenvectors.append(eigenfunction)

        # Print results
        print("\nEigenvalues (eV):")
        if self.log_writer is not None:
            self.log_writer.write("\nEigenvalues for 6-band model (eV):")
        for i, E in enumerate(eigenvalues):
            print(f"  E_{i+1} = {E:.6f}")
            if self.log_writer is not None:
                self.log_writer.write(f"  E_{i+1} = {E:.6f}")

        eps.destroy()
        A.destroy()
        B.destroy()

        return eigenvalues, eigenvectors

    def save_probability_density_ground_state(
        self, eigenvectors, eigenvalues, output_file="ground_state_probability.vtu"):

        psi = eigenvectors[0]
        E   = eigenvalues[0]

        # Check actual number of components
        num_comp = self.V.ufl_element().num_sub_elements
        print(f"  Mixed space has {num_comp} components")
        print(f"  Full eigenvector size: {len(psi.x.array)}")

        # Match to actual number of components in your mixed element
        all_band_names = ["HH_up", "HH_down", "LH_up", "LH_down", "SO_up", "SO_down","SO_up1", "SO_down1", "SO_down2","SO_up3", "SO_down3"]
        band_names = all_band_names[:num_comp]

        V0, dof_map0 = self.V.sub(0).collapse()
        topology, cell_types, geometry = plot.vtk_mesh(V0)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        total_prob = np.zeros(len(dof_map0))

        print(f"\nSaving ground state probability density")
        print(f"Energy: E = {E:.6f} eV")
        print("=" * 70)

        for i, name in enumerate(band_names):
            _, dof_map = self.V.sub(i).collapse()
            vals = psi.x.array[dof_map]

            prob      = np.abs(vals)**2
 
            print(f"  {name}: maxsi={prob.max():.3e}")
            grid.point_data[f"{name}_abs2"] = prob

        grid.point_data["total_prob"] = total_prob
        grid.save(output_file)
        print(f"  Saved to {output_file}")

