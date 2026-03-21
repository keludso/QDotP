import ufl
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
from dolfinx import fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py.PETSc import Options
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
import pyvista as pv
from dolfinx import plot

class nonlinear_solver:

    def __init__(self, domain, device, fermi):
        self.domain = domain
        self.device = device

        self.eps0 = 8.8541878128e-12
        
        # Create P1 versions of material parameters to avoid DG/Lagrange mixing
        V_P1 = fem.functionspace(domain, ('Lagrange', 1))
        
        self.eps = fem.Function(V_P1)
        self.eps.interpolate(device.eps)
        
        self.Eg = fem.Function(V_P1)
        self.Eg.interpolate(device.Eg)
        
        self.Nc = fem.Function(V_P1)
        self.Nc.interpolate(device.Nc)
        
        self.Nv = fem.Function(V_P1)
        self.Nv.interpolate(device.Nv)
        
        self.chi = fem.Function(V_P1)
        self.chi.interpolate(device.chi)

        self.Ev = fem.Function(V_P1, dtype=np.complex128)
        
        self.phiref = fem.Constant(self.domain, PETSc.ScalarType(4.7 - 0j))
        self.Ef = fem.Constant(self.domain, PETSc.ScalarType(fermi - 0j))

        # Calculate band edges

        Ec_cal = fem.Function(V_P1, dtype=np.complex128)
        Ev_cal = fem.Function(V_P1, dtype=np.complex128)

        Ec_cal.x.array[:] = np.complex128(-(self.phiref + self.chi.x.array))
        Ev_cal.x.array[:] = np.complex128(Ec_cal.x.array[:] - self.Eg.x.array)

        topology, cell_types, geometry = plot.vtk_mesh(V_P1)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["chi"] = np.real(self.chi.x.array)
        grid.point_data["EC"] = np.real(Ec_cal.x.array)
        grid.point_data["Ev"] = np.real(Ev_cal.x.array)
        grid.save("Start_check.vtu")
        
        print(f"Material parameters converted to real-valued:")
        print(f"  eps mean: {self.eps.x.array.mean():.3e}")
        print(f"  chi mean: {self.chi.x.array.mean():.3e}")



    def nonlin_solve(self, gate_voltages, ohmic_gate_name, save_file=True, max_iterations=10, tol=1e-9):
        """
        Solve nonlinear Poisson equation with charge density.
        
        Strategy:
        1. Linear solve (Laplace) for initial guess
        2. Nonlinear solve (Poisson with charge) from that guess
        """

        # Create function space and functions
        V = fem.functionspace(self.domain, ('Lagrange', 1))
        phi = fem.Function(V, dtype=np.complex128)  # Solution (complex-valued)
        dphi = ufl.TrialFunction(V)        # Newton increment
        # Define variational problem
        v = ufl.TestFunction(V)      # Test function
        u = ufl.TrialFunction(V)     # Trial function (for both linear and Newton increment)
        tdim = self.domain.topology.dim
        fdim = tdim - 1

        # =====================================================
        # BOUNDARY CONDITIONS
        # =====================================================
        bcs = []
        if self.domain.comm.rank == 0:
            print("\n" + "="*60)
            print("BOUNDARY CONDITIONS")
            print("="*60)
            print(f"{'Gate Name':<20} {'Voltage':>10}  {'DOFs':>8}")
            print("-"*60)
        
        for gate_name, voltage in gate_voltages.items():
            if gate_name not in self.device.facet_tags_map:
                continue
            
            tag = self.device.facet_tags_map[gate_name]
            dofs = fem.locate_dofs_topological(V, fdim, self.device.facet_tags.find(tag)) 
            if len(dofs) == 0:
                continue
            
            phi_v =  voltage    # Based on QCAD paper 
            bc = fem.dirichletbc(np.complex128(phi_v + 0j), dofs, V)
            bcs.append(bc)
            if self.domain.comm.rank == 0:
                print(f"{gate_name:<20} {phi_v:>10.3f} V {len(dofs):>8}")
        
        # Setting Ohmic Gates
        for gate_name in ohmic_gate_name:
            if gate_name not in self.device.facet_tags_map:
                raise KeyError(f"Boundary '{gate_name}' not found. Available: {list(self.device.facet_tags_map.keys())}")

            phys_tag = self.device.facet_tags_map[gate_name]
            facets = self.device.facet_tags.find(phys_tag)
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            bc_val = PETSc.ScalarType(0.0)
            bc = fem.dirichletbc(bc_val, dofs, V)
            bcs.append(bc)

        if self.domain.comm.rank == 0:
            print("="*60 + "\n")

        phi.x.array[:] = 0.0

        print(f"eps mean: {self.eps.x.array.mean():.3e}")
        print(f"Epsilon min:  {self.eps.x.array.min():.3e}")
        print(f"Epsilon max:  {self.eps.x.array.max():.3e}")
        print(f"Number of zeros in eps: {(self.eps.x.array == 0).sum()}")

        # =====================================================
        # STEP 1: LINEAR SOLVE (Laplace equation for initial guess)
        # =====================================================
        print("\nStep 1: Solving linear Laplace equation for initial guess...")
        print("="*60)
        
        # Linear form: a(u,v) = L(v)
        # In complex mode, MUST conjugate test function
        a_linear = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L_linear = 0.001 * ufl.conj(v) * ufl.dx



        problem_linear = fem.petsc.LinearProblem(
            a_linear, L_linear, bcs=bcs, 
            petsc_options_prefix="poisson_linear_",
            petsc_options={"ksp_type": "gmres", "pc_type": "gamg", "ksp_rtol": 1e-8}
        )
        phi_init = problem_linear.solve()
        
        # Copy initial guess to phi
        phi.x.array[:] = phi_init.x.array


        
        if self.domain.comm.rank == 0:
            print(f"Linear solve complete.")
            print(f"  phi min: {np.real(phi.x.array).min():.3e}")
            print(f"  phi max: {np.real(phi.x.array).max():.3e}")

        # =====================================================
        # STEP 2: NONLINEAR SOLVE (Poisson with charge density)
        # =====================================================
        print("\nStep 2: Solving nonlinear Poisson with charge density...")
        print("="*60)
        phi_old = fem.Function(V)
        

        print("\nStep 2: Solving nonlinear Poisson with charge density...")
        print("="*60)

        def rho_ufl(phi):

            kB = 1.380649e-23
            T  = 10.0
            kbT = kB * T / 1.6e-19
            q   = 1.6e-19

            Ec = -(phi - self.phiref + self.chi)
            Ev = Ec - self.Eg

            xe = (self.Ef - Ec) / kbT
            xh = (Ev - self.Ef) / kbT


            xe_c = 50.0 * ufl.tanh(xe / 50.0)

            step = 0.5 * (1.0 + ufl.tanh(xe_c))   
            xpos = 0.5 * (xe_c + ufl.sqrt(xe_c * xe_c + 1e-6))
            pow_part = 0.7523 * (xpos)**1.5
            nn = step * pow_part

            xh_c = 50.0 * ufl.tanh(xh / 50.0)

            step = 0.5 * (1.0 + ufl.tanh(xh_c))   
            xpos = 0.5 * (xh_c + ufl.sqrt(xh_c * xh_c + 1e-6))
            pow_part = 0.7523 * (xpos)**1.5
            pp = step * pow_part


            n = self.Nc * nn
            p = self.Nv * pp

            return q * (p - n)


        def rho_debug(phi):

            V = phi.function_space
            points = V.element.interpolation_points
            rho_expr = rho_ufl(phi)

            rho_func = fem.Function(V)
            rho_func.interpolate(fem.Expression(rho_expr, points))
            print(f"  Rho min: {np.real(rho_func.x.array).min():.3e}")
            print(f"  Rho max: {np.real(rho_func.x.array).max():.3e}")


        # Debug only
        rho_debug(phi)

        # Variational form uses only pure UFL
        F = (ufl.inner(self.eps * ufl.grad(phi), ufl.grad(v)) * ufl.dx
            - ufl.inner(rho_ufl(phi), v) * ufl.dx)

        J = ufl.derivative(F, phi, dphi)

        problem = NonlinearProblem(F, phi, J=J, bcs=bcs,
                           petsc_options_prefix="nls_solve_",
                           petsc_options={
                               "snes_type": "newtonls",
                               "snes_linesearch_type": "bt",
                               "snes_atol": 1e-8,
                               "snes_rtol": 1e-7,
                               "snes_max_it": 100,
                               "snes_monitor": "",
                               "ksp_type": "gmres",
                               "ksp_rtol": 1e-8,
                               "pc_type": "gamg"
                           })

        phi = problem.solve()
        converged_reason = problem.solver.getConvergedReason()
        print(f"  Converged reason: {converged_reason}")
        print(f"  phi min: {np.real(phi.x.array).min():.3e}")
        print(f"  phi max: {np.real(phi.x.array).max():.3e}")
                # =====================================================
        # SAVE RESULTS
        # =====================================================
        if save_file:
            chii_P1 = self.chi
            Eg_P1 = self.Eg

            Ec_cal = fem.Function(V)
            Ev_cal = fem.Function(V)

            # Calculate band edges from solution
            Ec_cal.x.array[:] = np.complex128(-(phi.x.array + chii_P1.x.array))
            Ev_cal.x.array[:] = np.complex128(Ec_cal.x.array[:] - Eg_P1.x.array)

            # Convert to VTK for visualization
            topology, cell_types, geometry = plot.vtk_mesh(V)
            grid = pv.UnstructuredGrid(topology, cell_types, geometry)
            grid.point_data["phi"] = np.real(phi.x.array)
            grid.point_data["EC"] = np.real(Ec_cal.x.array)
            grid.point_data["Ev"] = np.real(Ev_cal.x.array)
            grid.save("poisson.vtu")
            
            if self.domain.comm.rank == 0:
                print(f"\n✓ Results saved to poisson.vtu")

        return phi