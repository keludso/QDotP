import numpy as np
import pyvista as pv
import ufl
from petsc4py import PETSc
from dolfinx import fem, plot
from dolfinx.fem.petsc import NonlinearProblem


class NonlinearPoissonSolver:
    """
    Nonlinear Poisson solver that works directly with a Device object.

    The solver reuses the material functions stored in the Device class and
    takes the mesh from device.domain, so it can be shared cleanly across the
    package.
    """

    def __init__(self, device, fermi, verbose=False, log_writer=None):
        self.device = device
        self.domain = device.domain
        self.verbose = verbose
        self.log_writer = log_writer if log_writer is not None else getattr(device, "log_writer", None)

        self.eps0 = 8.8541878128e-12

        self.eps = device.eps 
        self.Eg = device.Eg
        self.Nc = device.Nc
        self.Nv = device.Nv
        self.chi = device.chi

        self.phiref = fem.Constant(self.domain, PETSc.ScalarType(4.3963- - 0j))
        self.Ef = fem.Constant(self.domain, PETSc.ScalarType(fermi - 0j))

    def _log(self, message):
        if self.log_writer is not None:
            self.log_writer.write(message)

    def _msg(self, message):
        if self.domain.comm.rank == 0 and self.verbose:
            print(message)
        self._log(message)

    def rho_ufl(self,phi):
        #Charge density

        kB = 1.380649e-23
        q  = 1.6e-19
        T  = 10.0                    # Kelvin
        kbT = kB * T / q             # in Volt (or eV if consistent)

        # Band edges — double-check the sign convention!
        Ec = -(phi - self.phiref + self.chi)
        Ev = Ec - self.Eg

        eta_e = (self.Ef - Ec) / kbT      # reduced Fermi level for electrons
        eta_h = (Ev - self.Ef) / kbT      # for holes


        def fermi_half(eta):
            # Joyce-Dixon like approximation or simple smooth transition
            # For many device simulations, people use:
            return ufl.exp(eta) / (1.0 + 0.27 * ufl.exp(eta))   # rough but stable

            # Or a better one (if you need high accuracy in degenerate regime):
            # exp_part = ufl.exp(eta)
            # return exp_part * (1.0 - 0.295 * exp_part / (1.0 + 0.5 * exp_part))  # example

        nn = fermi_half(eta_e)
        pp = fermi_half(eta_h)

        n = self.Nc * nn
        p = self.Nv * pp

        return q * (p - n)

    def _rho_debug(self, phi):
        V = phi.function_space
        points = V.element.interpolation_points
        rho_expr = self.rho_ufl(phi)
        rho_func = fem.Function(V)
        rho_func.interpolate(fem.Expression(rho_expr, points))
        self._msg(f"[POISSON] Rho min: {np.real(rho_func.x.array).min():.3e}")
        self._msg(f"[POISSON] Rho max: {np.real(rho_func.x.array).max():.3e}")

    def nonlin_solve(self, gate_voltages, ohmic_gate_name, save_file=True):
        """
        Solve the nonlinear Poisson equation.

        The overall flow is preserved from the original file:
        1. Solve a linear Laplace problem for the initial guess.
        2. Build the nonlinear residual and Jacobian.
        3. Solve the nonlinear system.
        """
        V = fem.functionspace(self.domain, ("Lagrange", 1))
        phi = fem.Function(V, dtype=np.complex128)
        dphi = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)

        tdim = self.domain.topology.dim
        fdim = tdim - 1

        bcs = []
        if self.domain.comm.rank == 0 and self.verbose:
            print("\n" + "=" * 60)
            print("BOUNDARY CONDITIONS")
            print("=" * 60)
            print(f"{'Gate Name':<20} {'Voltage':>10}  {'DOFs':>8}")
            print("-" * 60)

        for gate_name, voltage in gate_voltages.items():
            if gate_name not in self.device.facet_tags_map:
                continue
            tag = self.device.facet_tags_map[gate_name]
            dofs = fem.locate_dofs_topological(V, fdim, self.device.facet_tags.find(tag))
            if len(dofs) == 0:
                continue
            bc = fem.dirichletbc(np.complex128(voltage + 0j), dofs, V)
            bcs.append(bc)
            self._msg(f"[POISSON] Gate {gate_name}: {voltage:>10.3f} V, dofs={len(dofs)}")

        for gate_name in ohmic_gate_name:
            if gate_name not in self.device.facet_tags_map:
                raise KeyError(
                    f"Boundary '{gate_name}' not found. "
                    f"Available: {list(self.device.facet_tags_map.keys())}"
                )
            phys_tag = self.device.facet_tags_map[gate_name]
            facets = self.device.facet_tags.find(phys_tag)
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V))
            self._msg(f"[POISSON] Ohmic {gate_name}: grounded, dofs={len(dofs)}")

        # Step 1: linear solve for initial guess
        a_linear = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L_linear = 0.5 * ufl.conj(v) * ufl.dx
        problem_linear = fem.petsc.LinearProblem(
            a_linear,
            L_linear,
            bcs=bcs,
            petsc_options_prefix="poisson_linear_",
            petsc_options={"ksp_type": "gmres", "pc_type": "gamg", "ksp_rtol": 1e-8},
        )
        phi_init = problem_linear.solve()
        phi.x.array[:] = phi_init.x.array
        self._msg(f"[POISSON] Linear solve phi min: {np.real(phi.x.array).min():.3e}")
        self._msg(f"[POISSON] Linear solve phi max: {np.real(phi.x.array).max():.3e}")

        # Step 2: nonlinear solve
        self._rho_debug(phi)
        F = ufl.inner(self.eps * ufl.grad(phi), ufl.grad(v)) * ufl.dx - ufl.inner(
            self.rho_ufl(phi), v
        ) * ufl.dx
        J = ufl.derivative(F, phi, dphi)

        problem = NonlinearProblem(
            F,
            phi,
            J=J,
            bcs=bcs,
            petsc_options_prefix="nls_solve_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",          # or try "cp" or "none" for testing
                "snes_atol": 1e-10,
                "snes_rtol": 1e-8,
                "snes_stol": 1e-12,                    # step tolerance
                "snes_max_it": 50,
                "snes_monitor": None,                  # prints residual each iteration
                "snes_error_if_not_converged": True,   # raise exception on failure (recommended during debugging)
                "ksp_type": "gmres",
                "ksp_rtol": 1e-10,
                "ksp_max_it": 200,
                "pc_type": "gamg",                     # algebraic multigrid — good for Poisson
                })

        phi = problem.solve()
        converged_reason = problem.solver.getConvergedReason()
        print(f"  Converged reason: {converged_reason}")
        
        print(f"  phi min: {np.real(phi.x.array).min():.3e}")
        print(f"  phi max: {np.real(phi.x.array).max():.3e}")

        self._msg(f"[POISSON] Nonlinear phi min: {np.real(phi.x.array).min():.3e}")
        self._msg(f"[POISSON] Nonlinear phi max: {np.real(phi.x.array).max():.3e}")

        if save_file:
            Q = fem.functionspace(self.domain, ("DG", 0))

            Ec = fem.Function(Q)
            Ev = fem.Function(Q)

            phi_ref = self.phiref

            Ec_expr =  -(phi -phi_ref + self.chi) # Use chii_P1, not chii
            Ev_expr = Ec - self.Eg

            Ec.interpolate(fem.Expression(Ec_expr, Q.element.interpolation_points))
            Ev.interpolate(fem.Expression(Ev_expr, Q.element.interpolation_points))

            # Convert to VTK for visualization
            topology, cell_types, geometry = plot.vtk_mesh(self.domain, self.domain.topology.dim)
            grid = pv.UnstructuredGrid(topology, cell_types, geometry)

            grid.point_data["phi"] = np.real(phi.x.array)
            grid.cell_data["EC"] = np.real(Ec.x.array)
            grid.cell_data["Ev"] = np.real(Ev.x.array)
            grid.save("poisson.vtu")
            
            if self.domain.comm.rank == 0:
                print(f"\n? Results saved to poisson.vtu")
            self._msg("[POISSON] Results saved to poisson.vtu")

        return phi


# Backward-compatible alias
nonlinear_solver = NonlinearPoissonSolver
