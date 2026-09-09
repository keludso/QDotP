import os
import numpy as np
import pyvista as pv
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, plot, io
from dolfinx.fem.petsc import LinearProblem
from scipy.spatial import cKDTree


class Rabi_solver:
    """
    Rabi-frequency helper for a FEniCSx k.p quantum-dot solver.

    Main workflow
    -------------
    1. calculate_D1(...)
        Solve the linear gate-response problem on the parent/device mesh:

            div(eps grad D1) = 0

        with:
            D1 = 1 on the RF gate
            D1 = 0 on grounded metallic/ohmic contacts

        D1 = dphi / dV_gate is dimensionless, V/V.

    2. restrict_parent_scalar_to_submesh(...)
        Restrict D1 from the parent Poisson mesh to the k.p submesh using
        the vertex_map returned by device.create_submesh(...).

        This avoids geometric interpolation problems when:
            parent mesh coordinates are in meters
            k.p submesh coordinates are rescaled to nm

    3. M01(...)
        Compute:

            Mij = <psi_i | operator_sign * D1 | psi_j>

        Default operator_sign = -1.0, corresponding to A_g = -D1.

    4. rabi_frequency(...)
        Compute:

            f_R = |Vac * Mij| / h
    """

    def __init__(self, device, verbose=False, log_writer=None):
        self.device = device
        self.domain = device.domain
        self.verbose = verbose
        self.log_writer = (
            log_writer if log_writer is not None else getattr(device, "log_writer", None)
        )

        self.eps0 = 8.8541878128e-12

        # Parent-mesh material functions
        self.eps = device.eps
        self.Eg = getattr(device, "Eg", None)
        self.Nc = getattr(device, "Nc", None)
        self.Nv = getattr(device, "Nv", None)
        self.chi = getattr(device, "chi", None)

        self.phiref = fem.Constant(self.domain, PETSc.ScalarType(4.3963))
        self.Ef = fem.Constant(self.domain, PETSc.ScalarType(0.0))

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    def _log(self, message):
        if self.log_writer is not None:
            self.log_writer.write(str(message) + "\n")

    def _rank0_print(self, message, comm=None):
        if comm is None:
            comm = self.domain.comm
        if comm.rank == 0:
            print(message)

    def _msg(self, message, comm=None):
        if comm is None:
            comm = self.domain.comm
        if comm.rank == 0 and self.verbose:
            print(message)
        self._log(message)

    # -------------------------------------------------------------------------
    # Debug utilities
    # -------------------------------------------------------------------------
    def _global_minmax_array(self, arr, comm):
        arr = np.asarray(arr)
        if arr.size == 0:
            local_min = np.inf
            local_max = -np.inf
        else:
            local_min = np.min(np.real(arr))
            local_max = np.max(np.real(arr))

        global_min = comm.allreduce(local_min, op=MPI.MIN)
        global_max = comm.allreduce(local_max, op=MPI.MAX)

        return global_min, global_max

    def _global_minmax_function(self, function):
        mesh = function.function_space.mesh
        return self._global_minmax_array(function.x.array, mesh.comm)

    def _mesh_bbox(self, mesh, label):
        X = mesh.geometry.x

        if X.shape[0] == 0:
            local_min = np.full(mesh.geometry.dim, np.inf)
            local_max = np.full(mesh.geometry.dim, -np.inf)
        else:
            local_min = np.min(X, axis=0)
            local_max = np.max(X, axis=0)

        global_min = np.array(
            [mesh.comm.allreduce(local_min[i], op=MPI.MIN) for i in range(len(local_min))]
        )
        global_max = np.array(
            [mesh.comm.allreduce(local_max[i], op=MPI.MAX) for i in range(len(local_max))]
        )

        if mesh.comm.rank == 0:
            print(f"[{label}] bbox min = {global_min}")
            print(f"[{label}] bbox max = {global_max}")
            print(f"[{label}] extent   = {global_max - global_min}")

        return global_min, global_max

    def _print_function_stats(self, function, label):
        fmin, fmax = self._global_minmax_function(function)
        mesh = function.function_space.mesh

        if mesh.comm.rank == 0:
            print(f"[{label}] min = {fmin:.6e}")
            print(f"[{label}] max = {fmax:.6e}")
            print(f"[{label}] num local dofs = {len(function.x.array)}")

        return fmin, fmax

    # -------------------------------------------------------------------------
    # Save scalar function
    # -------------------------------------------------------------------------
    def _save_scalar_function(self, function, output_file):
        mesh = function.function_space.mesh
        comm = mesh.comm

        ext = os.path.splitext(output_file)[1].lower()

        if ext == ".xdmf":
            with io.XDMFFile(comm, output_file, "w") as xdmf:
                xdmf.write_mesh(mesh)
                xdmf.write_function(function)

            if comm.rank == 0:
                print(f"[SAVE] Saved {function.name} to {output_file}")
            return

        if ext == ".vtu":
            if comm.size == 1:
                topology, cell_types, geometry = plot.vtk_mesh(
                    mesh,
                    mesh.topology.dim,
                )

                grid = pv.UnstructuredGrid(topology, cell_types, geometry)
                grid.point_data[function.name] = np.real(function.x.array)
                grid.save(output_file)

                if comm.rank == 0:
                    print(f"[SAVE] Saved {function.name} to {output_file}")
            else:
                fallback = os.path.splitext(output_file)[0] + ".xdmf"

                if comm.rank == 0:
                    print(
                        "[SAVE] Parallel .vtu output with PyVista is not safe. "
                        f"Saving XDMF instead: {fallback}"
                    )

                with io.XDMFFile(comm, fallback, "w") as xdmf:
                    xdmf.write_mesh(mesh)
                    xdmf.write_function(function)

            return

        fallback = output_file + ".xdmf"

        with io.XDMFFile(comm, fallback, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(function)

        if comm.rank == 0:
            print(f"[SAVE] Unknown extension. Saved XDMF instead: {fallback}")

    # -------------------------------------------------------------------------
    # Linear gate-response solve on parent/device mesh
    # -------------------------------------------------------------------------
    def calculate_D1(
        self,
        rf_gate_name,
        grounded_gate_names=None,
        save_file=True,
        output_file="D1_gate_response.xdmf",
        petsc_options=None,
        debug=True,
    ):
        """
        Solve the linear gate-response equation on the parent/device mesh.

        Equation:
            div(eps grad D1) = 0

        Boundary conditions:
            D1 = 1 on rf_gate_name
            D1 = 0 on grounded_gate_names

        Returns
        -------
        D1 : dolfinx.fem.Function
            Parent-mesh gate response per volt.
        """

        if rf_gate_name not in self.device.facet_tags_map:
            raise KeyError(
                f"RF gate '{rf_gate_name}' not found. "
                f"Available boundaries: {list(self.device.facet_tags_map.keys())}"
            )

        if petsc_options is None:
            petsc_options = {
                "ksp_type": "cg",
                "pc_type": "gamg",
                "ksp_rtol": 1e-10,
                "ksp_atol": 1e-12,
                "ksp_max_it": 500,
            }

        if debug:
            self._rank0_print("\n" + "=" * 80)
            self._rank0_print("[D1] SOLVING GATE RESPONSE ON PARENT DEVICE MESH")
            self._rank0_print("=" * 80)
            self._mesh_bbox(self.domain, "D1 parent mesh")

        V = fem.functionspace(self.domain, ("Lagrange", 1))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        tdim = self.domain.topology.dim
        fdim = tdim - 1

        self.domain.topology.create_connectivity(fdim, tdim)
        self.domain.topology.create_connectivity(tdim, fdim)

        bcs = []

        if grounded_gate_names is None:
            grounded_gate_names = [
                name for name in self.device.facet_tags_map.keys()
                if name != rf_gate_name
            ]

        # RF gate: D1 = 1
        rf_tag = self.device.facet_tags_map[rf_gate_name]
        rf_facets = self.device.facet_tags.find(rf_tag)
        rf_dofs = fem.locate_dofs_topological(V, fdim, rf_facets)

        if len(rf_dofs) == 0:
            raise RuntimeError(
                f"No dofs found for RF gate '{rf_gate_name}'. "
                "Check facet tags and boundary markers."
            )

        bcs.append(
            fem.dirichletbc(
                PETSc.ScalarType(1.0),
                rf_dofs,
                V,
            )
        )

        self._rank0_print(
            f"[D1] RF gate {rf_gate_name}: 1.0 V, tag={rf_tag}, "
            f"facets={len(rf_facets)}, dofs={len(rf_dofs)}"
        )

        # Grounded gates/contacts: D1 = 0
        for gate_name in grounded_gate_names:
            if gate_name not in self.device.facet_tags_map:
                self._rank0_print(f"[D1] Skipping unknown boundary: {gate_name}")
                continue

            tag = self.device.facet_tags_map[gate_name]
            facets = self.device.facet_tags.find(tag)
            dofs = fem.locate_dofs_topological(V, fdim, facets)

            if len(dofs) == 0:
                self._rank0_print(
                    f"[D1] Boundary {gate_name}: tag={tag}, zero dofs. Skipping."
                )
                continue

            bcs.append(
                fem.dirichletbc(
                    PETSc.ScalarType(0.0),
                    dofs,
                    V,
                )
            )

            self._rank0_print(
                f"[D1] Grounded boundary {gate_name}: 0.0 V, tag={tag}, "
                f"facets={len(facets)}, dofs={len(dofs)}"
            )

        dx = ufl.dx(domain=self.domain)

        a = ufl.inner(self.eps * ufl.grad(u), ufl.grad(v)) * dx
        L = fem.Constant(self.domain, PETSc.ScalarType(0.0)) * ufl.conj(v) * dx

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="D1_solve_",
            petsc_options=petsc_options,
        )

        D1 = problem.solve()
        D1.name = f"D1_{rf_gate_name}_per_volt"
        D1.x.scatter_forward()

        self._print_function_stats(D1, "D1 parent result")

        if save_file:
            self._save_scalar_function(D1, output_file)

        return D1

    # -------------------------------------------------------------------------
    # Parent -> submesh restriction using vertex_map
    # -------------------------------------------------------------------------
    def restrict_parent_scalar_to_submesh(
        self,
        parent_function,
        submesh,
        vertex_map,
        name="D1_on_kp_submesh",
        debug=True,
        geometry_scale_for_debug=None,
        ncheck=10,
    ):
        """
        Restrict a CG1 scalar parent-mesh function to a CG1 scalar submesh function.

        This follows the same strategy as Device.submesh_parameters(...):

            submesh vertex -> parent vertex using vertex_map
            submesh dof    -> nearest submesh vertex using cKDTree
            parent value   -> submesh dof value

        This is topological transfer, not geometric interpolation.

        Parameters
        ----------
        parent_function : dolfinx.fem.Function
            Scalar CG1 function on the parent mesh, for example D1.

        submesh : dolfinx.mesh.Mesh
            k.p submesh used by eigenvectors.

        vertex_map : EntityMap
            vertex_map returned by device.create_submesh(...).

        geometry_scale_for_debug : float or None
            Only for printing coordinate mismatch.
            If parent is m and submesh is nm, use 1e9.
            If None, the code compares both 1 and 1e9 and reports the better one.

        Returns
        -------
        sub_function : dolfinx.fem.Function
            Scalar CG1 function on the k.p submesh.
        """

        parent_mesh = parent_function.function_space.mesh
        comm = submesh.comm

        if parent_mesh is submesh:
            if debug and comm.rank == 0:
                print("[RESTRICT] Parent and submesh are same object. No restriction needed.")
            return parent_function

        if debug:
            self._rank0_print("\n" + "=" * 80, comm)
            self._rank0_print("[RESTRICT] TRANSFERRING PARENT SCALAR FUNCTION TO KP SUBMESH", comm)
            self._rank0_print("=" * 80, comm)
            self._mesh_bbox(parent_mesh, "parent mesh")
            self._mesh_bbox(submesh, "kp submesh")
            self._print_function_stats(parent_function, "parent function before restriction")

        Vsub = fem.functionspace(submesh, ("Lagrange", 1))
        sub_function = fem.Function(Vsub)
        sub_function.name = name

        parent_function.x.scatter_forward()

        tdim = submesh.topology.dim

        parent_mesh.topology.create_connectivity(tdim, 0)
        parent_mesh.topology.create_connectivity(0, tdim)
        submesh.topology.create_connectivity(tdim, 0)
        submesh.topology.create_connectivity(0, tdim)

        imV_sub = submesh.topology.index_map(0)
        nV_sub_loc = imV_sub.size_local
        nV_sub_gh = imV_sub.num_ghosts
        nV_sub_tot = nV_sub_loc + nV_sub_gh

        imD_sub = Vsub.dofmap.index_map
        nD_sub_loc = imD_sub.size_local
        nD_sub_gh = imD_sub.num_ghosts
        nD_sub_tot = nD_sub_loc + nD_sub_gh

        if debug and comm.rank == 0:
            print(f"[RESTRICT] submesh vertices local={nV_sub_loc}, ghosts={nV_sub_gh}, total={nV_sub_tot}")
            print(f"[RESTRICT] submesh scalar dofs local={nD_sub_loc}, ghosts={nD_sub_gh}, total={nD_sub_tot}")
            print(f"[RESTRICT] parent scalar dofs local+ghost={len(parent_function.x.array)}")

        # Owned submesh vertices
        sub_vertices_owned = np.arange(nV_sub_loc, dtype=np.int32)

        # Map owned submesh vertices -> parent vertices
        parent_vertices = vertex_map.sub_topology_to_topology(
            sub_vertices_owned,
            inverse=False,
        ).astype(np.int32)

        x_sub = submesh.geometry.x
        x_parent = parent_mesh.geometry.x

        # Debug geometry mapping. If submesh was rescaled to nm, x_sub and x_parent
        # will not match directly, but topology mapping still works.
        if debug and len(sub_vertices_owned) > 0:
            gdim = submesh.geometry.dim

            x_sub_owned = x_sub[sub_vertices_owned, :gdim]
            x_parent_mapped = x_parent[parent_vertices, :gdim]

            if geometry_scale_for_debug is None:
                err_scale_1 = np.linalg.norm(x_sub_owned - x_parent_mapped, axis=1)
                err_scale_1e9 = np.linalg.norm(x_sub_owned - 1e9 * x_parent_mapped, axis=1)

                mean_1 = np.mean(err_scale_1)
                mean_1e9 = np.mean(err_scale_1e9)

                if mean_1e9 < mean_1:
                    scale_used = 1e9
                    err = err_scale_1e9
                else:
                    scale_used = 1.0
                    err = err_scale_1
            else:
                scale_used = geometry_scale_for_debug
                err = np.linalg.norm(x_sub_owned - scale_used * x_parent_mapped, axis=1)

            err_min = np.min(err)
            err_max = np.max(err)
            err_mean = np.mean(err)

            err_min_g = comm.allreduce(err_min, op=MPI.MIN)
            err_max_g = comm.allreduce(err_max, op=MPI.MAX)
            err_mean_g = comm.allreduce(err_mean, op=MPI.MAX)

            if comm.rank == 0:
                print("[RESTRICT] geometric mapping check:")
                print(f"[RESTRICT] debug coordinate scale parent -> sub = {scale_used:.3e}")
                print(
                    f"[RESTRICT] |x_sub - scale*x_parent(map)|: "
                    f"min={err_min_g:.3e}, max={err_max_g:.3e}, mean~={err_mean_g:.3e}"
                )

        # Map scalar P1 submesh dofs to nearest owned submesh vertices
        sub_dof_x = Vsub.tabulate_dof_coordinates()

        if nV_sub_loc == 0:
            raise RuntimeError("[RESTRICT] No owned submesh vertices on this rank.")

        tree = cKDTree(x_sub[sub_vertices_owned])
        dist, vlocal_idx = tree.query(sub_dof_x[:nD_sub_loc], k=1)

        dist_min = np.min(dist) if len(dist) > 0 else np.inf
        dist_max = np.max(dist) if len(dist) > 0 else -np.inf
        dist_mean = np.mean(dist) if len(dist) > 0 else np.inf

        dist_min_g = comm.allreduce(dist_min, op=MPI.MIN)
        dist_max_g = comm.allreduce(dist_max, op=MPI.MAX)
        dist_mean_g = comm.allreduce(dist_mean, op=MPI.MAX)

        if debug and comm.rank == 0:
            print("[RESTRICT] submesh dof -> nearest submesh vertex check:")
            print(
                f"[RESTRICT] distance: min={dist_min_g:.3e}, "
                f"max={dist_max_g:.3e}, mean~={dist_mean_g:.3e}"
            )

        sub_vertex_for_dof = sub_vertices_owned[vlocal_idx].astype(np.int32)

        parent_vertex_for_dof = vertex_map.sub_topology_to_topology(
            sub_vertex_for_dof,
            inverse=False,
        ).astype(np.int32)

        # Copy parent values. Do not scale D1 values.
        sub_function.x.array[:nD_sub_loc] = parent_function.x.array[parent_vertex_for_dof]
        sub_function.x.scatter_forward()

        if debug:
            vals_sub = np.real(sub_function.x.array[:nD_sub_loc])
            vals_parent = np.real(parent_function.x.array[parent_vertex_for_dof])

            dv = np.abs(vals_sub - vals_parent)
            dv_min = np.min(dv) if dv.size > 0 else np.inf
            dv_max = np.max(dv) if dv.size > 0 else -np.inf
            dv_mean = np.mean(dv) if dv.size > 0 else np.inf

            dv_min_g = comm.allreduce(dv_min, op=MPI.MIN)
            dv_max_g = comm.allreduce(dv_max, op=MPI.MAX)
            dv_mean_g = comm.allreduce(dv_mean, op=MPI.MAX)

            if comm.rank == 0:
                print("[RESTRICT] value-copy check:")
                print(
                    f"[RESTRICT] |sub - parent(mapped)|: "
                    f"min={dv_min_g:.3e}, max={dv_max_g:.3e}, mean~={dv_mean_g:.3e}"
                )

            self._print_function_stats(sub_function, "restricted submesh function")

            if comm.rank == 0:
                nprint = min(ncheck, nD_sub_loc)
                print("[RESTRICT] sample mapped values:")
                for k in range(nprint):
                    print(
                        f"  sub_dof={k:6d}, sub_vertex={sub_vertex_for_dof[k]:6d}, "
                        f"parent_vertex={parent_vertex_for_dof[k]:8d}, "
                        f"value={np.real(sub_function.x.array[k]): .6e}"
                    )

        return sub_function

    # -------------------------------------------------------------------------
    # Prepare D1 for kp eigenvector mesh
    # -------------------------------------------------------------------------
    def prepare_D1_for_kp_mesh(
        self,
        D1,
        psi_ref,
        vertex_map=None,
        debug=True,
    ):
        """
        Return D1 on the same mesh as psi_ref.

        Preferred path:
            use vertex_map from device.create_submesh(...)

        This is required when D1 is on the parent mesh and psi_ref is on
        the k.p submesh.
        """

        target_mesh = psi_ref.function_space.mesh
        source_mesh = D1.function_space.mesh

        if debug:
            self._rank0_print("\n" + "=" * 80, target_mesh.comm)
            self._rank0_print("[M01] PREPARING D1 FOR KP MESH", target_mesh.comm)
            self._rank0_print("=" * 80, target_mesh.comm)
            self._mesh_bbox(source_mesh, "D1 source mesh")
            self._mesh_bbox(target_mesh, "psi target mesh")
            self._print_function_stats(D1, "D1 source")

        if source_mesh is target_mesh:
            if debug and target_mesh.comm.rank == 0:
                print("[M01] D1 and eigenvectors are already on the same mesh.")
            return D1

        if vertex_map is None:
            raise RuntimeError(
                "D1 is not on the same mesh as the k.p eigenvectors, and no "
                "vertex_map was provided.\n\n"
                "Fix: pass the vertex_map returned by device.create_submesh(...):\n"
                "    M01 = rabisolver.M01(eigenvectors, D1, vertex_map=vertex_map)\n\n"
                "Do not use geometric interpolation here because your parent mesh is "
                "in meters and your k.p submesh has been rescaled to nm."
            )

        return self.restrict_parent_scalar_to_submesh(
            parent_function=D1,
            submesh=target_mesh,
            vertex_map=vertex_map,
            name=f"{D1.name}_on_kp_submesh" if D1.name else "D1_on_kp_submesh",
            debug=debug,
        )

    # -------------------------------------------------------------------------
    # Matrix element Mij
    # -------------------------------------------------------------------------
    def M01(
        self,
        eigen_vectors,
        D1,
        vertex_map=None,
        state_i=0,
        state_j=1,
        operator_sign=-1.0,
        debug=True,
    ):
        """
        Calculate the gate-transition matrix element:

            Mij = <psi_i | A_g | psi_j>

        where:

            A_g(r) = operator_sign * D1(r)

        Default:

            operator_sign = -1

        which corresponds to:

            A_g(r) = -D1(r)

        Parameters
        ----------
        eigen_vectors : list
            k.p eigenvectors as dolfinx fem.Function objects.

        D1 : dolfinx.fem.Function
            Parent-mesh gate-response potential from calculate_D1(...).

        vertex_map : EntityMap or None
            vertex_map returned by device.create_submesh(...).
            Required if D1 is on parent mesh and eigenvectors are on submesh.

        state_i : int
            Bra state index.

        state_j : int
            Ket state index.

        operator_sign : float
            -1.0 if Hamiltonian perturbation is -D1.
            +1.0 if Hamiltonian perturbation is +D1.

        Returns
        -------
        Mij : complex
            Matrix element.

            If Hamiltonian is in eV and potential enters as +/- phi,
            this is in eV/V.
        """

        psi_i = eigen_vectors[state_i]
        psi_j = eigen_vectors[state_j]

        kp_mesh = psi_i.function_space.mesh
        comm = kp_mesh.comm

        if psi_j.function_space.mesh is not kp_mesh:
            raise RuntimeError("[M01] psi_i and psi_j are not defined on the same mesh.")

        if debug:
            self._rank0_print("\n" + "=" * 80, comm)
            self._rank0_print("[M01] COMPUTING GATE MATRIX ELEMENT", comm)
            self._rank0_print("=" * 80, comm)
            print_i = f"state_i={state_i}"
            print_j = f"state_j={state_j}"
            self._rank0_print(f"[M01] {print_i}, {print_j}, operator_sign={operator_sign}", comm)

        D1_use = self.prepare_D1_for_kp_mesh(
            D1=D1,
            psi_ref=psi_i,
            vertex_map=vertex_map,
            debug=debug,
        )

        if D1_use.function_space.mesh is not kp_mesh:
            raise RuntimeError(
                "[M01] D1 and eigenvectors are still not on the same mesh after preparation."
            )

        if debug:
            self._print_function_stats(D1_use, "D1 used in M01")

            psi_i_min, psi_i_max = self._global_minmax_array(psi_i.x.array, comm)
            psi_j_min, psi_j_max = self._global_minmax_array(psi_j.x.array, comm)

            if comm.rank == 0:
                print(f"[M01] psi_i real min/max = {psi_i_min:.6e}, {psi_i_max:.6e}")
                print(f"[M01] psi_j real min/max = {psi_j_min:.6e}, {psi_j_max:.6e}")
                print(f"[M01] psi_i local dofs = {len(psi_i.x.array)}")
                print(f"[M01] psi_j local dofs = {len(psi_j.x.array)}")
                print(f"[M01] D1 local dofs    = {len(D1_use.x.array)}")

        dx = ufl.dx(domain=kp_mesh)

        # UFL complex convention:
        #     inner(a, b) = sum a_k * conj(b_k)
        #
        # Therefore:
        #     inner(psi_j, psi_i) = sum psi_j * conj(psi_i)
        #
        # This gives:
        #     psi_i^dagger psi_j
        #
        # so:
        #     operator_sign * D1 * inner(psi_j, psi_i)
        #
        # corresponds to:
        #     <psi_i | operator_sign * D1 | psi_j>
        integrand = PETSc.ScalarType(operator_sign) * D1_use * ufl.inner(psi_j, psi_i)

        form = fem.form(integrand * dx)

        Mij_local = fem.assemble_scalar(form)

        Mij_global = comm.allreduce(
            Mij_local,
            op=MPI.SUM,
        )

        Mij_global = np.complex128(Mij_global)

        if debug and comm.rank == 0:
            print(f"[M01] Mij local  = {Mij_local}")
            print(f"[M01] Mij global = {Mij_global}")
            print(f"[M01] |Mij|      = {abs(Mij_global):.6e}")

        return Mij_global

    # -------------------------------------------------------------------------
    # Rabi helper
    # -------------------------------------------------------------------------
    def rabi_frequency(
        self,
        eigen_vectors,
        D1,
        Vac,
        vertex_map=None,
        state_i=0,
        state_j=1,
        operator_sign=-1.0,
        units="eV",
        debug=True,
    ):
        """
        Compute Rabi frequency:

            f_R = |Vac * Mij| / h

        Parameters
        ----------
        Vac : float
            Peak RF voltage amplitude in volts.

        units : str
            "eV" if Mij is in eV/V.
            "J" if Mij is in J/V.
        """

        Mij = self.M01(
            eigen_vectors=eigen_vectors,
            D1=D1,
            vertex_map=vertex_map,
            state_i=state_i,
            state_j=state_j,
            operator_sign=operator_sign,
            debug=debug,
        )

        Hac = Vac * Mij

        if units.lower() == "ev":
            h = 4.135667696e-15  # eV*s
        elif units.lower() == "j":
            h = 6.62607015e-34  # J*s
        else:
            raise ValueError("units must be 'eV' or 'J'.")

        f_Rabi_Hz = abs(Hac) / h
        f_Rabi_MHz = f_Rabi_Hz / 1e6

        if debug:
            comm = eigen_vectors[state_i].function_space.mesh.comm
            if comm.rank == 0:
                print("\n" + "=" * 80)
                print("[RABI] RESULT")
                print("=" * 80)
                print(f"[RABI] Vac peak       = {Vac:.6e} V")
                print(f"[RABI] Mij            = {Mij}")
                print(f"[RABI] |Mij|          = {abs(Mij):.6e} {units}/V")
                print(f"[RABI] Hac            = {Hac}")
                print(f"[RABI] |Hac|          = {abs(Hac):.6e} {units}")
                print(f"[RABI] f_Rabi         = {f_Rabi_Hz:.6e} Hz")
                print(f"[RABI] f_Rabi         = {f_Rabi_MHz:.6e} MHz")

        return {
            "Mij": Mij,
            "Hac": Hac,
            "f_Rabi_Hz": f_Rabi_Hz,
            "f_Rabi_MHz": f_Rabi_MHz,
        }


# Backward-compatible alias
rabi_solver = Rabi_solver