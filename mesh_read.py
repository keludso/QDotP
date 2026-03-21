import ufl
import math
import numpy as np
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx import fem, mesh, io, nls, log, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.mesh import meshtags
from scipy.spatial import cKDTree
import pyvista as pv



# Reads the Mesh and returns domain, cell tags and facet tags 

class Mesh_read:
    def __init__(self, filename):
        self.mesh_filename = filename


        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.open(filename)
        
        # ============= EXTRACT PHYSICAL NAME â†’ TAG MAPPING =============
        # Get all physical groups (3D volumes for cells)
        dim = 3  # For 3D volumes (cells)
        physical_groups = gmsh.model.getPhysicalGroups(dim)

        # Create dictionary: name â†’ tag
        self.volume_tags = {}
        for dim_tag, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim_tag, tag)
            self.volume_tags[name] = tag
            print(f"Volume '{name}' has tag {tag}")

        # Also for facets (2D surfaces for boundary conditions)
        dim_facet = 2
        facet_groups = gmsh.model.getPhysicalGroups(dim_facet)

        self.facet_tags_map = {}
        for dim_tag, tag in facet_groups:
            name = gmsh.model.getPhysicalName(dim_tag, tag)
            self.facet_tags_map[name] = tag
            print(f"Facet '{name}' has tag {tag}")

        # Now import mesh
        result = gmshio.model_to_mesh(
            gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3
        )

        self.domain  = result[0]
        self.cell_tags  = result[1]
        self.facet_tags = result[2]
        
        gmsh.finalize()
        self.domain.geometry.x[:] *= 1e-6  # Convert um to m

        X = self.domain.geometry.x
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        extent = maxs - mins
        if self.domain.comm.rank == 0:
            print("BBox min:", mins, "max:", maxs, "extent:", extent)


        # Function space to define material parameters 
        
        self.Q = fem.functionspace(self.domain, ('Lagrange', 1))  
        self.eps = fem.Function(self.Q)
        self.Eg = fem.Function(self.Q)
        self.Nc = fem.Function(self.Q)
        self.Nv = fem.Function(self.Q)
        self.chi = fem.Function(self.Q)
        self.gamma1 = fem.Function(self.Q)
        self.gamma2 = fem.Function(self.Q)
        self.gamma3 = fem.Function(self.Q)
        self.delta = fem.Function(self.Q)


    # Setting region for with different materials properties

    def set_materials(self, materials_map):
        print("\n" + "="*70)
        print("MATERIAL ASSIGNMENT")
        print("="*70)
        print(f"{'Region':<20} {'Tag':>4} {'Material':<15} {'Eg':>6} {'chi':>6}")
        print("-"*70)

        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, tdim)  # needed sometimes
        self.domain.topology.create_connectivity(tdim, 0)     # cell -> vertex

        Q = self.eps.function_space  # your P1 space

        # owned dofs only (MPI safe)
        im = Q.dofmap.index_map
        n_local = im.size_local

        for region_name, mat in materials_map.items():
            if region_name not in self.volume_tags:
                print(f"Region '{region_name}' not in mesh!")
                continue

            tag = self.volume_tags[region_name]
            cells = self.cell_tags.find(tag)
            if len(cells) == 0:
                continue

            dofs = fem.locate_dofs_topological(Q, tdim, cells)
            dofs = dofs[dofs < n_local]

            self.eps.x.array[dofs]    = mat.eps
            self.Eg.x.array[dofs]     = mat.Eg
            self.Nc.x.array[dofs]     = mat.Nc
            self.Nv.x.array[dofs]     = mat.Nv
            self.chi.x.array[dofs]    = mat.chi
            self.gamma1.x.array[dofs] = mat.gamma[0]
            self.gamma2.x.array[dofs] = mat.gamma[1]
            self.gamma3.x.array[dofs] = mat.gamma[2]
            self.delta.x.array[dofs]  = mat.delta

            print(f"{region_name:<20} {tag:>4} {mat.name:<15} {mat.Eg:>6.2f} {mat.chi:>6.2f}")

        print("="*70 + "\n")

    

    def create_submesh(self, region_names, material_map, verbose: bool = True):
        """
        Create submesh from one or more regions.
        
        Parameters
        ----------
        region_names : str or list of str
            Single region name or list of region names to combine
        material_map : dict
            Material parameters for regions
        verbose : bool
            Print information
        
        Returns
        -------
        tuple : (submesh, vertex_map)
        """
        tdim = self.domain.topology.dim
        
        # Handle single region name
        if isinstance(region_names, str):
            region_names = [region_names]
        
        # Collect cells from all regions
        all_cells = []
        region_info = []
        
        for region_name in region_names:
            if region_name not in self.volume_tags:
                raise ValueError(f"Region '{region_name}' not found. "
                               f"Available: {list(self.volume_tags.keys())}")
            
            tag = self.volume_tags[region_name]
            cells = self.cell_tags.find(tag)
            all_cells.append(cells)
            region_info.append((region_name, tag, len(cells)))
        
        # Combine and remove duplicates
        combined_cells = np.concatenate(all_cells)
        unique_cells = np.unique(combined_cells)
        print("combined_cells tags:", combined_cells)
        print("Unique cell tags:", unique_cells)

        # Create submesh
        subdomain, entity_map, vertex_map, geom_map = mesh.create_submesh(
            self.domain, tdim, unique_cells
        )

        # entity_map maps submesh cells -> parent mesh cells
        parent_cells = entity_map.sub_topology_to_topology(np.arange(subdomain.topology.index_map(tdim).size_local, dtype=np.int32),
                                                        inverse=False)

        # tag values for those parent cells
        vals = self.cell_tags.values[parent_cells].astype(np.int32)

        submesh_cell_tags = meshtags(subdomain, tdim,
                                    entities=np.arange(len(vals), dtype=np.int32),
                                    values=vals)
        
        if subdomain.comm.rank == 0:
            print("Submesh cell tags object:", submesh_cell_tags)
            print("Number of submesh cells:", len(submesh_cell_tags.values))

        topology, cell_types, geometry = plot.vtk_mesh(subdomain, subdomain.topology.dim)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        grid.cell_data["region_tag"] = submesh_cell_tags.values

        grid.save("submesh_regions.vtu")

        # Create function spaces for material parameters on submesh
        Q = fem.functionspace(subdomain, ('Lagrange', 1))  
        self.eps_sub = fem.Function(Q)
        self.Eg_sub = fem.Function(Q)
        self.Nc_sub = fem.Function(Q)
        self.Nv_sub = fem.Function(Q)
        self.chi_sub = fem.Function(Q)
        self.gamma1_sub = fem.Function(Q)
        self.gamma2_sub = fem.Function(Q)
        self.gamma3_sub = fem.Function(Q)
        self.delta_sub = fem.Function(Q)

 
        for region_name, mat in material_map.items():
            if region_name not in self.volume_tags:
                print(f"Region '{region_name}' not in mesh!")
                continue

            tag = self.volume_tags[region_name]
            cells = submesh_cell_tags.find(tag)
            if len(cells) == 0:
                continue

            tdim = subdomain.topology.dim

            print("First command done")

            subdomain.topology.create_connectivity(tdim, 0)
            Q.mesh.topology.create_connectivity(0, tdim)      # vertex -> cell (sometimes needed)
            subdomain.topology.create_connectivity(tdim, tdim)
            print("topology create command done")
            tdim = Q.mesh.topology.dim
            imap = Q.mesh.topology.index_map(tdim)

            print("cells dtype:", cells.dtype)
            print("cells min/max:", int(cells.min()), int(cells.max()))
            print("n_local, n_ghost, n_tot:", imap.size_local, imap.num_ghosts, imap.size_local+imap.num_ghosts)
            print("Q.mesh id:", id(Q.mesh), "cells source mesh id:", id(self.domain))  # sanity

            dofs = fem.locate_dofs_topological(Q, tdim, cells)
            print("dofe command done")

            # MPI-safe: keep only owned dofs
            im = Q.dofmap.index_map
            dofs = dofs[dofs < im.size_local]
            
            if len(cells) > 0:
                self.eps_sub.x.array[dofs] = mat.eps
                self.Eg_sub.x.array[dofs] = mat.Eg
                self.Nc_sub.x.array[dofs] = mat.Nc
                self.Nv_sub.x.array[dofs] = mat.Nv
                self.chi_sub.x.array[dofs] = mat.chi
                self.gamma1_sub.x.array[dofs] = mat.gamma[0]
                self.gamma2_sub.x.array[dofs] = mat.gamma[1]
                self.gamma3_sub.x.array[dofs] = mat.gamma[2]
                self.delta_sub.x.array[dofs] = mat.delta
                
                if self.domain.comm.rank == 0:
                    print(f"{region_name:<20} {tag:>4} {mat.name:<15} {mat.Eg:>6.2f} {mat.chi:>6.2f}")

        # Print summary
        if verbose and self.domain.comm.rank == 0:
            print("\nSubmesh regions:")
            total_cells = 0
            for name, tag, num_cells in region_info:
                print(f"  {name:<25} {tag:>5} {num_cells:>10} cells")
                total_cells += num_cells
            print(f"  Total: {total_cells} cells -> {len(unique_cells)} submesh cells\n")

        if self.domain.comm.rank == 0:
            topology, cell_types, geometry = plot.vtk_mesh(Q)
            grid = pv.UnstructuredGrid(topology, cell_types, geometry)

            grid.point_data["chi"] = np.real(self.chi_sub.x.array)

            grid.save("submesh_check.vtu")    


        return subdomain, vertex_map



    def submesh_parameters(self,phi_parent, submesh, vertex_map, ncheck=10):
        """
        Restrict a CG1 (Lagrange P1) function from parent mesh to submesh,
        with heavy debug prints and geometric validation.

        Parameters
        ----------
        phi_parent : fem.Function on parent mesh (CG1)
        submesh : dolfinx.mesh.Mesh
        vertex_map : dolfinx.mesh.EntityMap  (submesh vertices -> parent vertices)
        parent_mesh : dolfinx.mesh.Mesh (phi_parent.mesh)
        ncheck : int, number of sample vertices to print
        """

        parent_mesh = self.domain
        Vsub = fem.functionspace(submesh, ("Lagrange", 1))
        phi_sub = fem.Function(Vsub)

        # Sync parent ghosts
        phi_parent.x.scatter_forward()

        # --- sizes ---
        imV_sub = submesh.topology.index_map(0)
        nV_sub_loc = imV_sub.size_local
        nV_sub_gh  = imV_sub.num_ghosts
        nV_sub_tot = nV_sub_loc + nV_sub_gh

        imV_par = parent_mesh.topology.index_map(0)
        nV_par_loc = imV_par.size_local
        nV_par_gh  = imV_par.num_ghosts
        nV_par_tot = nV_par_loc + nV_par_gh

        imD_sub = Vsub.dofmap.index_map
        nD_sub_loc = imD_sub.size_local
        nD_sub_gh  = imD_sub.num_ghosts

        if submesh.comm.rank == 0:
            print("\n" + "="*80)
            print("SUBMESH RESTRICTION DEBUG (P1)")
            print("="*80)
            print(f"Submesh vertices: local={nV_sub_loc}, ghosts={nV_sub_gh}, total={nV_sub_tot}")
            print(f"Parent  vertices: local={nV_par_loc}, ghosts={nV_par_gh}, total={nV_par_tot}")
            print(f"Submesh dofs(P1): local={nD_sub_loc}, ghosts={nD_sub_gh}")
            print("NOTE: For P1, dofs ~ vertices, but ordering may differ -> we must verify.")
            print("="*80)

        # --- mapping: submesh vertex indices -> parent vertex indices ---
        sub_vertices_owned = np.arange(nV_sub_loc, dtype=np.int32)
        parent_vertices = vertex_map.sub_topology_to_topology(sub_vertices_owned, inverse=False).astype(np.int32)

        # --- coords check ---
        x_sub = submesh.geometry.x
        x_par = parent_mesh.geometry.x

        # Make sure indices are in bounds (local+ghost arrays)
        assert parent_vertices.max() < x_par.shape[0], \
            f"Mapped parent vertex index out of bounds: max={parent_vertices.max()}, parent x size={x_par.shape[0]}"
        assert sub_vertices_owned.max() < x_sub.shape[0], \
            f"Sub vertex index out of bounds: max={sub_vertices_owned.max()}, sub x size={x_sub.shape[0]}"

        dx = x_sub[sub_vertices_owned] - x_par[parent_vertices]
        err = np.linalg.norm(dx, axis=1)

        if submesh.comm.rank == 0:
            print(f"Geometric mapping check (owned sub vertices):")
            print(f"  coord error |x_sub - x_par(map)|: min={err.min():.3e}, max={err.max():.3e}, mean={err.mean():.3e}")

            # show a few samples: evenly spaced
            idxs = np.linspace(0, nV_sub_loc-1, min(ncheck, nV_sub_loc), dtype=int)
            print("\nSample vertex map (sub_v -> par_v) with coordinates:")
            for i in idxs:
                sv = int(sub_vertices_owned[i])
                prv = int(parent_vertices[i])
                print(f"  sub_v {sv:8d} -> par_v {prv:8d} | "
                    f"x_sub={x_sub[sv]} | x_par={x_par[prv]} | err={err[i]:.3e}")

        # --- IMPORTANT: how to map DOFs correctly ---
        # For P1, the robust way is:
        # 1) get submesh dof coordinates
        # 2) map those dofs to nearest mapped parent vertices by coordinate match
        # But first we attempt the simplest vertex-based assignment:
        #
        # phi_sub at sub vertices should equal phi_parent at mapped parent vertices.
        #
        # HOWEVER: phi_sub.x.array is DOF-ordered, not necessarily vertex-ordered.
        # So we must build a dof->vertex correspondence via dof coordinates.
        #
        sub_dof_x = Vsub.tabulate_dof_coordinates()  # (ndofs_local, 3) in physical coords for CG1
        # For CG1, there is 1 dof per vertex, but ordering may differ.
        # Build KDTree from submesh vertex coords (owned) to map each DOF to a vertex index.
        
        tree = cKDTree(x_sub[sub_vertices_owned])
        dist, vlocal_idx = tree.query(sub_dof_x[:nD_sub_loc], k=1)

        if submesh.comm.rank == 0:
            print("\nDOF->vertex association (via coordinates):")
            print(f"  dof->vertex distance: min={dist.min():.3e}, max={dist.max():.3e}, mean={dist.mean():.3e}")
            if dist.max() > 1e-12:
                print("  WARNING: Some DOF coords do not match vertex coords tightly. Check mesh units / geometry.")

        # Now we know: sub DOF i corresponds to sub vertex sub_vertices_owned[vlocal_idx[i]]
        sub_vertex_for_dof = sub_vertices_owned[vlocal_idx].astype(np.int32)
        parent_vertex_for_dof = vertex_map.sub_topology_to_topology(sub_vertex_for_dof, inverse=False).astype(np.int32)

        # Assign using DOF mapping (correct even if ordering differs)
        phi_sub.x.array[:nD_sub_loc] = phi_parent.x.array[parent_vertex_for_dof]

        phi_sub.x.scatter_forward()

        # --- value check ---
        # Compare values at sampled DOFs
        if submesh.comm.rank == 0:
            vals_sub = np.real(phi_sub.x.array[:nD_sub_loc])
            vals_par = np.real(phi_parent.x.array[parent_vertex_for_dof])
            dv = np.abs(vals_sub - vals_par)
            print("\nValue mapping check (owned dofs):")
            print(f"  |phi_sub - phi_parent(map)|: min={dv.min():.3e}, max={dv.max():.3e}, mean={dv.mean():.3e}")

            idxs = np.linspace(0, nD_sub_loc-1, min(ncheck, nD_sub_loc), dtype=int)
            print("\nSample DOF map (sub_dof -> sub_v -> par_v) values:")
            for i in idxs:
                sd = int(i)
                sv = int(sub_vertex_for_dof[sd])
                prv = int(parent_vertex_for_dof[sd])
                print(f"  dof {sd:8d} -> sub_v {sv:8d} -> par_v {prv:8d} | "
                    f"phi_sub={vals_sub[sd]:.6e} phi_par={vals_par[sd]:.6e} diff={dv[sd]:.3e}")

        # --- write VTU for debugging in ParaView ---
        # IMPORTANT: use plot.vtk_mesh(submesh, ...) not vtk_mesh(Vsub)
        topology, cell_types, geometry = plot.vtk_mesh(submesh, submesh.topology.dim)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        # point_data must match #points in grid
        # grid.n_points should equal submesh.geometry.x rows (local+ghost in serial)
        # In serial, OK. In MPI, only rank0 writing is typical.
        if submesh.comm.rank == 0:
            grid.point_data["phi_sub"] = np.real(phi_sub.x.array)
            # also store an "expected" parent-mapped value at sub vertices (for comparison)
            # Build vertex-wise expected values:
            par_v_all = vertex_map.sub_topology_to_topology(
                np.arange(submesh.topology.index_map(0).size_local, dtype=np.int32),
                inverse=False
            )
            expected = np.zeros_like(np.real(phi_sub.x.array))
            expected[:nV_sub_loc] = np.real(phi_parent.x.array[par_v_all])
            grid.point_data["phi_parent_mapped_vertex"] = expected
            grid.point_data["phi_diff_vertex"] = grid.point_data["phi_sub"] - grid.point_data["phi_parent_mapped_vertex"]

            grid.save("phi_sub_debug.vtu")
            print("\nSaved: phi_sub_debug.vtu (phi_sub, phi_parent_mapped_vertex, phi_diff_vertex)")

        return phi_sub
