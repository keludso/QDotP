"""
===============================================================================
Project: kdotP 
Description: Device class handling material assignment, regions, and submesh extraction

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
import pyvista as pv
from dolfinx import fem, mesh, plot
from dolfinx.mesh import meshtags
from scipy.spatial import cKDTree

from .meshread import MeshRead


class Device(MeshRead):
    """
    Device class built on top of MeshRead.

    It keeps the original material-field and submesh utilities, while reusing
    the mesh-reading logic from the base class.
    """

    def __init__(self, filename, verbose=False, log_writer=None):
        super().__init__(filename, verbose=verbose, log_writer=log_writer)

        # Function space for material parameters on the parent mesh
        self.Q = fem.functionspace(self.domain, ("DG", 0))
        self.eps = fem.Function(self.Q)
        self.Eg = fem.Function(self.Q)
        self.Nc = fem.Function(self.Q)
        self.Nv = fem.Function(self.Q)
        self.chi = fem.Function(self.Q)
        self.gamma1 = fem.Function(self.Q)
        self.gamma2 = fem.Function(self.Q)
        self.gamma3 = fem.Function(self.Q)
        self.delta = fem.Function(self.Q)
        self.kappa = fem.Function(self.Q)
        self.Kane_Ep = fem.Function(self.Q)
        self.Me_x = fem.Function(self.Q)
        self.Me_y = fem.Function(self.Q)
        self.Me_z = fem.Function(self.Q)
        #deformation pot
        self.av = fem.Function(self.Q)
        self.bv = fem.Function(self.Q)
        self.dv = fem.Function(self.Q)

        # Submesh fields are populated later when create_submesh is called.
        self.eps_sub = None
        self.Eg_sub = None
        self.Nc_sub = None
        self.Nv_sub = None
        self.chi_sub = None
        self.gamma1_sub = None
        self.gamma2_sub = None
        self.gamma3_sub = None
        self.delta_sub = None
        self.kappa_sub = None
        self.Kane_Ep_sub = None
        self.Me_x_sub = None
        self.Me_y_sub = None
        self.Me_z_sub = None



        self.av_sub = None
        self.bv_sub = None
        self.dv_sub = None

    def set_materials(self, materials_map):
        """Assign material properties to each named physical region."""
        if self.domain.comm.rank == 0:
            print("\n" + "=" * 70)
            print("MATERIAL ASSIGNMENT")
            print("=" * 70)
            print(f"{'Region':<20} {'Tag':>4} {'Material':<15} {'Eg':>6} {'chi':>6}")
            print("-" * 70)

        self._log("[INFO], Material Assignment")
        self._log(f"{'Region':<20} {'Tag':>4} {'Material':<15} {'Eg':>6} {'chi':>6}")

        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, tdim)
        self.domain.topology.create_connectivity(tdim, 0)

        im = self.Q.dofmap.index_map
        n_local = im.size_local
        self.region_names = list(self.volume_tags.keys())

        for region_name, mat in materials_map.items():
            if region_name not in self.volume_tags:
                self._msg(f"[DEVICE] Region '{region_name}' not in mesh")
                continue

            tag = self.volume_tags[region_name]
            cells = self.cell_tags.find(tag)
            if len(cells) == 0:
                continue

            dofs = fem.locate_dofs_topological(self.Q, tdim, cells)
            dofs = dofs[dofs < n_local]

            self.eps.x.array[dofs] = mat.eps
            self.Eg.x.array[dofs] = mat.Eg
            self.Nc.x.array[dofs] = mat.Nc
            self.Nv.x.array[dofs] = mat.Nv
            self.chi.x.array[dofs] = mat.chi
            self.gamma1.x.array[dofs] = mat.gamma[0]
            self.gamma2.x.array[dofs] = mat.gamma[1]
            self.gamma3.x.array[dofs] = mat.gamma[2]
            self.delta.x.array[dofs] = mat.delta
            self.kappa.x.array[dofs] = mat.kappa
            self.Kane_Ep.x.array[dofs] = mat.Kane_Ep
            self.Me_x.x.array[dofs] = mat.mc[0]
            self.Me_y.x.array[dofs] = mat.mc[1]
            self.Me_z.x.array[dofs] = mat.mc[2]


            self.av.x.array[dofs] = mat.deformation_pot[0]
            self.bv.x.array[dofs] = mat.deformation_pot[1]
            self.dv.x.array[dofs] = mat.deformation_pot[2]

            line = f"{region_name:<20} {tag:>4} {mat.name:<15} {mat.Eg:>6.2f} {mat.chi:>6.2f}"
            if self.domain.comm.rank == 0:
                print(line)
            self._log(line)

        if self.domain.comm.rank == 0:
            print("=" * 70 + "\n")

    def _build_submesh_functions(self, subdomain):
        """Create all P1 material functions on the submesh."""
        Q_sub = fem.functionspace(subdomain, ("DG", 0))
        self.eps_sub = fem.Function(Q_sub)
        self.Eg_sub = fem.Function(Q_sub)
        self.Nc_sub = fem.Function(Q_sub)
        self.Nv_sub = fem.Function(Q_sub)
        self.chi_sub = fem.Function(Q_sub)
        self.gamma1_sub = fem.Function(Q_sub)
        self.gamma2_sub = fem.Function(Q_sub)
        self.gamma3_sub = fem.Function(Q_sub)
        self.delta_sub = fem.Function(Q_sub)
        self.kappa_sub = fem.Function(Q_sub)
        self.Kane_Ep_sub = fem.Function(Q_sub)
        self.Me_x_sub = fem.Function(Q_sub)
        self.Me_y_sub = fem.Function(Q_sub)
        self.Me_z_sub = fem.Function(Q_sub)

        self.av_sub = fem.Function(Q_sub)
        self.bv_sub = fem.Function(Q_sub)
        self.dv_sub = fem.Function(Q_sub)

        return Q_sub

    def create_submesh(self, region_names, material_map, verbose=None):
        """
        Create a submesh from one or more named regions and assign submesh
        material functions.
        """
        if verbose is None:
            verbose = self.verbose

        tdim = self.domain.topology.dim
        if isinstance(region_names, str):
            region_names = [region_names]

        all_cells = []
        region_info = []
        for region_name in region_names:
            if region_name not in self.volume_tags:
                raise ValueError(
                    f"Region '{region_name}' not found. Available: {list(self.volume_tags.keys())}"
                )
            tag = self.volume_tags[region_name]
            cells = self.cell_tags.find(tag)
            all_cells.append(cells)
            region_info.append((region_name, tag, len(cells)))

        combined_cells = np.concatenate(all_cells)
        unique_cells = np.unique(combined_cells)
        self._msg(f"[DEVICE] combined_cells tags: {combined_cells}")
        self._msg(f"[DEVICE] unique cell ids: {unique_cells}")

        subdomain, entity_map, vertex_map, geom_map = mesh.create_submesh(
            self.domain, tdim, unique_cells
        )

        parent_cells = entity_map.sub_topology_to_topology(
            np.arange(subdomain.topology.index_map(tdim).size_local, dtype=np.int32),
            inverse=False,
        )
        vals = self.cell_tags.values[parent_cells].astype(np.int32)
        self.submesh_cell_tags = meshtags(
            subdomain,
            tdim,
            entities=np.arange(len(vals), dtype=np.int32),
            values=vals,
        )

        if subdomain.comm.rank == 0 and verbose:
            print("Submesh cell tags object:", self.submesh_cell_tags)
            print("Number of submesh cells:", len(self.submesh_cell_tags.values))

        Q_sub = self._build_submesh_functions(subdomain)
        self.submesh_region_map = {}

        # Create needed connectivity only once.
        subdomain.topology.create_connectivity(tdim, 0)
        subdomain.topology.create_connectivity(tdim, tdim)
        Q_sub.mesh.topology.create_connectivity(0, tdim)

        for region_name, mat in material_map.items():
            if region_name not in self.volume_tags:
                self._msg(f"[DEVICE] Region '{region_name}' not in mesh")
                continue

            tag = self.volume_tags[region_name]
            cells = self.submesh_cell_tags.find(tag)
            if len(cells) == 0:
                continue

            dofs = fem.locate_dofs_topological(Q_sub, tdim, cells)
            im = Q_sub.dofmap.index_map
            dofs = dofs[dofs < im.size_local]

            self.eps_sub.x.array[dofs] = mat.eps
            self.Eg_sub.x.array[dofs] = mat.Eg
            self.Nc_sub.x.array[dofs] = mat.Nc
            self.Nv_sub.x.array[dofs] = mat.Nv
            self.chi_sub.x.array[dofs] = mat.chi
            self.gamma1_sub.x.array[dofs] = mat.gamma[0]
            self.gamma2_sub.x.array[dofs] = mat.gamma[1]
            self.gamma3_sub.x.array[dofs] = mat.gamma[2]
            self.delta_sub.x.array[dofs] = mat.delta
            self.kappa_sub.x.array[dofs] = mat.kappa
            self.av_sub.x.array[dofs] = mat.deformation_pot[0]
            self.bv_sub.x.array[dofs] = mat.deformation_pot[1]
            self.dv_sub.x.array[dofs] = mat.deformation_pot[2]
            self.Kane_Ep_sub.x.array[dofs] = mat.Kane_Ep
      
            self.Me_x_sub = mat.mc[0]
            self.Me_y_sub = mat.mc[1]
            self.Me_z_sub = mat.mc[2]
            
            self.submesh_region_map[region_name] = dofs

            if subdomain.comm.rank == 0 and verbose:
                print(f"{region_name:<20} {tag:>4} {mat.name:<15} {mat.Eg:>6.2f} {mat.chi:>6.2f}")

        if verbose and self.domain.comm.rank == 0:
            print("\nSubmesh regions:")
            total_cells = 0
            for name, tag, num_cells in region_info:
                print(f"  {name:<25} {tag:>5} {num_cells:>10} cells")
                total_cells += num_cells
            print(f"  Total: {total_cells} cells -> {len(unique_cells)} submesh cells\n")

        return subdomain, vertex_map

    def submesh_parameters(self, phi_parent, submesh, vertex_map, ncheck=10):
        """
        Restrict a CG1 parent-mesh function to the submesh.

        The implementation is kept close to your original code, but the debug
        prints are controlled by the verbose flag.
        """
        parent_mesh = self.domain
        Vsub = fem.functionspace(submesh, ("Lagrange", 1))
        phi_sub = fem.Function(Vsub)
        phi_parent.x.scatter_forward()

        imV_sub = submesh.topology.index_map(0)
        nV_sub_loc = imV_sub.size_local
        nV_sub_gh = imV_sub.num_ghosts
        nV_sub_tot = nV_sub_loc + nV_sub_gh

        imD_sub = Vsub.dofmap.index_map
        nD_sub_loc = imD_sub.size_local

        sub_vertices_owned = np.arange(nV_sub_loc, dtype=np.int32)
        parent_vertices = vertex_map.sub_topology_to_topology(
            sub_vertices_owned, inverse=False
        ).astype(np.int32)

        x_sub = submesh.geometry.x
        x_par = parent_mesh.geometry.x

        dx = x_sub[sub_vertices_owned] - x_par[parent_vertices]
        err = np.linalg.norm(dx, axis=1)
        if submesh.comm.rank == 0 and self.verbose:
            print("Geometric mapping check (owned sub vertices):")
            print(
                f"  coord error |x_sub - x_par(map)|: "
                f"min={err.min():.3e}, max={err.max():.3e}, mean={err.mean():.3e}"
            )

        sub_dof_x = Vsub.tabulate_dof_coordinates()
        tree = cKDTree(x_sub[sub_vertices_owned])
        dist, vlocal_idx = tree.query(sub_dof_x[:nD_sub_loc], k=1)

        sub_vertex_for_dof = sub_vertices_owned[vlocal_idx].astype(np.int32)
        parent_vertex_for_dof = vertex_map.sub_topology_to_topology(
            sub_vertex_for_dof, inverse=False
        ).astype(np.int32)

        phi_sub.x.array[:nD_sub_loc] = phi_parent.x.array[parent_vertex_for_dof]
        phi_sub.x.scatter_forward()

        if submesh.comm.rank == 0 and self.verbose:
            vals_sub = np.real(phi_sub.x.array[:nD_sub_loc])
            vals_par = np.real(phi_parent.x.array[parent_vertex_for_dof])
            dv = np.abs(vals_sub - vals_par)
            print(
                f"[DEVICE] submesh restriction check: "
                f"min={dv.min():.3e}, max={dv.max():.3e}, mean={dv.mean():.3e}"
            )

        topology, cell_types, geometry = plot.vtk_mesh(submesh, submesh.topology.dim)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        if submesh.comm.rank == 0:
            grid.point_data["phi_sub"] = np.real(phi_sub.x.array)

        return phi_sub


# Backward-compatible aliases
class device(Device):
    pass
