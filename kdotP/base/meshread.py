"""
===============================================================================
Project: kdotP 
Description: Mesh reading and preprocessing utilities using Gmsh and DolfinX

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

import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio


class MeshRead:
    """
    Base mesh reader.

    This class is intentionally kept focused on reading the Gmsh mesh and
    exposing the core mesh objects and physical-tag maps. Device-specific
    material and submesh utilities are added in the Device class.
    """

    def __init__(self, filename, verbose=False, log_writer=None):
        self.mesh_filename = filename
        self.verbose = verbose
        self.log_writer = log_writer

        self._msg(f"[MESH] Reading mesh file: {filename}")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.open(filename)

        # Extract physical group names for 3D regions
        self.volume_tags = {}
        for dim_tag, tag in gmsh.model.getPhysicalGroups(3):
            name = gmsh.model.getPhysicalName(dim_tag, tag)
            self.volume_tags[name] = tag
            self._msg(f"[MESH] Volume '{name}' has tag {tag}")

        # Extract physical group names for 2D facets/boundaries
        self.facet_tags_map = {}
        for dim_tag, tag in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(dim_tag, tag)
            self.facet_tags_map[name] = tag
            self._msg(f"[MESH] Facet '{name}' has tag {tag}")

        # Now import mesh
        result = gmshio.model_to_mesh(
            gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3
        )

        self.region_names = None
        self.submesh_region_map = {}

        self.submesh_cell_tags = None

        self.domain  = result[0]
        self.cell_tags  = result[1]
        self.facet_tags = result[2]
        
        gmsh.finalize()
        self.domain.geometry.x[:] *= 1e-9  # Convert um to m

        x = self.domain.geometry.x
        mins = x.min(axis=0)
        maxs = x.max(axis=0)
        extent = maxs - mins
        self._msg(f"[MESH] BBox min: {mins}, max: {maxs}, extent: {extent}")

    def set_logger(self, log_writer):
        self.log_writer = log_writer
        self._log(f"[INFO], filename: {self.mesh_filename}")

    def _log(self, message):
        if self.log_writer is not None:
            self.log_writer.write(message)

    def _msg(self, message):
        if self.domain.comm.rank == 0 if hasattr(self, 'domain') else True:
            if self.verbose:
                print(message)
        self._log(message)


# Backward-compatible alias
Mesh_read = MeshRead
