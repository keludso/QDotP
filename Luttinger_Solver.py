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


class LK_hamiltonian_4band:

    def __init__(self,device,phi,chi, Eg, gamma1, gamma2, gamma3,delta):
        self.domain = device
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

        # Rearragning the mesh to nm for easy solving 
        self.domain.geometry.x[:] *= 1e9  # Convert m to nm
        X = self.domain.geometry.x
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        extent = maxs - mins
        if self.domain.comm.rank == 0:
            print("BBox min:", mins, "max:", maxs, "extent:", extent)


        self.dx = ufl.Measure("dx", domain=self.domain)

        self.phi= phi
        self.chi = chi
        self.Eg = Eg

        
        print(f"phi min: {self.phi.x.array.min():.6e}")
        print(f"phi max: {self.phi.x.array.max():.6e}")
        print(f"chi min: {self.chi.x.array.min():.6e}")

        self.h2m = 0.0381     #eV/um#0.07619902 #hbar**2/(2*m0)/charge

        print(self.h2m)
        self.delta = delta
        #self.delta.x.array[:] *= 1e18
        
        V = fem.functionspace(device, ("Lagrange", 1))

        chii_P1 = chi
        Eg_P1 = Eg


        self.Ec_cal = fem.Function(V)  # Conduction band edge [eV]
        self.Ev_cal = fem.Function(V)  # Valence band edge [eV]

        # Calculate band edges - now all arrays have the same shape!
        phiref = 2.7  # [eV]
        self.Ec_cal.x.array[:] = phi.x.array[:] + Eg_P1.x.array[:]/2 # Use chii_P1, not chii
        self.Ev_cal.x.array[:] = self.Ec_cal.x.array[:] - Eg_P1.x.array[:] +0.4
        print(f"Ev max: {self.Ev_cal.x.array.max():.6e}")          # Use Eg_P1, not Eg

        self.Ev_cal.x.scatter_forward() 

        # Build VTK topology/geometry from the mesh, not from Q
        topology, cell_types, geometry = plot.vtk_mesh(V)

        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        # DG0 ? one value per cell ? use cell_data, not point_data
        grid.point_data["Ev"] = np.real(self.Ev_cal.x.array)
        grid.point_data["Ec"] = np.real(self.Ec_cal.x.array)
        grid.point_data["phi"] = np.real(self.phi.x.array)

        grid.save("Ev_parabolic.vtu")


    def sym_d(self, u, v, i, j):
        return 0.5 * (ufl.inner(ufl.grad(u)[i] , ufl.grad(v)[j]) + ufl.inner(ufl.grad(u)[j], ufl.grad(v)[i]))

    def P_operator(self, u, v):
        return self.h2m * self.gamma1 * (ufl.inner(ufl.grad(u)[0] , ufl.grad(v)[0])  + ufl.inner(ufl.grad(u)[1] , ufl.grad(v)[1])  + ufl.inner(ufl.grad(u)[2] , ufl.grad(v)[2]))

    def Q_operator(self, u, v):
        return self.h2m * self.gamma2 * (ufl.inner(ufl.grad(u)[0] , ufl.grad(v)[0]) + ufl.inner(ufl.grad(u)[1] , ufl.grad(v)[1]) - 2 * ufl.inner(ufl.grad(u)[2] , ufl.grad(v)[2])) 

    def R_operator_real(self, u, v):
        return math.sqrt(3) * self.h2m * (self.gamma2) * (-ufl.inner(ufl.grad(u)[0], ufl.grad(v)[0]) + ufl.inner(ufl.grad(u)[1], ufl.grad(v)[1])) 

    def R_operator_imag(self, u, v):
        kxky_sym = self.sym_d(u, v, 0, 1)
        return 2*1j * math.sqrt(3) * self.h2m * self.gamma3 * kxky_sym 

    def Splus_operator_real(self, u, v):
        kxkz_sym = self.sym_d(u, v, 0, 2)
        return 2 * math.sqrt(3) * self.h2m * self.gamma3 * kxkz_sym

    def Splus_operator_imag(self, u, v):
        kykz_sym = self.sym_d(u, v, 1, 2)
        return 2j*math.sqrt(3)*self.h2m*self.gamma3*(kykz_sym)
    
    
    def Sneg_operator_real(self, u, v):
        kxkz_sym = self.sym_d(u, v, 0, 2)
        return 2 * math.sqrt(3) * self.h2m * self.gamma3 * kxkz_sym 

    def Sneg_operator_imag(self, u, v):
        kykz_sym = self.sym_d(u, v, 1, 2)

        return -2j*math.sqrt(3)*self.h2m*self.gamma3*(kykz_sym)
    

    def assemble_components(self, u1, u2, u3, u4, v1, v2, v3, v4):
        # Apply conjugation to all test functions for complex mode
        v1c = v1
        v2c = v2
        v3c = v3
        v4c = v4

        # 1st row
        H11 = self.P_operator(u1, v1c)*self.dx + self.Q_operator(u1, v1c)*self.dx  #P+Q
        H13 = -1* (self.Sneg_operator_real(u1, v3c)*self.dx + self.Sneg_operator_imag(u1, v3c)*self.dx)   #-S-
        H14 = (self.R_operator_real(u1, v4c)*self.dx + self.R_operator_imag(u1, v4c)*self.dx)  #R

        # 2nd row 
        H22 = self.P_operator(u2, v2c)*self.dx + self.Q_operator(u2, v2c)*self.dx   #P+Q
        H23 = -1*(self.R_operator_real(u2, v3c)*self.dx - self.R_operator_imag(u2, v3c)*self.dx)   #-R*
        H24 = -1* (self.Splus_operator_real(u2, v4c)*self.dx + self.Splus_operator_imag(u2, v4c)*self.dx)   #-Splus

        # 3rd row
        H31 = -1* (self.Sneg_operator_real(u3, v1c)*self.dx - self.Sneg_operator_imag(u3, v1c)*self.dx)   #-S*-
        H32 = -1*(self.R_operator_real(u3, v2c)*self.dx + self.R_operator_imag(u3, v2c)*self.dx)   #-R
        H33 = self.P_operator(u3, v3c)*self.dx - self.Q_operator(u3, v3c)*self.dx

        # 4th row
        
        H41 = (self.R_operator_real(u4, v1c)*self.dx - self.R_operator_imag(u4, v1c)*self.dx)  #R*
        H42 = -1* (self.Splus_operator_real(u4, v2c)*self.dx - self.Splus_operator_imag(u4, v2c)*self.dx)   #-Splus*
        H44 = self.P_operator(u4, v4c)*self.dx - self.Q_operator(u4, v4c)*self.dx



        if self.phi is not None:


            print(f"Ev min: {self.Ev_cal.x.array.min():.6e}")
            print(f"Ev max: {self.Ev_cal.x.array.max():.6e}")


            H11 = H11 - self.Ev_cal * ufl.inner(u1, v1) * self.dx
            H22 = H22 - self.Ev_cal * ufl.inner(u2, v2) * self.dx
            H33 = H33 - self.Ev_cal * ufl.inner(u3, v3) * self.dx
            H44 = H44 - self.Ev_cal * ufl.inner(u4, v4) * self.dx



        # NO band edge potential - kinetic energy only
        a = (H11 +  H13 + H14+
              H22 +  H23+ H24 +
              H31 +  H32 + H33 + 
              H41 + H42 +  H44 )
        
        return a

    def create_mass_matrix(self, u1, u2, u3, u4, v1, v2, v3, v4):

        b = (ufl.inner(u1,v1) * self.dx + 
             ufl.inner(u2,v2) * self.dx + 
             ufl.inner(u3,v3) * self.dx + 
             ufl.inner(u4,v4) * self.dx )
        return b
    

    def solve_LK(self):
        scalar_type = PETSc.ScalarType  # complex128 if PETSc built with complex
        P1 = element("Lagrange", self.domain.basix_cell(), 1)
        mixed_elem = basix.ufl.mixed_element([P1, P1, P1, P1])
        self.V = fem.functionspace(self.domain, mixed_elem)

        u1, u2, u3, u4 = ufl.TrialFunction(self.V)
        v1, v2, v3, v4 = ufl.TestFunction(self.V)

        a = self.assemble_components(u1, u2, u3, u4, v1, v2, v3, v4)

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

        b = self.create_mass_matrix(u1, u2, u3, u4, v1, v2, v3, v4)
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
        eps.setDimensions(nev=nev, ncv=max(40, 2*nev+20))
        eps.setTolerances(tol=1e-8, max_it=500)

        # Step 2: Configure ST AFTER setOperators
        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(target+0.005)                                       #  use target directly, no offset

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
            print(f"i={i}  E={eigenval:.6e}  rel={rel:.3e}  abs={absr:.3e}")

            # Create a NEW Function each iteration — reusing the same object
            # means all entries in eigenvectors point to the same data
            eigenfunction = fem.Function(self.V)
            eps.getEigenpair(i, eigenfunction.x.petsc_vec)
            eigenfunction.x.scatter_forward()

            eigenvectors.append(eigenfunction)

        # Print results
        print("\nEigenvalues (eV):")
        for i, E in enumerate(eigenvalues):
            print(f"  E_{i+1} = {E:.6f}")

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
            real_part = np.real(vals)
            imag_part = np.imag(vals)

            print(f"  {name}: maxsi={prob.max():.3e}")
            grid.point_data[f"{name}_abs2"] = prob

        grid.point_data["total_prob"] = total_prob
        grid.save(output_file)
        print(f"  Saved to {output_file}")

