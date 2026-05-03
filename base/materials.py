"""
Material database for semiconductor simulations.
Usage: 
    from materials import Ge, SiGe70
    eps = Ge.eps
    Eg = Ge.Eg
"""

# ============================================================================
# CONSTANTS
# ============================================================================

class Constants:
    """Physical constants."""
    q = 1.6e-19                   # C
    hbar = 6.582119569e-16        # eV·s
    m0 = 9.1093837015e-31         # kg
    kbT = 0.0259                  # eV at 300K

CONST = Constants()


# ============================================================================
# MATERIAL CLASS
# ============================================================================

class Material:
    """Material with properties accessible as attributes."""
    
    def __init__(self, name, eps, Eg, Nc, Nv, chi, mc, mv,gamma,delta,kappa,magq,deformation_pot,Kane_Ep):
        """
        Create material.
        
        Parameters
        ----------
        name : str
            Material name
        eps : float
            Relative permittivity
        Eg : float
            Bandgap [eV]
        Nc : float
            Conduction band DOS [cm^-3]
        Nv : float
            Valence band DOS [cm^-3]
        chi : float
            Electron affinity [eV]
        m_e : float
            Electron effective mass [m0]
        m_lh : float
            Light hole mass [m0]
        m_hh : float
            Heavy hole mass [m0]
        """
        self.name = name
        self.eps = eps
        self.Eg = Eg
        self.Nc = Nc
        self.Nv = Nv
        self.chi = chi
        self.mc = mc
        self.mv = mv
        self.gamma = gamma
        self.delta = delta
        self.kappa = kappa
        self.magq = magq
        self.Kane_Ep = Kane_Ep

        #DEformation Potential 
        self.deformation_pot = deformation_pot

    
    def __repr__(self):
        return f"Material('{self.name}', Eg={self.Eg:.2f} eV, χ={self.chi:.2f} eV)"




# ============================================================================
# DEFINE ALL MATERIALS AS MODULE-LEVEL OBJECTS
# ============================================================================

# Pure semiconductors

eps0 = 8.8541878128e-12 #in um


Ge = Material("Ge",
    eps=16.5*eps0, Eg=0.6657, Nc = 2.5797e24, Nv = 5.3321e24,
    chi=4.0, mc=0.21941449,mv=[0.34624157,0.34624157,0.04230065,0.04230065],
    gamma=[13.380, 4.24, 5.69], delta = 0.2964,kappa =3.41, magq=0.06,deformation_pot=[2.0,-2.16,-6.06], Kane_Ep = 24.0)


Al2O3 = Material("Al2O3",
    eps=9.0*eps0, Eg=6.2, Nc=1.5e-2, Nv = 1.5e-2,
    chi=1.95, mc=0.2, mv=[0.78623495,0.78623495,0.1625266,0.1625266],
    gamma = [3.76 ,0.82,1.420], delta=0.3,kappa =3.41, magq=0.06,deformation_pot=[2.0,-2.16,-6.06], Kane_Ep = 3.0)


#SiGe80 = Material("SiGe80",
#    eps=15.56*eps0, Eg=0.8576, Nc = 4.5358e9, Nv = 9.1199e9,
 #   chi=4.01, mc=0.23972605,mv=[0.36933235,0.36933235,0.05108412,0.05108412],
 #   gamma=[11.5610, 3.4598, 4.8412], delta =0.246)
 
SiGe80 = Material("SiGe80",
    eps=15.56*eps0, Eg=0.8576, Nc = 2.9461e24, Nv = 5.9235e24,
    chi=4.01, mc=0.23972605,mv=[0.36933235,0.36933235,0.05108412,0.05108412],
    gamma=[11.5610, 3.4598, 4.8412], delta =0.246,kappa =3.41, magq=0.06,deformation_pot=[2.0,-2.16,-6.06], Kane_Ep = 21.0)
 

GeSn14 = Material("GeSn14",
    eps=15.56*eps0, Eg=0.8576, Nc = 2.9461e24, Nv = 5.9235e24,
    chi=4.01, mc=0.23972605,mv=[0.36933235,0.36933235,0.05108412,0.05108412],
    gamma=[11.5610, 3.4598, 4.8412], delta =0.246,kappa =3.41, magq=0.06,deformation_pot=[2.0,-2.16,-6.06], Kane_Ep = 21.0)
 


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def list_materials():
    """List all available pre-defined materials."""
    print("\nPre-defined materials:")
    print("-" * 50)
    for name, obj in globals().items():
        if isinstance(obj, Material):
            print(f"{obj.name:<15} Eg={obj.Eg:5.2f} eV  χ={obj.chi:5.2f} eV  εᵣ={obj.eps:5.1f}")
    print("-" * 50)
    print("\nCreate custom alloys with: create_sige_alloy(x_Ge)")


