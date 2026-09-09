// Gmsh project created on Fri Sep 19 09:12:55 2025
// 
//
SetFactory("OpenCASCADE");

// unit are in nm
// defining the width and height of the device 
D_x = 200;
D_y = 150;

barrier_width = 15;

plunger_width = 30;
// Define the Quantum submesh region

substrate_ht = 50;
BOX_ht = 15;
epi_Si = 15;
Thermal_SiO2 = 10;
Plunger_barrier_ht = 7;
cap_layer = 10;

// GAte thickness
accumulation_gate_thick = 30 ;
electrode_thick = 30;

Rectangle(1) = {-D_x/2, -D_y/2, 0, D_x, D_y, 0};
Rectangle(2) = {-D_x/4, -D_y/4, 0, D_x/2, D_y/2, 0};



// Making accumulation gates
// 
Rectangle(3) = {-D_x/4, -D_y/2, 0, barrier_width, D_y, 0};
Rectangle(4) = {D_x/4, -D_y/2, 0, -barrier_width, D_y, 0};



// Making plunger
// 
Rectangle(5) = {-plunger_width/2, -D_y/2, 0,plunger_width , D_y,0};

// Boolean fragments
BooleanFragments{ Surface{1:5}; Delete; }{ }


// Building the substrate 

Extrude {0, 0, BOX_ht} {Surface{1:17};}

Extrude {0, 0, epi_Si} {Surface{24,28,32,35,38,42,45,48,52,55,58,62,65,68,74,76,78};}

Extrude {0, 0, Thermal_SiO2} {Surface{85,89,93,96,99,103,106,109,113,116,119,123,126,129,135,137,139};}

Extrude {0, 0, Plunger_barrier_ht} {Surface{146,150,154,157,160,164,167,170,174,177,180,184,187,190,196,198,200};}

Extrude {0, 0, -substrate_ht} {Surface{1:17};}


//===============================================
// VOLUME GROUPS
//===============================================

Physical Volume("Si_substrate") = {69:71,73,74,76,77,79,80,82,83,85};
Physical Volume("Si_substrate_QD") = {72,75,78,81,84};

Physical Volume("BOX") = {1,2,3,5,6,8,9,11,12,14,15,17};
Physical Volume("BOX_QD") = {4,7,10,13,16};

Physical Volume("Epi_Si") = {18,19,20,22,23,25,26,28,29,31,32,34};
Physical Volume("Epi_Si_QD") = {21,24,27,30,33};

Physical Volume("Thermal_SiO2") = {35,36,37,39,40,42,43,45,46,48,49,51};            // Layer 2 volumes
Physical Volume("Thermal_SiO2_QD") = {38,41,44,47,50};            // Layer 2 volumes

Physical Volume("SiO2_cap") = {52:68};            // Layer 2 volumes





//===============================================
// SURFACE RELABELING WITH DIRECTIONAL GROUPING
//===============================================
// Bottom surfaces
Physical Surface("bottom_surface") = {268,272,276,279,282,286,289,292,296,299,302,306,309,312,318,320,322};

// Contact surfaces
Physical Surface("Barrier_left") = {150,157,160};
Physical Surface("Plunger") = {225,238,241};
Physical Surface("Barrier_right") = {184,198,200};





//===============================================
// MESH PARAMETERS
//===============================================
MeshSize{ PointsOf{ Volume{1,109,37}; } } = 0.5;
MeshSize{ EdgesOf{  Volume{1,109,37}; } } = 0.5;

Mesh.CharacteristicLengthMin = 2;
Mesh.CharacteristicLengthMax = 5;
     

// Mesh sizing


//===============================================
// MESH GENERATION
//===============================================
Mesh 1;
Mesh 2;
Mesh 3;



//Save
Save "Sven_device_v1.msh2";



Show "*";
Printf("========================================");
Printf("MESH GENERATION COMPLETE");
Printf("========================================");//+

