// Gmsh project created on Fri Sep 19 09:12:55 2025
// 
//
SetFactory("OpenCASCADE");

// unit are in nm
// defining the width and height of the device 
// unit are in nm
// defining the width and height of the device 
D_x = 250;
D_y = 50;

barrier_width = 25;
gap = 15;
plunger_width = 25;

QD_x = barrier_width + gap + plunger_width/2;
QD_y = D_y/2;
// Define the Quantum submesh region

substrate_ht = 20;
BOX_ht = 15;
epi_Si = 15;
Thermal_SiO2 = 5;

Plunger_barrier_ht = 2;
cap_layer = 10;



Rectangle(1) = {0, -D_y/2, 0, barrier_width, D_y, 0};
Rectangle(2) = {barrier_width, -D_y/2, 0, gap, D_y, 0};
Rectangle(3) = {barrier_width+gap, -D_y/2, 0, plunger_width, D_y, 0};
Rectangle(4) = {barrier_width+gap+plunger_width, -D_y/2, 0, gap, D_y, 0};

Rectangle(5) = {barrier_width+2*gap+plunger_width, -D_y/2, 0, barrier_width, D_y, 0};
Rectangle(6) = {2*barrier_width+2*gap+plunger_width, -D_y/2, 0, gap, D_y, 0};
Rectangle(7) = {2*barrier_width+3*gap+plunger_width, -D_y/2, 0, plunger_width, D_y, 0};
Rectangle(8) = {2*barrier_width+3*gap+2*plunger_width, -D_y/2, 0, gap, D_y, 0};

Rectangle(9) = {2*barrier_width+4*gap+2*plunger_width, -D_y/2, 0, barrier_width, D_y, 0};



// Boolean fragments
BooleanFragments{ Surface{1:9}; Delete; }{ }


// Building the substrate 

Extrude {0, 0, BOX_ht} {Surface{1:9};}

Extrude {0, 0, epi_Si} {Surface{14,18,22,26,30,34,38,42,46};}

Extrude {0, 0, Thermal_SiO2} {Surface{51,55,59,63,67,71,75,79,83};}

Extrude {0, 0, -substrate_ht} {Surface{1:9};}

//===============================================
// VOLUME GROUPS
//===============================================

Physical Volume("Si_substrate") = {28:36};

Physical Volume("BOX") = {1:9};

Physical Volume("Epi_Si") = {10:18};

Physical Volume("Thermal_SiO2") = {19:27};            // Layer 2 volumes



//===============================================
// SURFACE RELABELING WITH DIRECTIONAL GROUPING
//===============================================
// Bottom surfaces
Physical Surface("bottom_surface") = {125,129,133,137,141,145,149,153,157};

// Contact surfaces
Physical Surface("Barrier1") = {88};
Physical Surface("Barrier2") = {104};
Physical Surface("Barrier3") = {120};

Physical Surface("Plunger1") = {96};
Physical Surface("Plunger2") = {112};





//===============================================
// MESH PARAMETERS
//===============================================

Mesh.CharacteristicLengthMin = 0.5;
Mesh.CharacteristicLengthMax = 1.1;
     

// Mesh sizing

//=============================================
// MESH GENERATION
//===============================================
Mesh 1;
Mesh 2;
Mesh 3;



//Save
Save "Sven_barrier_25_gap_15.msh2";




Show "*";
Printf("========================================");
Printf("MESH GENERATION COMPLETE");
Printf("========================================");//+

