# dssolver

Last commit: 5/8

Solver: 
- Now can handle evenly distributed loads 

GUI is usable: 
- Can click to start or end elements, apply loads and boundary conditions 
- Can also apply evenly distributed loads 
- Canvas coordinate system and problem coordinate systems are now separate 
- Because ^, the view can be scaled and translated while maintaining its relationship to the problem csys 
- Rod elements are now supported, but they have 3 dof's per node and only axial stiffness, so they will make the stiffness matrix singular if not connected to a beam or properly supported by boundary conditions. 

WIP: 
- Problems when drawing displaced shapes, sign of displacements etc due to mis-transformation between canvas and problem coordinates 
- Drawing normal force, shear force and moment diagrams for problems
- Labeling of distributed force
