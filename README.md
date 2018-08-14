# DSSolver

DSSolver is a direct stiffness method-based solver for simple 2D linear statics problems. 

To run, execute dssolver.bat (Windows) or dssolver.sh (Linux) or just execute gui.py with your Python interpreter. Requires Python 3 with tkinter and NumPy.

Features: 
- 6-DOF Beam and Rod elements 
- Any number of fixed-, pinned-, roller- and rotation lock-supports 
- Point loads and evenly distributed loads 
- Calculates displacements, forces and stresses 
- Drawing displaced shapes, shear force- and moment diagrams 

Last commit: 14/8-2018 
Recent changes: 
- There is now a right side menu is added with show/hide toggles and element info 
- Solver now calculates stress (x-top, x-bottom, average shear) and "half section height" is now a beam element attribute. Stress is shown in the right side menu 
- There is now a section manager for auto-calculation of area, second moment of area and half section height for rectangular, circular and I-beam cross sections
