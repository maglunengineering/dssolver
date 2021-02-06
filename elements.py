from typing import Dict

import numpy as np
import tkinter as tk

from extras import DSSCanvas
from plugins import DSSPlugin

class DSSModelObject(DSSPlugin):
    def draw_on_canvas(self, canvas, **kwargs):
        raise NotImplementedError

class Node(DSSModelObject):
    settings = {'Loads': True,
                'Boundary conditions': True}

    def __init__(self, xy, draw=False):
        self.x, self.y = xy
        self.r = np.array(xy)

        self.elements = list()
        self.loads = np.zeros(3) # self.loads (global Fx, Fy, M) assigned on loading
        self.displacements = np.zeros(3)

        self.number = None  # (node id) assigned on creation
        self.dofs:np.ndarray = None  # self.dofs (dof1, dof2, dof3) assigned on creation
        self.boundary_condition = None  # 'fixed', 'pinned', 'roller', 'locked', 'glider'

        self.draw = draw  # Node is not drawn unless it is interesting

    def add_beam(self, beam):
        self.elements.append(beam)

    @property
    def abs_forces(self):
        self.forces = np.array([0,0,0])
        for element in self.elements:
            if np.array_equal(element.r1, self.r):
                self.forces = self.forces + np.abs(element.forces[0:3]) / 2
            elif np.array_equal(element.r2, self.r):
                self.forces = self.forces + np.abs(element.forces[3:6]) / 2
        return self.forces + np.abs(self.loads/2)

    def connected_nodes(self):
        other_nodes = []
        for element in self.elements:
            for node in element.nodes:
                if not node == self:
                    other_nodes.append(node)
        return other_nodes

    def translate(self, dx, dy):
        self.x, self.y = self.x+dx, self.y+dy
        self.r = np.array([self.x, self.y])

    def draw_on_canvas(self, canvas:DSSCanvas, **kwargs):
        canvas.draw_node(self.r, 2.5, **kwargs)

        # If lump force, draw an arrow
        if self.settings['Loads']:
            self.draw_loads(canvas, **kwargs)

        if self.settings['Boundary conditions']:
            self.draw_boundary_condition(canvas, **kwargs)

    def draw_loads(self, canvas:DSSCanvas):
        scale = 100
        if np.any(np.round(self.loads[0:2])):
            arrow_start = self.r
            arrow_end = self.r + self.loads[0:2]/np.linalg.norm(self.loads[0:2])*scale
            canvas.draw_line(arrow_start, arrow_end,
                             arrow='last', fill='blue', tag='mech')
            canvas.draw_text((arrow_start + arrow_end)/2,
                             '{}'.format(self.loads[0:2]),
                             anchor='sw', tag='mech')

        # If moment, draw a circular arrow
        if self.loads[2] != 0:
            sign = np.sign(self.loads[2])
            arc_start = self.r + np.array([0, -scale/2])*sign
            arc_mid = self.r + np.array([scale/2, 0])*sign
            arc_end = self.r + np.array([0, scale/2])*sign

            arrow = 'first' if sign == 1 else 'last'
            canvas.draw_arc(arc_start, arc_mid, arc_end,
                            smooth=True,
                            arrow=arrow, fill='blue', tag='mech')
            canvas.draw_text(arc_start,
                             text='{}'.format(self.loads[2]),
                             anchor='ne', tag='mech')
            
    def draw_boundary_condition(self, canvas:DSSCanvas):
        scale = 50
        linewidth = 2

        if self.boundary_condition == 'fixed':
            angle_vector = sum(n.r - self.r for n in self.connected_nodes())
            angle = np.arctan2(*angle_vector[::-1])
            c, s = np.cos(-angle), np.sin(-angle)
            rotation = np.array([[c, -s], [s, c]])

            canvas.draw_line((self.r + rotation@[0, scale]), (self.r + rotation@[0, -scale]),
                                    width=linewidth, fill='black', tag='bc')
            for offset in np.linspace(0, 2*scale, 6):
                canvas.draw_line((self.r + rotation@[0, -scale + offset]),
                                        (self.r + rotation@[0, -scale + offset] + rotation@[-scale/2, scale/2]),
                                        width=linewidth, fill='black', tag='bc')

        elif self.boundary_condition == 'pinned' or self.boundary_condition == 'roller':
            k = 1.5  # constant - triangle diameter

            canvas.draw_oval((self.r - scale/4), (self.r + scale/5))
            canvas.draw_line(self.r, (self.r + np.array([-np.sin(np.deg2rad(30)),
                                                                  np.cos(np.deg2rad(30))])*k*scale),
                                    width=linewidth, fill='black', tag='bc')
            canvas.draw_line(self.r, (self.r + np.array([np.sin(np.deg2rad(30)),
                                                                  np.cos(np.deg2rad(30))])*k*scale),
                                    width=linewidth, fill='black', tag='bc')

            canvas.draw_line((self.r + (np.array([-np.sin(np.deg2rad(30)),
                                                          np.cos(np.deg2rad(30))])
                                                + np.array([-1.4/(k*scale), 0])
                                                )*k*scale),
                                    (self.r + (np.array([np.sin(np.deg2rad(30)),
                                                          np.cos(np.deg2rad(30))])
                                                + np.array([1.4/(k*scale), 0])
                                                )*k*scale),
                                    width=linewidth, fill='black', tag='bc')
            if self.boundary_condition == 'roller':
                canvas.draw_line((self.r + np.array([-np.sin(np.deg2rad(30)),
                                                             np.cos(np.deg2rad(30))])*k*scale
                                          + np.array([-scale/2, scale/4])),
                                        (self.r + np.array([np.sin(np.deg2rad(30)),
                                                             np.cos(np.deg2rad(30))])*k*scale)
                                         + np.array([scale/2, scale/4]),
                                        width=linewidth, fill='black', tag='bc')


        elif self.boundary_condition == 'locked':
            canvas.draw_oval((self.r + np.array([-scale, -scale])),
                                    (self.r - np.array([-scale, -scale])),
                                    width=linewidth, tag='bc')
            canvas.draw_line(self.r, (self.r + np.array([scale/2, -scale])*1.4),
                                    width=linewidth, fill='black', tag='bc')

        elif self.boundary_condition == 'glider':
            angle = 0  # Could be pi
            c, s = np.cos(-angle), np.sin(-angle)
            rotation = np.array([[c, -s], [s, c]])

            canvas.draw_line((self.r + rotation@[0, scale]), (self.r + rotation@[0, -scale]),
                                    width=linewidth, fill='black', tag='bc')
            canvas.draw_oval((self.r + rotation@[0, -scale/4]), (self.r + rotation@[scale/2, -3*scale/4]))
            canvas.draw_oval((self.r + rotation@[0, scale/4]), (self.r + rotation@[scale/2, 3*scale/4]))
            canvas.draw_line((self.r + rotation@[scale/2, 0] + rotation@[0, scale]),
                                    (self.r + rotation@[scale/2, 0] + rotation@[0, -scale]),
                                    width=linewidth, fill='black', tag='bc')

            for offset in np.linspace(0, 2*scale, 6):
                canvas.draw_line((self.r + rotation@[scale/2, -scale + offset]),
                                        (self.r + rotation@[scale/2, -scale + offset]
                                          + rotation@[scale/2, scale/2]),
                                        width=linewidth, fill='black', tag='bc')

    def __str__(self):
        return '{},{}'.format(self.x, self.y)

    def __hash__(self):
        return id(self)

class FiniteElement(DSSModelObject):
    settings = {'Displaced': False}

    def __init__(self, node1:Node, node2:Node):
        self.nodes = [node1, node2]
        self.node1 = node1
        self.node2 = node2

        for node in self.nodes:
            node.add_beam(self)

    def draw_on_canvas(self, canvas, **kwargs):
        canvas.draw_line(self.node1.r, self.node2.r, **kwargs)
        if self.settings['Displaced']:
            canvas.draw_line(self.node1.r + self.node1.displacements[0:2],
                             self.node2.r + self.node2.displacements[0:2],
                             fill='red', dash=(1,), **kwargs)



class Beam(FiniteElement):
    def __init__(self, node1:Node, node2:Node, E=2e5, A=1e5, I=1e5, z=None):
        super().__init__(node1, node2)

        self.E = E
        self.A = A
        self.I = I
        self.z = z if z else np.sqrt(I/A)/3

        self.angle = np.arctan2( *(self.node2.r - self.node1.r)[::-1] )
        self.length = np.linalg.norm(node2.r - node1.r)

        self.number = None  # (beam id) assigned on creation
        self.stress = np.zeros(6)  # local csys(1x6) assigned on solution
        # Stress given as sigma_x (top), sigma_x (bottom), tau_xy (average!)
        self.member_loads = np.zeros(6)  # local csys distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12
        self.distributed_load = 0
        # Distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12


        self.kn = A*E/self.length * (E*I/self.length**3)**(-1)
        self.ke = E*I/self.length**3 * np.array(
             [[self.kn, 0, 0, -self.kn, 0, 0],
              [0, 12, 6*self.length, 0, -12, 6*self.length],
              [0, 6*self.length, 4*self.length**2, 0, -6*self.length, 2*self.length**2],
              [-self.kn, 0, 0, self.kn, 0, 0],
              [0, -12, -6*self.length, 0, 12, -6*self.length],
              [0, 6*self.length, 2*self.length**2, 0, -6*self.length, 4*self.length**2]])

        self.k = self.beta(self.angle).T @ self.ke @ self.beta(self.angle)

        self.cpl = np.zeros((6,6))
        self.cpl_ = np.array([[1/self.A, 0, -self.z/self.I],
                             [1/self.A, 0, self.z/self.I],
                             [0,     1/self.A,   0]])
        self.cpl[0:3, 0:3] = self.cpl[3:6, 3:6] = self.cpl_

    def beta(self, angle):
        s, c = np.sin(angle), np.cos(angle)
        return np.array([[c, s, 0, 0, 0, 0],
                         [-s, c, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, c, s, 0],
                         [0, 0, 0, -s, c, 0],
                         [0, 0, 0, 0, 0, 1]])

    def Ki(self, newdim):
        dofs = np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E @ self.k @ E.T

    def Qfi(self, newdim):
        dofs = np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E @ self.member_loads

    def Ex(self, newdim):
        dofs = np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E

    def translate(self, translation, loaded=True):
        # translation: global vector [u1,v1,theta1, u2,v2,theta2]
        self.nodes[0].translate(translation[0], translation[1])
        self.nodes[1].translate(translation[3], translation[4])
        if loaded:
            self.forces = self.forces + self.k @ translation
            self.displacements = self.displacements + translation
        tr_local = beta(self.angle).T @ translation
        angle_change = np.arcsin((tr_local[4] - tr_local[1]) / self.length)
        self.k = beta(angle_change).T @ self.k @ beta(angle_change)

    def strain_energy(self, delta_displacements=np.zeros(6)):
        """
        :param delta_displacements: Additional displacements for strain energy calculations.
        The derivative of strain_energy wrt. displacement 2 is thus (strain_energy([0,0,dd,0,0,0])-strain_energy)/dd
        """
        N1,V1,M1,N2,V2,M2 = self.ke @ beta(self.angle) @ (self.displacements + delta_displacements)
        L = self.length
        return 1/(2*self.E*self.I) * (M1**2*L - M1*V1*L**2 + V1**2 * L**3/3) + \
               (N2**2 * L)/(2*self.E*self.A)

    @property
    def dofs(self):
        return np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))

    def get_displacements(self):
        return np.hstack((self.nodes[0].displacements, self.nodes[1].displacements))

    def get_forces(self):
        return self.k @ self.get_displacements()

    @property
    def r1(self):
        return self.node1.r

    @property
    def r2(self):
        return self.node2.r

    def __eq__(self, other):
        return np.allclose(np.array([self.r1, self.r2]), np.array([other.r1, other.r2]))

class Rod(Beam):

    def __init__(self, r1, r2, E=2e5, A=1e5, *args, **kwargs):
        super().__init__(r1=r1, r2=r2, E=E, A=A)

        self.kn = A*E/self.length
        self.ke = np.array(      [[self.kn, 0, 0, -self.kn, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [-self.kn, 0, 0, self.kn, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]])

        self.k = self.beta(self.angle).T @ self.ke @ self.beta(self.angle)

def beta(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, s, 0, 0, 0, 0],
                     [-s, c, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, c, s, 0],
                     [0, 0, 0, -s, c, 0],
                     [0, 0, 0, 0, 0, 1]])

