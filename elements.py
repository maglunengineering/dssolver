from typing import Dict

import numpy as np
import tkinter as tk

from extras import *
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

    def add_element(self, beam):
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

    def copy(self):
        new_node = Node(self.r)
        new_node.dofs = self.dofs
        return new_node

    def __str__(self):
        return '{},{}'.format(self.x, self.y)

    def __hash__(self):
        return id(self)

class FiniteElement2Node(DSSModelObject):
    settings = {'Displaced': False}

    def __init__(self, node1:Node, node2:Node):
        self.nodes = [node1, node2]
        self.node1 = node1
        self.node2 = node2

        for node in self.nodes:
            node.add_element(self)

    @property
    def dofs(self):
        return np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))

    def get_displacements(self):
        return np.hstack((self.nodes[0].displacements, self.nodes[1].displacements))

    def get_forces(self):
        return self.stiffness_matrix_global() @ self.get_displacements()

    def draw_on_canvas(self, canvas, **kwargs):
        canvas.draw_line(self.node1.r, self.node2.r, **kwargs)
        if self.settings['Displaced']:
            canvas.draw_line(self.node1.r + self.node1.displacements[0:2],
                             self.node2.r + self.node2.displacements[0:2],
                             fill='red', dash=(1,), **kwargs)


class Beam(FiniteElement2Node):
    def __init__(self, node1:Node, node2:Node, E=2e5, A=1e5, I=1e5, z=None):
        super().__init__(node1, node2)

        self.E = E
        self.A = A
        self.I = I
        self.z = z if z else np.sqrt(I/A)/3

        angle = np.arctan2( *(self.node2.r - self.node1.r)[::-1] )
        length = np.linalg.norm(node2.r - node1.r)

        self.number = None
        self.stress = np.zeros(6)

        # Stress given as sigma_x (top), sigma_x (bottom), tau_xy (average!)
        self.member_loads = np.zeros(6)  # local csys distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12
        self.distributed_load = 0
        # Distr load: 0, pL/2, pLL/12, 0, pL/2, -pLL/12

        kn = A*E/length * (E*I/length**3)**(-1)
        self.stiffness_matrix_local = E*I/length**3 * np.array(
             [[kn, 0, 0, -kn, 0, 0],
              [0, 12, 6*length, 0, -12, 6*length],
              [0, 6*length, 4*length**2, 0, -6*length, 2*length**2],
              [-kn, 0, 0, kn, 0, 0],
              [0, -12, -6*length, 0, 12, -6*length],
              [0, 6*length, 2*length**2, 0, -6*length, 4*length**2]])

        self.cpl = np.zeros((6,6))
        self.cpl_ = np.array([[1/self.A, 0, -self.z/self.I],
                             [1/self.A, 0, self.z/self.I],
                             [0,     1/self.A,   0]])
        self.cpl[0:3, 0:3] = self.cpl[3:6, 3:6] = self.cpl_

    def transform(self):
        e1 = ((self.node2.r + self.node2.displacements[:2]) -
              (self.node1.r + self.node1.displacements[0:2])) / self._deformed_length()
        e2 = R(np.deg2rad(90)) @ e1 # TODO: Just make it like [-e2, e1] or whatever
        T = np.array([[*e1, 0, *np.zeros(3)],
                      [*e2, 0, *np.zeros(3)],
                      [0, 0, 1,*np.zeros(3)],
                      [*np.zeros(3), *e1, 0],
                      [*np.zeros(3), *e2, 0],
                      [*np.zeros(3), 0, 0, 1]])
        return T

    def _get_forces_local(self):
        undeformed_length = np.linalg.norm(self.node2.r - self.node1.r)
        dl = self._deformed_length() - undeformed_length

        tan_e = (self.node2.r - self.node1.r)/undeformed_length
        tan_ed = (self.node2.r + self.node2.displacements[0:2] -
                  self.node1.r - self.node1.displacements[0:2])/self._deformed_length()
        tan_1 = R(self.node1.displacements[2])@tan_e
        tan_2 = R(self.node2.displacements[2])@tan_e

        th1 = np.arcsin(tan_ed[0]*tan_1[1] - tan_ed[1]*tan_1[0])
        th2 = np.arcsin(tan_ed[0]*tan_2[1] - tan_ed[1]*tan_2[0])

        displacements_local = np.array([-dl/2, 0, th1, dl/2, 0, th2])
        forces_local = self.stiffness_matrix_local @ displacements_local
        return forces_local

    def get_forces(self):
        return self.transform().T @ self._get_forces_local()

    #@log
    def stiffness_matrix_global(self):
        T = self.transform()
        return T.T @ (self.stiffness_matrix_local ) @ T + self.stiffness_matrix_geometric()

    #@Logger.log
    def stiffness_matrix_geometric(self):
        fx1,fy1,m1,fx2,fy2,m2 = self._get_forces_local()
        deformed_length = self._deformed_length()

        forces_permuted = np.array([-fy1, fx1, 0, -fy2, fx2, 0])
        G = np.array([0, -1/deformed_length, 0, 0, 1/deformed_length, 0])
        T = self.transform()
        return T.T @ np.outer(forces_permuted, G) @ T

    def member_loads_expanded(self, newdim):
        return self._Ex(newdim) @ self.member_loads

    def _Ex(self, newdim):
        dofs = np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))
        E = np.zeros((newdim, len(dofs)))
        for i, j in enumerate(dofs):
            E[j, i] = 1
        return E

    def expand(self, arr, newdim):
        E = self._Ex(newdim)
        if len(arr.shape) == 1:
            return E @ arr
        elif len(arr.shape) == 2:
            return E @ arr @ E.T

    def _deformed_length(self):
        return np.linalg.norm((self.node2.r + self.node2.displacements[:2] -
                               self.node1.r - self.node1.displacements[:2]))

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
        super().__init__(node1=r1, node2=r2, E=E, A=A)

        self.kn = A*E/self.length
        self.ke = np.array(      [[self.kn, 0, 0, -self.kn, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [-self.kn, 0, 0, self.kn, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]])

        self.k = self.transform(self.angle).T@self.ke@self.transform(self.angle)

def beta(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, s, 0, 0, 0, 0],
                     [-s, c, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, c, s, 0],
                     [0, 0, 0, -s, c, 0],
                     [0, 0, 0, 0, 0, 1]])

