import os
import sys
import tkinter as tk
import numpy as np

sys.path.append(os.path.dirname(__file__))
from guis.tkinter.plugin_base import DSSPlugin

class StandardProblemMenu(DSSPlugin):
    instantiate = True
    def __init__(self, owner:'DSS'):
        super().__init__(owner)

        self.menu_stdcases:tk.Menu

    def on_after_dss_built(self):
        self.dss.add_topmenu_item('Standard problems', 'Cantilever beam', self.cantilever_beam)
        self.dss.add_topmenu_item('Standard problems', 'Deep arch', self.deep_arch_half)
        self.dss.add_topmenu_item('Standard problems', 'Deep arch (full)', self.deep_arch)
        self.dss.add_topmenu_item('Standard problems', 'von Mises truss', self.von_mises_truss)
        self.dss.add_topmenu_item('Standard problems', 'Snapback von Mises truss', self.von_mises_truss_snapback)
        self.dss.add_topmenu_item('Standard problems', 'Standing rod', self.standing_rod)
        self.dss.add_topmenu_item('Standard problems', '270 arch', self.arch_270)
        self.dss.add_topmenu_item('Standard problems', 'Pendulum', self.pendulum)
        self.dss.add_topmenu_item('Standard problems', 'von Mises Truss (spring BC)', self.von_mises_truss_springbc)

    def cantilever_beam(self):
        self.dss.new_problem()
        self.dss.problem.create_beams((0, 0), (1000, 0), n=8)
        self.dss.problem.node_at((0, 0)).fix()
        self.dss.autoscale()

    def deep_arch_half(self):
        self.dss.new_problem()
        p = self.dss.problem
        N = 16
        dofs = 2*(3*N - 2,)
        start = np.pi - np.arctan(600/800)
        end = np.pi/2

        node_angles = np.linspace(start, end, N)
        node_points = 1000*np.array([np.cos(node_angles), np.sin(node_angles)]).T
        for r in node_points:
            p.get_or_create_node(r)
        for n1, n2 in zip(p.nodes, p.nodes[1:]):
            p.create_beam(n1, n2, E=2.1e5, A=10, I=10**3/12)

        p.reassign_dofs()
        p.constrained_dofs = [0, 1, 3*N - 3, 3*N - 1]
        p.nodes[-1].loads = np.array([0, -3600, 0])

        p.nodes[0].pin()
        p.nodes[-1].glider()

        self.dss.autoscale()

    def deep_arch(self):
        self.dss.new_problem()
        p = self.dss.problem
        N = 17
        dofs = 2*(3*N - 2,)
        start = np.pi - np.arctan(600/800)
        end = np.pi + np.arctan(600/800)

        node_angles = np.linspace(start, end, N)
        node_points = 1000*np.array([np.sin(node_angles), -np.cos(node_angles)]).T
        for r in node_points:
            p.get_or_create_node(r)
        for n1, n2 in zip(p.nodes, p.nodes[1:]):
            p.create_beam(n1, n2, E=2.1e5, A=10, I=10**3/12)

        p.reassign_dofs()
        #p.constrained_dofs = [0, 1, 3*N-2, 3*N-1]
        p.nodes[len(p.nodes)//2].loads = np.array([0, -1000, 0])

        p.nodes[0].pin()
        p.nodes[-1].pin()

        self.dss.autoscale()

    def von_mises_truss(self):
        self.dss.new_problem()
        p = self.dss.problem
        n1 = p.get_or_create_node((0,0))
        n2 = p.get_or_create_node((1000,200))
        p.create_beam(n1, n2, A=10)
        n1.pin()
        n2.roller90()
        n2.loads = np.array([0, -10000, 0])
        self.dss.autoscale()

    def von_mises_truss_snapback(self):
        self.dss.new_problem()
        p = self.dss.problem
        n1 = p.get_or_create_node((0, 0))
        n2 = p.get_or_create_node((1000, 200))
        n3 = p.get_or_create_node((1000, 600))
        p.create_beam(n1, n2, A=10)
        p.create_rod(n2, n3, A=0.05)
        n1.pin()
        n2.roller90()
        n3.glider()
        n3 = np.array([0, -4000, 0])
        self.dss.autoscale()

    def von_mises_truss_springbc(self):
        self.dss.new_problem()
        p = self.dss.problem
        n1 = p.get_or_create_node((0, 0))
        n2 = p.get_or_create_node((1000, 200))
        n3 = p.get_or_create_node((1000, 600))
        n4 = p.get_or_create_node((0, 400))
        p.create_beam(n1, n2, A=10)
        p.create_rod(n2, n3, A=0.05)
        p.create_rod(n1, n4, A=0.05)
        n4.fix()
        n1.roller90()
        n2.roller90()
        n3.glider()
        n3 = np.array([0, -4000, 0])
        self.dss.autoscale()

    def standing_rod(self):
        self.dss.new_problem()
        p = self.dss.problem
        n1 = p.get_or_create_node((0,0))
        n2 = p.get_or_create_node((0,1000))
        p.create_rod(n1, n2, A=10)
        n1.glider()
        n2.fix()
        n1.loads = np.array([0, 1e6, 0])

        self.dss.autoscale()

    def arch_270(self):
        self.dss.new_problem()
        p = self.dss.problem
        start = np.deg2rad(225)
        end = np.deg2rad(-45)
        n = 31
        node_angles = np.linspace(start, end, n)
        node_points = 500 * np.array([np.cos(node_angles), np.sin(node_angles)]).T + np.array([0, 500])

        for r1, r2 in zip(node_points, node_points[1:]):
            p.create_beam(r1, r2)

        p.nodes[0].pin()
        p.nodes[-1].fix()
        p.nodes[n//2].loads = np.array([0, -200000, 0])
        self.dss.autoscale()

    def pendulum(self):
        self.dss.new_problem()
        p = self.dss.problem
        n1 = p.get_or_create_node((0, 0))
        n2 = p.get_or_create_node((1000, 0))
        p.create_rod(n1, n2, A=10)
        n1.pin()

        self.dss.autoscale()