from typing import Dict, Iterable, Callable
import tkinter as tk
import numpy as np


class DSSPlugin:
    settings = {}
    instantiate = False
    def __init__(self, owner:'DSS'):
        self.dss: 'DSS' = owner

    @classmethod
    def load_plugin(cls, owner:'DSS'):
        owner.plugins[cls] = DSSPlugin(owner) # Placeholder for plugins that are not instantiated
        if cls.instantiate:
            instance = cls(owner)
            owner.plugins[cls] = instance
            instance.load_instance()

    @classmethod
    def get_settings(cls) -> Dict[str, bool]:
        return cls.settings

    @classmethod
    def set_setting(cls, key, value):
        cls.settings[key] = value

    def load_instance(self):
        pass

    def on_after_dss_built(self):
        pass

    def get_functions(self) -> Dict[str, Callable]:
        return {}

class StandardProblemMenu(DSSPlugin):
    instantiate = True
    def __init__(self, owner:'DSS'):
        super().__init__(owner)

        self.menu_stdcases:tk.Menu

    #@classmethod
    #def get_functions(cls, caller) -> Dict[str, Callable]:
    #    return {'Cantilever beam': lambda: cls.get_model(caller, 1),
    #            'Circular arch' : lambda: cls.get_model(caller, 4)}

    def on_after_dss_built(self):
        menu_stdcases = tk.Menu(self.dss.topmenu)
        self.dss.topmenu.add_cascade(label='Standard problems',
                                     menu=menu_stdcases)

        menu_stdcases.add_command(label='Cantilever beam',
                                  command = self.cantilever_beam)
        menu_stdcases.add_command(label='Deep arch',
                                  command = self.deep_arch_half)
        menu_stdcases.add_command(label='von Mises truss',
                                  command = self.von_mises_truss)
        #menu_stdcases.add_command(label='Simply supported beam',
        #                          command=lambda: self.get_model(2))
        #menu_stdcases.add_command(label='Fanned out cantilever elements',
        #                          command=lambda: self.get_model(3))
        #menu_stdcases.add_command(label='Circular arch',
        #                          command=lambda: self.get_model(4))
        #menu_stdcases.add_command(label='270 arch',
        #                          command=lambda: self.get_model(5))

    def cantilever_beam(self):
        self.dss.new_problem()
        self.dss.problem.create_beams((0, 0), (1000, 0), n=4)
        self.dss.problem.fix(self.dss.problem.node_at((0, 0)))
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
        p.constrained_dofs = np.array([0, 1, 3*N - 3, 3*N - 1])
        p.load_node(p.nodes[-1], np.array([0, -3600, 0]))

        p.pin(p.nodes[0])
        p.glider(p.nodes[-1])

        self.dss.autoscale()

    def von_mises_truss(self):
        self.dss.new_problem()
        p = self.dss.problem
        n1 = p.get_or_create_node((0,0))
        n2 = p.get_or_create_node((1000,200))
        p.create_beam(n1, n2, A=10)
        p.pin(n1)
        p.roller90(n2)
        p.load_node(n2, np.array([0, -10000, 0]))
        self.dss.autoscale()

    def get_model(self, caller, model = 1):
        caller.new_problem()
        if model == 1:  # Cantilever beam, point load
            caller.problem.create_beams((0,0), (1000,0), n=4)
            caller.problem.fix(caller.problem.node_at((0,0)))


        if model == 2:  # Simply supported beam, no load
            caller.problem.create_beams((0,0), (1000,0))
            caller.problem.pin(caller.problem.node_at((0,0)))
            caller.problem.roller(caller.problem.node_at((1000,0)))


        if model == 3:  # Fanned out cantilever elements with load=10 distr loads
            for point in ((1000,0),(707,-707),(0,-1000),(-707,-707),(-1000,0)):
                caller.problem.create_beams((0,0),point, n=2)
                caller.problem.load_members_distr((0,0),point, load=10)

            caller.problem.fix(caller.problem.node_at((0,0)))

        if model == 4: # Circular arch
            start = np.pi - np.arctan(600 / 800)
            end = np.arctan(600 / 800)

            node_angles = np.linspace(start, end, 15)
            node_points = 1000 * np.array([np.cos(node_angles), np.sin(node_angles)]).T + np.array([800,-1600])
            for r1, r2 in zip(node_points, node_points[1:]):
                caller.problem.create_beam(r1, r2, E=2.1e5, I=10**3/12, A=10)
            caller.problem.pin(caller.problem.node_at((0,0)))
            caller.problem.pin(caller.problem.node_at((1600,0)))
            for node in caller.problem.nodes:
                node.draw = False

        if model == 5: # 270 degree arch
            start = np.deg2rad(225)
            end = np.deg2rad(-45)
            node_angles = np.linspace(start, end, 31)
            node_points = 500 * np.array([np.cos(node_angles), np.sin(node_angles)]).T + [0, 500]

            for r1, r2 in zip(node_points, node_points[1:]):
                caller.problem.create_beam(r1, r2)

            for node in caller.problem.nodes:
                node.draw = False

        caller.upd_rsmenu()
        caller.autoscale()
