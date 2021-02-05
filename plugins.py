from typing import Dict, Iterable, Callable
import tkinter as tk
import numpy as np

class DSSPlugin:
    settings = {}

    @classmethod
    def load_plugin(cls, owner):
        owner.plugin_types[cls.__name__] = cls

    @classmethod
    def get_settings(cls) -> Dict[str, bool]:
        return cls.settings

    @classmethod
    def set_setting(cls, key, value):
        cls.settings[key] = value

    @classmethod
    def get_functions(cls) -> Dict[str, Callable]:
        return {}

class StandardProblemMenu(DSSPlugin):
    def __init__(self, owner):
        self.dss = owner

        self.menu_stdcases:tk.Menu

    #@classmethod
    #def load_plugin(cls, dss):
    #    dss.plugin_instances[cls] = []
    #    instance = cls(dss)
    #    instance.load_instance()

    @classmethod
    def get_functions(cls, caller) -> Dict[str, Callable]:
        return {'Cantilever beam': lambda: cls.get_model(caller, 1),
                'Circular arch' : lambda: cls.get_model(caller, 4)}

    def load_instance(self):
        self.dss.plugin_instances[self.__class__].append(self)
        menu_stdcases = tk.Menu(self.dss.topmenu)
        self.dss.topmenu.add_cascade(label='Standard problems',
                                       menu=menu_stdcases)

        menu_stdcases.add_command(label='Cantilever beam',
                                  command=lambda: self.get_model(1))
        menu_stdcases.add_command(label='Simply supported beam',
                                  command=lambda: self.get_model(2))
        menu_stdcases.add_command(label='Fanned out cantilever elements',
                                  command=lambda: self.get_model(3))
        menu_stdcases.add_command(label='Circular arch',
                                  command=lambda: self.get_model(4))
        menu_stdcases.add_command(label='270 arch',
                                  command=lambda: self.get_model(5))
    @classmethod
    def get_model(cls, caller, model = 1):
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
