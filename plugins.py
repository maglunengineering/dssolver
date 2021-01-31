import tkinter as tk
import numpy as np

class DSSPlugin:
    def __init__(self, owner):
        self.owner = owner
        self.owner.plugins[self.__class__.__name__] = self

    def init(self):
        pass

    @classmethod
    def create_instance(cls, owner):
        print("Called init on class " + cls.__name__)
        return cls(owner)


class StandardProblemMenu(DSSPlugin):
    def __init__(self, owner):
        super().__init__(owner)
        self.dss = owner

        self.menu_stdcases:tk.Menu

    def init(self):
        menu_stdcases = tk.Menu(self.owner.topmenu)
        self.owner.topmenu.add_cascade(label="Standard problems",
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

    def get_model(self, loadcase = 1):
        self.dss.new_problem()
        if loadcase == 1:  # Cantilever beam, point load
            self.dss.problem.create_beams((0,0), (1000,0), n=4)
            self.dss.problem.fix(self.dss.problem.node_at((0,0)))


        if loadcase == 2:  # Simply supported beam, no load
            self.dss.problem.create_beams((0,0), (1000,0))
            self.dss.problem.pin(self.dss.problem.node_at((0,0)))
            self.dss.problem.roller(self.dss.problem.node_at((1000,0)))


        if loadcase == 3:  # Fanned out cantilever elements with load=10 distr loads
            for point in ((1000,0),(707,-707),(0,-1000),(-707,-707),(-1000,0)):
                self.dss.problem.create_beams((0,0),point, n=2)
                self.dss.problem.load_members_distr((0,0),point, load=10)

            self.dss.problem.fix(self.dss.problem.node_at((0,0)))

        if loadcase == 4: # Circular arch
            start = np.pi - np.arctan(600 / 800)
            end = np.arctan(600 / 800)

            node_angles = np.linspace(start, end, 15)
            node_points = 1000 * np.array([np.cos(node_angles), np.sin(node_angles)]).T + np.array([800,-1600])
            for r1, r2 in zip(node_points, node_points[1:]):
                self.dss.problem.create_beam(r1, r2, E=2.1e5, I=10**3/12, A=10)
            self.dss.problem.pin(self.dss.problem.node_at((0,0)))
            self.dss.problem.pin(self.dss.problem.node_at((1600,0)))
            for node in self.dss.problem.nodes:
                node.draw = False

        if loadcase == 5: # 270 degree arch
            start = np.deg2rad(225)
            end = np.deg2rad(-45)
            node_angles = np.linspace(start, end, 31)
            node_points = 500 * np.array([np.cos(node_angles), np.sin(node_angles)]).T + [0, 500]

            for r1, r2 in zip(node_points, node_points[1:]):
                self.dss.problem.create_beam(r1, r2)

            for node in self.dss.problem.nodes:
                node.draw = False

        self.dss.upd_rsmenu()
        self.dss.autoscale()
