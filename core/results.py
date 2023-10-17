from typing import Dict,Iterable,TypeVar,Sized

import numpy as np
import matplotlib.pyplot as plt
from core.elements import FiniteElement2Node, Node, DSSModelObject

T = TypeVar('T')

class Results:
    def __init__(self, problem):
        self.nodes = list(problem.nodes)
        self.elements = list(problem.elements)

        self.current_result_vector = 0
        self.num_displ_sets = 1

    def get_objects(self) -> Iterable[DSSModelObject]:
        yield from self.nodes
        yield from self.elements

    def get_buttons(self):
        return {}

    def get_text(self):
        return f'Current results vector: {self.current_result_vector}'

    def increment(self):
        if self.current_result_vector < self.num_displ_sets - 1:
            self.current_result_vector += 1
        else:
            self.current_result_vector = 0

        self.set_displacements()

    def decrement(self):
        if self.current_result_vector > 0:
            self.current_result_vector -= 1
        else:
            self.current_result_vector = self.num_displ_sets - 1

        self.set_displacements()

    def animate(self):
        pass

    def set_displacements(self):
        pass

    def reset_animation(self):
        pass

    def on_after_resultview_built(self, view):
        pass


class ResultsStaticLinear(Results):
    def __init__(self, problem, displacements:np.ndarray):
        super().__init__(problem)

        self.displacements = displacements
        for node in self.nodes:
            node.displacements = displacements[node.dofs]


class ResultsStaticNonlinear(Results):
    def __init__(self, problem, displacements:Sized, forces:Sized):
        super().__init__(problem)

        self.forces = forces
        self.displacements = displacements
        self.num_displ_sets = len(displacements)

    def set_displacements(self):
        displacements = self.displacements[self.current_result_vector]
        for node in self.nodes:
            node.displacements = displacements[node.dofs]

    def animate(self):
        interval = int(1000 * 2 / self.num_displ_sets)
        self.current_result_vector = 0
        for step in range(self.num_displ_sets):
            self.set_displacements()
            self.current_result_vector += 1
            yield interval
        yield False

    def quickplot(self):
        for node in self.nodes:
            if node.loads.any():
                break
        dof = node.dofs[np.abs(node.loads).argmax()]
        displ_history = self.displacements[:,dof]
        sign = np.sign(np.average(displ_history))
        load_history = self.forces

        plt.ylabel('Control parameter')
        plt.xlabel('Displacement')
        plt.title(f'Displacement vs control parameter at dof {dof}')
        plt.plot(sign * displ_history, load_history)
        plt.show()

    def get_buttons(self):
        return {'Increment' : self.increment,
                'Decrement' : self.decrement,
                'Quick plot' : self.quickplot}


class ResultsModal(Results):
    def __init__(self, problem, eigenvalues, eigenvectors):
        super().__init__(problem)

        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.num_result_vectors = len(eigenvalues)
        self.current_result_vector = 0
        self.scale = problem.model_size() / 5

    def set_displacements(self):
        eigenvector = self.eigenvectors[self.current_result_vector] * self.scale
        eigenvalue = self.eigenvalues[self.current_result_vector]
        for node in self.nodes:
            node.displacements = eigenvector[node.dofs]

    def on_after_resultview_built(self, view):
        for eigenvector in self.eigenvectors:
            view.listbox.add(eigenvector)
        self.set_displacements()

    def animate(self):
        # 50 steps
        oldscale = self.scale
        for sine in np.sin(np.linspace(-np.pi, np.pi, 51)):
            self.scale = sine * oldscale
            self.set_displacements()
            yield 20
        self.scale = oldscale
        self.set_displacements()
        yield False

    def get_text(self):
        return f'Eigenvalue no. {self.current_result_vector}: ' \
               f'{np.round(self.eigenvalues[self.current_result_vector], 2)}'

    def get_buttons(self):
        return {'Increment': self.increment,
                'Decrement': self.decrement}


class ResultsDynamicTimeIntegration(ResultsStaticNonlinear):
    def __init__(self, problem, displacements):
        super().__init__(problem, displacements, np.zeros_like(displacements))
