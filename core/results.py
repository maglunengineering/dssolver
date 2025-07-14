from typing import Dict,Iterable,TypeVar,Sized

import numpy as np
from core.elements import FiniteElement, Node, DSSModelObject

T = TypeVar('T')

class _IndexListOfArrays:
    def __init__(self, list_of_2d_arrays:list[np.ndarray]):
        self._list_of_2d_arrays:list[np.ndarray] = list_of_2d_arrays
    def __getitem__(self, ijk) -> np.ndarray:
        return self._list_of_2d_arrays[ijk[0]][ijk[1:]]

class Results:
    def __init__(self, problem):
        self.nodes = list(problem.nodes)
        self.elements = list(problem.elements)

        self._displacement_results:list[np.ndarray] = [] # [iLc][iHist, iDof]
        self._force_results:list[np.ndarray] = []

        self._disp_slice = _IndexListOfArrays(self._displacement_results)
        self._force_slice = _IndexListOfArrays(self._force_results)

    def get_size(self) -> tuple[int, list[int], int]:
        return len(self._displacement_results), [x.shape[0] for x in self._displacement_results], self._displacement_results[0].shape[-1]

    def get_displacement(self, iLc, iHist, iDof) -> float:
        return self._displacement_results[iLc][iHist, iDof]

    @property
    def get_displacement_slice(self) -> np.ndarray:
        return self._disp_slice
    
    @property
    def get_force_slice(self) -> np.ndarray:
        return self._force_slice

    def get_objects(self) -> Iterable[DSSModelObject]:
        yield from self.nodes
        yield from self.elements

    def get_actions(self):
        return {}
    
    def increment(self):
        if self.current_displ_set < self.num_displ_sets - 1:
            self.current_displ_set += 1
        else:
            self.current_displ_set = 0

        return self.current_displ_set

    def decrement(self):
        if self.current_displ_set > 0:
            self.current_displ_set -= 1
        else:
            self.current_displ_set = self.num_displ_sets - 1

        return self.current_displ_set
    
    def animate(self):
        pass

    def reset_animation(self):
        pass

    def on_after_resultview_built(self, view):
        pass


class ResultsStaticLinear(Results):
    def __init__(self, problem, forces:np.ndarray, displacements:np.ndarray):
        super().__init__(problem)

        if displacements.ndim == 1: # (ndofs,)
            self._displacement_results = [displacements.reshape((1,-1))]
        elif displacements.ndim == 2: # (nlc, ndofs)
            self._displacement_results = list(displacements.reshape((displacements.shape[0], 1, -1)))
        else:
            raise ValueError(f"Illegal ndim for displacements, expected 1 or 2, got {displacements.ndim}")

        if forces.ndim == 1:
            self._force_results = [forces.reshape((1,-1))]
        elif forces.ndim == 2:
            self._force_results = list(forces)
        else:
            raise ValueError(f"Illegal ndim for forces, expected 1 or 2, got {forces.ndim}")

        self._disp_slice = _IndexListOfArrays(self._displacement_results)
        self._force_slice = _IndexListOfArrays(self._force_results)
        self.num_displ_sets = len(self._displacement_results)

class ResultsStaticNonlinear(Results):
    def __init__(self, problem, disp_histories:list[np.ndarray], load_histories:list[np.ndarray]):
        super().__init__(problem)

        self._displacement_results = disp_histories
        self._force_results = load_histories

        self.num_displ_sets = len(disp_histories)


    def get_actions(self):
        return {'Increment' : self.increment,
                'Decrement' : self.decrement}


class ResultsModal(Results):
    def __init__(self, problem, eigenvalues, eigenvectors):
        super().__init__(problem)

        self.displacements = eigenvectors
        self.eigenvalues = eigenvalues
        self.num_result_vectors = len(eigenvalues)
        self.current_result_vector = 0

    def on_after_resultview_built(self, view):
        for eigenvector in self.displacements:
            view.listbox.add(eigenvector)
        self.set_displacements()

    def animate(self):
        # 50 steps
        for sine in np.sin(np.linspace(-np.pi, np.pi, 51)):
            self.scale = sine * oldscale
            yield 20
        self.set_displacements()
        yield False

    def get_actions(self):
        return {'Increment': self.increment,
                'Decrement': self.decrement}


class ResultsDynamicTimeIntegration(ResultsStaticNonlinear):
    def __init__(self, problem, displacements):
        super().__init__(problem, displacements, np.zeros_like(displacements))
