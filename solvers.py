from typing import *
import numpy as np

from plugins import DSSPlugin
from dssgui import DSS
from problem import Problem
import results
import extras

class Solver(DSSPlugin):
    def __init__(self, owner):
        super().__init__(owner)
        self.results = None

    def solve(self, problem:Problem) -> results.Results:
        pass

    @classmethod
    def load_plugin(cls, owner: DSS):
        super().load_plugin(owner)

        instance = cls(owner)
        if not cls in owner.plugin_instances:
            owner.plugin_instances[cls] = []
        owner.plugin_instances[cls].append(instance)


class LinearSolver(Solver):
    def solve(self, problem:'Problem') -> results.ResultsStaticLinear:
        problem.reassign_dofs()
        problem.remove_dofs()

        ndofs = 3 * len(problem.nodes)
        free_dofs = np.delete(np.arange(ndofs), problem.constrained_dofs)
        stiffness_matrix = problem.K(reduced=True)
        loads = problem.loads[free_dofs]

        reduced_displacements = np.linalg.solve(stiffness_matrix, loads)
        displacements = np.zeros(ndofs)
        displacements[free_dofs] = reduced_displacements

        # To be removed:
        problem.displacements = displacements
        problem.upd_obj_displacements() # To be removed

        return results.ResultsStaticLinear(problem.nodes, problem.elements, displacements)

    def get_functions(self) -> Dict[str, Callable]:
        return {'Linear' : lambda : self.solve(self.owner.problem)}

class NonLinearSolver(Solver):
    def solve(self, problem:'Problem') -> results.ResultsStaticNonlinear:
        steps = 600
        arclength = 35
        A = 0
        dA = 1
        max_it = 35

        @extras.log
        def get_residual(q, a):
            return self.get_internal_forces(problem)[free_dofs] - q*a

        problem.reassign_dofs()
        problem.remove_dofs()
        free_dofs = problem.free_dofs()

        max_A = np.linalg.norm(problem.loads[free_dofs])
        q = problem.loads[free_dofs] / max_A
        displacements = np.zeros(len(problem.nodes) * 3)

        displ_storage = np.zeros((steps, 3 * len(problem.nodes)))

        i = 0
        while i < steps and A < max_A:
            print("Predictor step {}".format(i))
            K = problem.K(True)
            wq0 = np.linalg.solve(K, q)
            f = np.sqrt(1 + wq0 @ wq0)

            sign = np.sign(wq0@v0 if i > 1 else 1)
            dA = arclength / f * sign
            #dA = np.abs(dA) * sign
            v0 = dA*wq0
            A += dA

            displacements[free_dofs] = displacements[free_dofs] + v0
            for node in problem.nodes:
                node.displacements = displacements[node.dofs]

            # Corrector
            k = 0
            residual = get_residual(q, A)
            while np.linalg.norm(residual) > 1e-3 and k < max_it:
                K = problem.K(True)
                wq = np.linalg.solve(K, q)
                wr = np.linalg.solve(K, -residual)
                dA_ = -wq @ wr / (1 + wq @ wq)
                A += dA_

                displacements[free_dofs] = displacements[free_dofs] + (wr + dA_*wq)
                for node in problem.nodes:
                    node.displacements = displacements[node.dofs]
                k += 1

                residual = get_residual(q, A)

            displ_storage[i] = displacements
            i += 1

            print(f"Finished in {k} corrector steps (stepped {dA}, arclength {arclength})")
            if k < 4:
                arclength *= 1.1
            elif k > 8:
                arclength *= 0.6
            else:
                arclength *= 0.85

        print(displ_storage)
        print("Ended at A = {}".format(A))

        return results.ResultsStaticNonlinear(problem.nodes, problem.elements, displ_storage)

    def get_internal_forces(self, problem):
        ndofs = 3 * len(problem.nodes)
        forces = np.zeros(ndofs)
        for element in problem.elements:
            forces = forces + element.expand(element.get_forces(), ndofs)

        return forces




    def get_functions(self) -> Dict[str, Callable]:
        return {'Nonlinear': lambda : self.solve(self.owner.problem)}