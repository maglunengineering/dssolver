from typing import *
import numpy as np

from plugins import DSSPlugin
from problem import Problem
import results

class Solver(DSSPlugin):
    instantiate = True
    def __init__(self, owner):
        super().__init__(owner)
        self.results = None

    def solve(self, problem:Problem) -> results.Results:
        pass

    def on_after_dss_built(self):
        pass


class LinearSolver(Solver):
    def solve(self, problem:'Problem') -> results.ResultsStaticLinear:
        problem.reassign_dofs()
        problem.remove_dofs()

        ndofs = 3 * len(problem.nodes)
        free_dofs = problem.free_dofs()
        stiffness_matrix = problem.K(reduced=True)
        loads = problem.loads[free_dofs]

        reduced_displacements = np.linalg.solve(stiffness_matrix, loads)
        displacements = np.zeros(ndofs)
        displacements[free_dofs] = reduced_displacements

        for node in problem.nodes:
            node.displacements = displacements[node.dofs]

        # To be removed:
        problem.displacements = displacements
        problem.upd_obj_displacements() # To be removed

        return results.ResultsStaticLinear(problem, displacements)

    def get_functions(self) -> Dict[str, Callable]:
        return {'Linear' : lambda : self.solve(self.dss.problem.clone())}


class NonLinearSolver(Solver):
    def solve(self, problem:Problem) -> results.ResultsStaticNonlinear:
        steps = 800
        arclength = 45
        A = 0
        max_it = 35

        def get_residual(q, a):
            return self.get_internal_forces(problem)[free_dofs] - q*a

        problem.reassign_dofs()
        problem.remove_dofs()
        free_dofs = problem.free_dofs()

        max_A = np.linalg.norm(problem.loads[free_dofs])
        q = problem.loads[free_dofs] / max_A
        displacements = np.zeros(len(problem.nodes) * 3)
        loads = np.zeros_like(displacements)

        #displ_storage = np.zeros((steps, 3 * len(problem.nodes)))
        displ_storage = []
        force_storage = []

        i = 0
        while i < steps and A < max_A:
            print("Predictor step {}".format(i))
            K = problem.K(True)
            wq0 = np.linalg.solve(K, q)
            f = np.sqrt(1 + wq0 @ wq0)

            sign = np.sign(wq0 @ v0 if i > 1 else 1)
            dA = arclength / f * sign
            v0 = dA*wq0
            A += dA

            displacements[free_dofs] = displacements[free_dofs] + v0
            for node in problem.nodes:
                node.displacements = displacements[node.dofs]

            # Corrector
            k = 0
            residual = get_residual(q, A)
            while np.linalg.norm(residual) > 1e-4 and k < max_it:
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

            #displ_storage[i] = displacements
            displ_storage.append(np.array(displacements))
            loads[free_dofs] = q*A
            force_storage.append(A)
            i += 1

            print(f"Finished in {k} corrector steps (stepped {dA}, arclength {arclength})")
            if k < 4:
                arclength *= 1.05
            elif k > 8:
                arclength *= 0.6
            else:
                arclength *= 0.85


        print(np.asarray(displ_storage))
        print("Ended at A = {}".format(A))

        return results.ResultsStaticNonlinear(problem, np.asarray(displ_storage),
                                              np.asarray(force_storage))

    def get_internal_forces(self, problem):
        ndofs = 3 * len(problem.nodes)
        forces = np.zeros(ndofs)
        for element in problem.elements:
            forces = forces + element.expand(element.get_forces(), ndofs)

        return forces

    def get_functions(self) -> Dict[str, Callable]:
        return {'Nonlinear': lambda : self.solve(self.dss.problem.clone())}


class ModalSolver(Solver):
    def __init__(self, owner):
        super().__init__(owner)

        self.eigenvalues = np.zeros(0)
        self.eigenvectors = np.zeros((0,0))

    def solve(self, problem:Problem):
        problem.reassign_dofs()
        M = problem.M(True)
        K = problem.K(True)
        ndofs = 3 * len(problem.nodes)
        free_dofs = problem.free_dofs()
        full_eigenvectors = np.zeros((len(free_dofs), ndofs))

        # Unsymmetric reduction
        A = np.linalg.solve(M, K)

        # Symmetry preserving reduction
        #L = np.linalg.cholesky(M)
        #A = np.linalg.inv(L) @ K @ np.linalg.inv(L.T)

        eigenvalues, eigenvectors = np.linalg.eig(A)

        eigenvectors = eigenvectors.T # Row-major
        eigenvectors = eigenvectors[eigenvalues.argsort()]
        eigenvalues.sort()
        full_eigenvectors[:,free_dofs] = eigenvectors

        return results.ResultsModal(problem, eigenvalues, full_eigenvectors)

    def get_functions(self) -> Dict[str, Callable]:
        return {'Modal': lambda: self.solve(self.dss.problem.clone())}

