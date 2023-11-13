import time
import numpy as np
from typing import Dict, Callable, Iterable, Optional

from guis.tkinter.plugin_base import DSSPlugin
from core.problem import Problem
import core.results as results
import core.settings as settings


class Solver(DSSPlugin):
    instantiate = True

    def __init__(self, owner):
        super().__init__(owner)
        self.results = None

    def solve(self) -> Iterable[Optional[results.Results]]:
        pass

class LinearSolver(Solver):
    def solve(self) -> Iterable[Optional[results.Results]]:
        problem = self.dss.problem
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
        problem.upd_obj_displacements()  # To be removed

        yield results.ResultsStaticLinear(problem, displacements)

class NonLinearSolver(Solver):
    def solve(self) -> results.ResultsStaticNonlinear:
        problem = self.dss.problem
        steps = 500
        arclength = 45
        A = 0
        max_it = 35
        t0 = time.time()

        problem.reassign_dofs()
        problem.remove_dofs()
        free_dofs = problem.free_dofs()

        max_A = np.linalg.norm(problem.loads[free_dofs])
        q = problem.loads[free_dofs] / max_A
        displacements = np.zeros(len(problem.nodes) * 3)
        loads = np.zeros_like(displacements)

        displ_storage = []
        force_storage = []

        i = 0
        while i < steps and A < max_A:
            if settings.get_setting('dss.verbose', False):
                print("Predictor step {}".format(i))
            K = problem.K(True)
            wq0 = np.linalg.solve(K, q)
            f = np.sqrt(1 + wq0 @ wq0)

            sign = np.sign(wq0 @ v0 if i > 1 else 1)
            dA = arclength / f * sign
            v0 = dA * wq0
            A += dA

            displacements[free_dofs] = displacements[free_dofs] + v0
            for node in problem.nodes:
                node.displacements = displacements[node.dofs]

            # Corrector
            k = 0
            residual = self.get_internal_forces(problem)[free_dofs] - q * A
            while np.linalg.norm(residual) > 1e-3 and k < max_it:
                K = problem.K(True)
                wq = np.linalg.solve(K, q)
                wr = np.linalg.solve(K, -residual)
                dA_ = -wq @ wr / (1 + wq @ wq)
                A += dA_

                displacements[free_dofs] = displacements[free_dofs] + (wr + dA_ * wq)
                for node in problem.nodes:
                    node.displacements = displacements[node.dofs]
                k += 1

                residual = self.get_internal_forces(problem)[free_dofs] - q * A

            # displ_storage[i] = displacements
            displ_storage.append(np.array(displacements))
            loads[free_dofs] = q * A
            force_storage.append(A)
            yield None
            i += 1

            # print(f"Finished in {k} corrector steps (stepped {dA}, arclength {arclength})")
            if k < 4:
                arclength *= 1.05
            elif k > 8:
                arclength *= 0.6
            else:
                arclength *= 0.85

        t1 = time.time()
        dt = t1 - t0
        if settings.get_setting('dss.verbose', False):
            print(np.asarray(displ_storage))
            print(f"Ended at A = {A} in {dt} seconds")

        yield results.ResultsStaticNonlinear(problem, np.asarray(displ_storage),
                                              np.asarray(force_storage))

    def get_internal_forces(self, problem):
        ndofs = 3 * len(problem.nodes)
        forces = np.zeros(ndofs)
        for element in problem.elements:
            forces[element.dofs] += element.get_forces()
        #    forces = forces + element.expand(element.get_forces(), ndofs)

        return forces

class ModalSolver(Solver):
    def __init__(self, owner):
        super().__init__(owner)

        self.eigenvalues = np.zeros(0)
        self.eigenvectors = np.zeros((0, 0))

    def solve(self):
        problem = self.dss.problem
        problem.reassign_dofs()
        M = problem.M(True)
        K = problem.K(True)
        ndofs = 3 * len(problem.nodes)
        free_dofs = problem.free_dofs()
        full_eigenvectors = np.zeros((len(free_dofs), ndofs))

        # Unsymmetric reduction
        A = np.linalg.solve(M, K)

        # Symmetry preserving reduction
        # L = np.linalg.cholesky(M)
        # A = np.linalg.inv(L) @ K @ np.linalg.inv(L.T)

        eigenvalues, eigenvectors = np.linalg.eig(A)

        eigenvectors = eigenvectors.T  # Row-major
        eigenvectors = eigenvectors[eigenvalues.argsort()]
        eigenvalues.sort()
        full_eigenvectors[:, free_dofs] = eigenvectors

        return results.ResultsModal(problem, eigenvalues, full_eigenvectors)

class DynamicSolver(Solver):
    def __init__(self, owner):
        super().__init__(owner)

    def load_by_structural_weight(self, problem):
        gravity = 9810
        for node in problem.nodes:
            mass = 0
            for element in node._elements:
                mass += np.linalg.norm(
                    element.r2 - element.r1) * element.A * 7.86e-9  # Density of steel in tonnes / mm^3
            node.loads += np.array([0, -mass * gravity * 0.5, 0])

    def get_internal_forces(self, problem):
        ndofs = 3 * len(problem.nodes)
        forces = np.zeros(ndofs)
        for element in problem._elements:
            forces[element.dofs] += element.get_forces()

        return forces

    def solve(self):
        return self.solve_explicit(self.dss.problem)

    def solve_explicit(self, problem):
        problem.reassign_dofs()
        problem.remove_dofs()
        dt = 1e-4
        ndofs = 3 * len(problem.nodes)
        num_steps = 50
        free_dofs = problem.free_dofs()
        ndofs_free = len(free_dofs)

        self.load_by_structural_weight(problem)

        f = problem.loads[free_dofs]
        u = np.zeros(ndofs)

        q = np.zeros(2 * ndofs_free)

        K = problem.K(True)
        M = problem.M(True)
        C = 0.01 * K

        O = np.zeros_like(M)
        B = np.hstack((np.zeros(ndofs_free), f))

        disp_history = []

        t = 0
        i = 0
        tmax = 2
        num_timesteps = tmax / dt
        interval = int(num_timesteps / num_steps)
        while t < tmax:
            K = problem.K(True)

            B[ndofs_free:] = f - self.get_internal_forces(problem)[free_dofs]

            AA = np.block([[-M, O], [O, K]])
            AB = np.block([[O, M], [M, C]])
            A = np.linalg.solve(-AB, AA)

            q = (A@q + B)*dt

            u[free_dofs] = q[ndofs_free:]
            for node in problem.nodes:
                node.displacements = u[node.dofs]

            yield
            t += dt
            i += 1
            if i % interval == 0:
                print(f'Step {i}: t = {np.round(t, 3)}. Displacements {u[free_dofs]}')
                print(f'f={self.get_internal_forces(problem)[free_dofs]}')
                disp_history.append(np.array(u))

        return results.ResultsDynamicTimeIntegration(problem, np.asarray(disp_history))

    def solve_implicit(self, problem):
        problem.reassign_dofs()
        problem.remove_dofs()
        dt = 1e-3
        ndofs = 3 * len(problem.nodes)
        num_steps = 50
        free_dofs = problem.free_dofs()
        ndofs_free = len(free_dofs)

        self.load_by_structural_weight(problem)

        f = problem.loads[free_dofs]
        u = np.zeros(ndofs)

        q = np.zeros(2 * ndofs_free)

        K = problem.K(True)
        M = problem.M(True)
        C = 0.01 * K


        I = np.eye(2 * ndofs_free)
        O = np.zeros_like(M)
        B = np.hstack((np.zeros(ndofs_free), f))

        disp_history = []

        t = 0
        i = 0
        tmax = 2
        num_timesteps = tmax/dt
        interval = int(num_timesteps / num_steps)
        while t < tmax:
            K = problem.K(True)
            B[ndofs_free:] = f - self.get_internal_forces(problem)[free_dofs]

            AA = np.block([[-M, O], [O, K]])
            AB = np.block([[O, M], [M, C]])
            A = np.linalg.solve(-AB, AA)

            q = np.linalg.solve(I - A*dt, q + B*dt)

            u[free_dofs] = q[ndofs_free:]
            for node in problem.nodes:
                node.displacements = u[node.dofs]

            t += dt
            i += 1
            if i % interval == 0:
                print(f'Step {i}: t = {np.round(t, 3)}. Displacements {u[free_dofs]}')
                print(f'f={self.get_internal_forces(problem)[free_dofs]}')
                disp_history.append(np.array(u))

        return results.ResultsDynamicTimeIntegration(problem, np.asarray(disp_history))

    def set_displacements(self, displ, prob):
        for node in prob.nodes:
            node.displacements = displ[node.dofs]
