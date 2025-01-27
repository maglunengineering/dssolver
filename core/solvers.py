import time
import numpy as np
from typing import Dict, Callable, Iterable, Optional

import core.results as results
import core.settings as settings
import core.problem as problem


class Solver:
    instantiate = True

    def __init__(self, problem:problem.Problem):
        self.problem:problem.Problem = problem
        self.results = None

    def solve(self) -> Iterable[Optional[results.Results]]:
        pass

    def solveall(self) -> results.Results:
        for x in self.solve():
            if x:
                return x

class LinearSolver(Solver):
    def solve(self) -> Iterable[Optional[results.Results]]:
        yield self.problem.solve()

class NonLinearSolver(Solver):
    def solve(self) -> results.ResultsStaticNonlinear:
        p = self.problem
        num_lc = max(n.loads.shape[1] if n.loads.ndim > 1 else 1 for n in p.nodes)
        ndofs = sum(node.ndofs() for node in self.problem.nodes)

        arclength = 1000
        for i_lc in range(num_lc):

            A = 0
            max_it = 15
            t0 = time.time()

            p.reassign_dofs()
            p.remove_dofs()
            free_dofs = p.free_dofs()

            target_load = p.assemble_vector(p.nodes, lambda n:n.loads, ndofs)[i_lc, free_dofs]
            displacements = p.assemble_vector(p.nodes, lambda n:n.displacements, ndofs)[i_lc, :]

            if i_lc == 0:
                accumulated_load = np.zeros(1)
                current_load = np.zeros(len(free_dofs)) # Otherwise, just let it continue on
            else:
                accumulated_load = accumulated_load + current_load

            max_A = np.linalg.norm(target_load - accumulated_load)
            q = (target_load - accumulated_load) / max_A

            displ_storage = [np.array(displacements)]
            force_storage = [A]

            i = 0
            while A < max_A:
                if settings.get_setting('dss.verbose', False):
                    print(f"Predictor step {i}")

                p.nonlin_update(i_lc)
                K = p.K()[np.ix_(free_dofs, free_dofs)]

                wq0 = np.linalg.solve(K, q)
                f = np.sqrt(1 + wq0 @ wq0)

                sign = np.sign(wq0 @ v0) if i > 1 else 1
                dA = arclength / f * sign
                dA = min(0.1*max_A, dA, max_A - A)
                v0 = dA * wq0
                A += dA

                displacements[free_dofs] = displacements[free_dofs] + v0
                for node in p.nodes:
                    node.displacements[i_lc:, :node.ndofs()] = displacements[node.dofs].reshape((1,-1))

                # Corrector
                p.nonlin_update(i_lc)
                residual = self.get_internal_forces(p, i_lc, ndofs)[free_dofs] - q * A - accumulated_load
                for k in range(max_it):
                    K = p.K()[np.ix_(free_dofs, free_dofs)]
                    wq = np.linalg.solve(K, q)
                    wr = np.linalg.solve(K, -residual)
                    dA_ = -wq @ wr / (1 + wq @ wq)
                    A += dA_

                    displacements[free_dofs] = displacements[free_dofs] + (wr + dA_ * wq)
                    for node in p.nodes:
                        node.displacements[:node.ndofs(),i_lc:] = displacements[node.dofs].reshape((-1,1))

                    p.nonlin_update(i_lc)
                    residual = self.get_internal_forces(p, i_lc, ndofs)[free_dofs] - q * A - accumulated_load
                    if np.linalg.norm(residual) < 1e-3:
                        break
                else:
                    if len(displ_storage) > 1:
                        displacements = displ_storage.pop()
                        A = force_storage.pop()
                    else:
                        displacements = displ_storage[0]
                        A = force_storage[0]

                    for node in p.nodes:
                        node.displacements[:node.ndofs(),i_lc:] = displacements[node.dofs].reshape((-1,1))
                    arclength /= 2
                    if settings.get_setting('dss.verbose', False):
                        print(f'Resetting displacements and split arclength. {arclength=} {A=}')
                    continue

                if settings.get_setting('dss.verbose', False):
                    print(f'Increasing arclength')
                arclength *= 1.2

                displ_storage.append(np.array(displacements))
                current_load = q * A + accumulated_load
                force_storage.append(A)
                yield None
                i += 1

            t1 = time.time()
            dt = t1 - t0
            if settings.get_setting('dss.verbose', False):
                print(np.asarray(displ_storage))
                print(f"Ended at A = {A} in {dt} seconds")

            yield results.ResultsStaticNonlinear(p, np.asarray(displ_storage),
                                                  np.asarray(force_storage))

    def get_internal_forces(self, problem, i_lc, ndofs):
        return problem.assemble_vector(problem.elements, lambda e: e.get_forces(), ndofs)[i_lc, :]

class ModalSolver(Solver):
    def __init__(self, owner):
        super().__init__(owner)

        self.eigenvalues = np.zeros(0)
        self.eigenvectors = np.zeros((0, 0))

    def solve(self):
        p = self.problem
        p.reassign_dofs()
        free_dofs = p.free_dofs()

        M = p.M()[np.ix_(free_dofs, free_dofs)]
        K = p.K()[np.ix_(free_dofs, free_dofs)]
        ndofs = 3 * len(p.nodes)

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

        return results.ResultsModal(p, eigenvalues, full_eigenvectors)