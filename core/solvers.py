import time
from functools import reduce
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

    def solve(self) -> Optional[results.Results]:
        pass

    def solveall(self) -> results.Results:
        return self.solve()

class LinearSolver(Solver):
    def solve(self) -> Optional[results.Results]:
        return self.problem.solve()

class NonLinearSolver(Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self.arclength = kwargs.get('arclength', 45)
        self.iteration_mode = kwargs.get('iteration_mode', 0)
        self.multi_run = kwargs.get('multi_run', 0) # Multirun: Same load case scaled differently
        self.max_iter = kwargs.get('max_iter', 25)

    def solve(self)-> results.ResultsStaticNonlinear:
        item = None
        for _item in self._solve_impl():
            item = _item
        return _item

    def solve_iter(self) -> Iterable:
        return self._solve_impl()

    def _solve_impl(self):
        verbose = settings.get_setting('dss.verbose', 0)

        p = self.problem
        p.reassign_dofs()
        num_lc = max(n.loads.shape[0] if n.loads.ndim > 1 else 1 for n in p.nodes)
        ndofs = sum((n.ndofs() for n in p.nodes))
        disp_final = np.zeros((num_lc, ndofs))
        k = 0

        target_load_all = p.assemble_vector(p.nodes, lambda n:n.loads, ndofs).reshape((num_lc, ndofs))

        disp_histories = []
        load_histories = []

        arclength = self.arclength
        for i_lc in range(num_lc):
            A = np.array([0.0])

            p.reassign_dofs()
            p.remove_dofs()
            free_dofs = p.free_dofs()
            ix = np.ix_(free_dofs, free_dofs)

            target_load = target_load_all[i_lc, free_dofs]
            if i_lc == 0:
                accumulated_load = np.zeros(1)
                current_load = np.zeros(len(free_dofs)) # Otherwise, just let it continue on
            else:
                accumulated_load = accumulated_load + current_load

            max_A = np.linalg.norm(target_load - accumulated_load)
            q = ((target_load - accumulated_load) / max_A).flatten()

            displ_storage = [np.zeros(ndofs)]
            force_storage = [0.0]
            displacements = np.zeros(ndofs)

            if self.multi_run > 1:
                displacements = np.zeros((self.multi_run, ndofs))

            i = 0
            while A.max() < 0.999*max_A:
                if verbose:
                    print(f"Predictor step {i}")

                #p.nonlin_update(i_lc, displacements)

                K = p.K(displacements)[..., ix[0], ix[1]]
                wq0 = np.linalg.solve(K, q)
                f = np.sqrt(1 + (wq0 * wq0).sum(-1)) # wq0@wq0 if 1d. This works also for (n,ndofs) wq0

                sign = np.sign((wq0 * v0).sum(-1)) if i > 1 else np.ones_like(f)
                dA = (arclength / f * sign)[..., np.newaxis]
                if self.multi_run > 1:
                    dA = dA * np.logspace(-1, 1, self.multi_run, dtype=float).reshape((-1,1))
                dA = reduce(np.minimum, (0.1*max_A, dA, max_A - A))

                v0 = dA * wq0
                A = A + dA

                displacements[..., free_dofs] = (displacements[..., free_dofs] + v0)

                # Corrector
                residual = self.get_internal_forces(p, displacements, ndofs)[..., free_dofs] - q * A - accumulated_load

                if self.iteration_mode >= 1:
                    yield {'displacements':displacements, 'residual':residual, 'control_param':A, 'last_num_iter':k}


                for k in range(self.max_iter):
                    K = p.K(displacements)[..., ix[0], ix[1]]
                    wq = np.linalg.solve(K, q)
                    wr = np.linalg.solve(K, -residual[..., np.newaxis])[...,0]
                    dA_ = -((wq * wr) / (1 + (wq * wq))).sum(-1)[..., np.newaxis]
                    A = A + dA_

                    du = wr + dA_ * wq
                    displacements[..., free_dofs] = (displacements[..., free_dofs] + du)
                    dw = np.abs((du * residual).sum(-1))

                    if self.iteration_mode >= 2:
                        residual = self.get_internal_forces(p, displacements, ndofs)[..., free_dofs] - q * A - accumulated_load
                        yield {'displacements':displacements, 'residual':residual, 'control_param':A}

                    criterion = 1e-4/ndofs
                    if self.multi_run > 1:
                        cur_mr_idx = np.count_nonzero(np.where(dw<criterion, dw, 0.0)) - 1
                        if np.all(dw < criterion):
                            break
                    elif dw < criterion:
                        break # If we are in multirun, run max_iter iterations regardless and pick the one

                    residual = self.get_internal_forces(p, displacements, ndofs)[..., free_dofs] - q * A - accumulated_load

                else:
                    # We didn't converge
                    if self.multi_run <= 1:
                        if len(displ_storage) > 1:
                            displacements = displ_storage.pop()
                            A = np.asarray(force_storage.pop())
                        else:
                            displacements = displ_storage[0]
                            A = np.asarray(force_storage[0])

                        arclength /= 2
                        if verbose:
                            print(f'Resetting displacements and split arclength. {arclength=} {A=}')
                        continue
                    elif cur_mr_idx < 0:
                        raise ValueError('Multi-run failed')

                if verbose:
                    print(f'Increasing arclength')
                arclength *= 1.2

                if self.multi_run > 1:
                    displ_storage.append(displacements[cur_mr_idx])
                    force_storage.append(A[cur_mr_idx])
                    displacements[:, ...] = displacements[cur_mr_idx, ...]
                    A[:, ...] = A[cur_mr_idx, ...]

                else:
                    displ_storage.append(displacements.flatten().copy())
                    force_storage.append(A.flatten()[0])

                current_load = q * A + accumulated_load

                i += 1

            # End of i_lc
            if verbose:
                print(f'Finished {i_lc=}/{num_lc}')

            if self.multi_run > 1:
                disp_final[i_lc] = displacements[cur_mr_idx]
            else:
                disp_final[i_lc] = displacements

            disp_histories.append(np.asarray(displ_storage))
            load_histories.append(np.asarray(force_storage))
            


        force_storage[0] = np.zeros_like(force_storage[1])
        yield results.ResultsStaticNonlinear(p, disp_histories, load_histories)

    def get_internal_forces(self, problem, displacements, ndofs):
        return problem.assemble_vector(problem.elements, lambda e: e.get_external_forces(displacements), ndofs)

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