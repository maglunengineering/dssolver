from typing import *
import time
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
            residual = self.get_internal_forces(problem)[free_dofs] - q*A
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

                residual = self.get_internal_forces(problem)[free_dofs] - q*A

            #displ_storage[i] = displacements
            displ_storage.append(np.array(displacements))
            loads[free_dofs] = q*A
            force_storage.append(A)
            i += 1

            #print(f"Finished in {k} corrector steps (stepped {dA}, arclength {arclength})")
            if k < 4:
                arclength *= 1.05
            elif k > 8:
                arclength *= 0.6
            else:
                arclength *= 0.85

        t1 = time.time()
        dt = t1 - t0
        print(np.asarray(displ_storage))
        print(f"Ended at A = {A} in {dt} seconds")

        return results.ResultsStaticNonlinear(problem, np.asarray(displ_storage),
                                              np.asarray(force_storage))

    def get_internal_forces(self, problem):
        ndofs = 3 * len(problem.nodes)
        forces = np.zeros(ndofs)
        for element in problem.elements:
            forces[element.dofs] += element.get_forces()
        #    forces = forces + element.expand(element.get_forces(), ndofs)

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

class DynamicSolver(Solver):
    def __init__(self, owner):
        super().__init__(owner)

    def load_by_structural_weight(self, problem):
        gravity = 9810
        for node in problem.nodes:
            mass = 0
            for element in node.elements:
                mass += np.linalg.norm(element.r2 - element.r1) * element.A * 7.86e-9 # Density of steel in tonnes / mm^3
            node.loads += np.array([0, -mass*gravity*0.5, 0])

    def get_internal_forces(self, problem):
        ndofs = 3 * len(problem.nodes)
        forces = np.zeros(ndofs)
        for element in problem.elements:
            forces[element.dofs] += element.get_forces()

        return forces

    def solve_explicit(self, problem):
        problem.reassign_dofs()
        problem.remove_dofs()
        dt = 5e-3
        ndofs = 3 * len(problem.nodes)
        num_steps = 25
        free_dofs = problem.free_dofs()
        model_size = problem.model_size()

        for node in problem.nodes:
            node.velocities = np.zeros(3)
            node.accelerations = np.zeros(3)
        self.load_by_structural_weight(problem)

        lbd = problem.elements[0].E
        mu = lbd/2
        cd = np.sqrt((lbd + 2*mu)/7.86e-9)
        dt = 1/np.sqrt(2) * problem.elements[0]._deformed_length() / cd
        dt *= 0.01

        f = problem.loads[free_dofs]
        u = np.zeros(ndofs)
        v = np.zeros(ndofs)
        a = np.zeros(ndofs)

        ur = u[free_dofs]
        vr = v[free_dofs]
        ar = a[free_dofs]

        K = problem.K(True)
        M = problem.M(True)
        C = 0.01 * K
        invM = np.linalg.inv(M)

        disp_history = []
        vel_history = []
        acc_history = []
        dt_history = []

        t = 0
        i = 0
        tmax = 1
        nonlinear = True
        while t < tmax:
            if nonlinear and i % num_steps == 0:
                K = problem.K(True)
                M = problem.M(True)
                invM = np.linalg.inv(M)

            ur, vr, ar =    ur + vr*dt + 0.5*ar*dt**2, \
                            vr + ar*dt, \
                            invM @ (f - K@ur)

            u[free_dofs] = ur
            v[free_dofs] = vr
            a[free_dofs] = ar

            t += dt
            i += 1
            if i % num_steps == 0:
                for node in problem.nodes:
                    node.displacements = u[node.dofs]
                disp_history.append(np.array(u))
                vel_history.append(v)
                acc_history.append(a)

                print(f'Step {i}: Stepped {dt}. Displacements: {ur}')

        return results.ResultsDynamicTimeIntegration(problem, np.asarray(disp_history))

    def solve_implicit(self, problem):
        problem.reassign_dofs()
        problem.remove_dofs()
        dt = 1e-3
        ndofs = 3 * len(problem.nodes)
        num_steps = 200
        free_dofs = problem.free_dofs()
        ndofs_free = len(free_dofs)

        self.load_by_structural_weight(problem)

        f = problem.loads[free_dofs]
        u = np.zeros(ndofs)
        v = np.zeros(ndofs)
        a = np.zeros(ndofs)

        ur = np.zeros(ndofs_free)
        vr = np.zeros(ndofs_free)
        ar = np.zeros(ndofs_free)

        q = np.zeros(2 * ndofs_free)

        K = problem.K(True)
        M = problem.M(True)
        C = 0.01 * K
        invM = np.linalg.inv(M)

        I = np.eye(2 * ndofs_free)
        O = np.zeros_like(M)
        B = np.hstack((np.zeros(ndofs_free), f))

        a_m = 0.3
        a_f = 0.4
        beta = 0.3 + 0.5*(a_f - a_m)
        gamma = 0.5 - a_m + a_f

        disp_history = []

        t = 0
        i = 0
        tmax = 5
        nonlinear = True
        total_work_done = 0
        du = np.zeros(ndofs_free)
        while t < tmax:
            K = problem.K(True)
            #M = problem.M(True)
            #invM = np.linalg.inv(M)
            B[ndofs_free:] = f# - self.get_internal_forces(problem)[free_dofs]

            AA = np.block([[-M, O], [O, K]])

            AB = np.block([[O, M], [M, C]])
            A = np.linalg.solve(-AB, AA)


            qdot = np.linalg.solve(I - dt*A, q + dt*B)
            q[free_dofs] += qdot[:ndofs_free] * dt
            q[ndofs_free:] += qdot[ndofs_free:] * dt

            u[free_dofs] = q[ndofs_free:]
            for node in problem.nodes:
                node.displacements = u[node.dofs]

            t += dt
            i += 1
            if i % num_steps == 0:
                print(f'Step {i}: Stepped {dt}. Displacements {u[free_dofs]}')
                disp_history.append(np.array(u))

        return results.ResultsDynamicTimeIntegration(problem, np.asarray(disp_history))

    def solve_whoknows(self, problem):
        problem.reassign_dofs()
        problem.remove_dofs()
        dt = 1e-3
        ndofs = 3*len(problem.nodes)
        num_steps = 300
        free_dofs = problem.free_dofs()
        ndofs_free = len(free_dofs)

        self.load_by_structural_weight(problem)
        set_displacements = lambda : self.set_displacements(u, problem)

        f = problem.loads[free_dofs]
        u = np.zeros(ndofs)
        v = np.zeros(ndofs)
        a = np.zeros(ndofs)

        ur = np.zeros(ndofs_free)
        vr = np.zeros(ndofs_free)
        ar = np.zeros(ndofs_free)

        K = problem.K(True)
        M = problem.M(True)
        C = 0.01*K
        invM = np.linalg.inv(M)

        disp_history = []

        def get_strain_energy_for(reduced_displacements):
            uu = np.zeros(ndofs)
            uu[free_dofs] = reduced_displacements
            old_disps = {}
            for node in problem.nodes:
                old_disps[node] = node.displacements
                node.displacements = uu[node.dofs]
            strain_energy = sum(e.get_strain_energy() for e in problem.elements)
            for node in problem.nodes:
                node.displacements = old_disps[node]
            return strain_energy


        t = 0
        i = 0
        tmax = 1
        nonlinear = True
        total_work = 0
        strain_energy = 0
        kinetic_energy = 0
        while t < tmax:
            K = problem.K(True)
            M = problem.M(True)
            C = 0.01 * K
            invM = np.linalg.inv(M)

            ar = invM @ (f - K@ur)
            delta_ur = vr*dt + 0.5*ar*dt**2
            ur += delta_ur
            vr += ar * dt
            vr += 0.5*ar*dt

            u[free_dofs] = ur

            set_displacements()

            strain_energy_rate = dt * vr@K@vr
            kinetic_energy_rate = vr@M@ar
            rel_kinetic_energy_rate = kinetic_energy_rate/(kinetic_energy_rate + strain_energy_rate)

            diff_work = f @ delta_ur
            total_work += diff_work

            # Diff ratio must equal rate ratio from before the step
            target_kinetic_energy = kinetic_energy + rel_kinetic_energy_rate * diff_work
            target_strain_energy = strain_energy + (1 - rel_kinetic_energy_rate) * diff_work

            # Iterate on the force to get the correct strain energy
            residual = self.get_internal_forces(problem)[free_dofs]
            residual_acceleration = - invM @ residual

            j = 1e-6
            dj = 1e-8
            while not np.isclose(strain_energy, target_strain_energy):
                derivative = (get_strain_energy_for(ur + (j+dj)*residual_acceleration) - get_strain_energy_for(ur + j*residual_acceleration)) / dj
                j += (target_strain_energy - strain_energy) / derivative
                u[free_dofs] = ur + j*residual_acceleration
                total_work += f @ (j * residual_acceleration)
                set_displacements()
                strain_energy = sum(e.get_strain_energy() for e in problem.elements)

            ur = u[free_dofs]
            while kinetic_energy and not np.isclose(kinetic_energy, target_kinetic_energy):
                ratio = kinetic_energy / target_kinetic_energy
                vr /= np.sqrt(ratio)
                kinetic_energy = 0.5 * vr.T @ M @ vr

            strain_energy = sum(e.get_strain_energy() for e in problem.elements)
            kinetic_energy = 0.5*vr@M@vr

            print(f'Work done: {total_work}, Strain energy: {strain_energy}, kinetic energy: {kinetic_energy}')

            disp_history.append(np.array(u))
            t += dt

        return results.ResultsDynamicTimeIntegration(problem, np.asarray(disp_history))

    def set_displacements(self, displ, prob):
        for node in prob.nodes:
            node.displacements = displ[node.dofs]

    def get_functions(self) -> Dict[str, Callable]:
        return {'Explicit time integration' : lambda: self.solve_explicit(self.dss.problem.clone()),
                'Implicit time integration' : lambda: self.solve_implicit(self.dss.problem.clone()),
                'WIP algorithm' : lambda : self.solve_whoknows(self.dss.problem.clone())}