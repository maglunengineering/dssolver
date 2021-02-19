import unittest
import time
import problem
import solvers

from elements import *

class ElementTest(unittest.TestCase):
    def setUp(self):
        self.p = problem.Problem()
        self.n1 = Node((0, 0))
        self.n2 = Node((1000, 0))

        self.rod = Rod(self.n1, self.n2)
        self.beam = Beam(self.n1, self.n2)

    def test_should_have_same_axial_stiffness_with_no_displacements(self):

        for indices in ([0,0],[0,3],[3,0],[3,3]):
            i,j = indices
            rod_k00 = self.rod.stiffness_matrix_global()[i,j]
            beam_k00 = self.beam.stiffness_matrix_global()[i,j]

            self.assertEqual(rod_k00, beam_k00)

    def test_should_have_same_axial_stiffness_with_same_axial_displacements(self):
        self.n2.displacements = np.array([-10, 0, 0])

        for indices in ([0,0],[0,3],[3,0],[3,3]):
            i,j = indices
            rod_k00 = self.rod.stiffness_matrix_global()[i,j]
            beam_k00 = self.beam.stiffness_matrix_global()[i,j]

            self.assertEqual(rod_k00, beam_k00)

    def test_should_have_same_forces_with_same_axial_displacements(self):
        self.n2.displacements = np.array([-10, 0, 0])

        self.assertTrue(np.allclose(self.rod.get_forces(), self.beam.get_forces()))

    def test_strain_energy_should_be_force_times_distance(self):
        self.p.nodes.append(self.n1)
        self.p.nodes.append(self.n2)
        self.p.create_beam(self.n1, self.n2)
        self.p.fix(self.n1)
        f = 10
        self.p.load_node(self.n2, np.array([0, -f, 0]))
        self.p.reassign_dofs()
        self.p.remove_dofs()
        solver = solvers.LinearSolver(None)
        solver.solve(self.p)
        work = -self.n2.displacements[1] * f * 0.5

        self.assertAlmostEqual(work, self.p.elements[0].get_strain_energy(), places=2)

    def test_strain_energy_should_be_uTku(self):
        self.p.nodes.append(self.n1)
        self.p.nodes.append(self.n2)
        self.p.create_beam(self.n1, self.n2)
        self.p.fix(self.n1)
        f = 5
        self.p.load_node(self.n2, np.array([0, -f, 0]))
        self.p.reassign_dofs()
        self.p.remove_dofs()
        solver = solvers.LinearSolver(None)
        solver.solve(self.p)

        e = self.p.elements[0]
        u = e.get_displacements()
        k = e.stiffness_matrix_global()
        work = e.get_strain_energy()

        self.assertAlmostEqual(work, 0.5*u.T@k@u, places=2)


class ProblemTest(unittest.TestCase):
    def setUp(self) -> None:
        self.p = problem.Problem()
        self.n1 = Node((0, 0))
        self.n2 = Node((1000, 0))
        self.p.nodes.append(self.n1)
        self.p.nodes.append(self.n2)

    def test_dynamic_timeint_pendulum(self):
        rod = self.p.create_rod(self.n1, self.n2)
        self.p.pin(self.n1)
        solver = solvers.DynamicSolver(None)
        solver.solve_whoknows(self.p)


def timeit(func, *args):
    def inner(*args):
        t0 = time.time()
        func(*args)
        t1 = time.time()
        print(f'Finished {func.__name__} in {t1 - t0} seconds')
    return inner


class PerformanceTest(unittest.TestCase):
    def setUp(self):
        n = 1000
        self.matrix = np.random.random((n, n))
        self.vec1 = np.random.random(n)
        self.vec2 = np.random.random(n)

    @timeit
    def test_timeit_onebyone(self):
        sol1 = np.linalg.solve(self.matrix, self.vec1)
        sol2 = np.linalg.solve(self.matrix, self.vec2)

    @timeit
    def test_timeit_both(self):
        sols = np.linalg.solve(self.matrix, np.array((self.vec1, self.vec2)).T)

