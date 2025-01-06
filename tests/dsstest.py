import unittest
import time
from core import problem, solvers

from core.elements import *

def profile(func):
    """
    Decorator that causes the decorated function to be run using cProfile. Output is printed to stdout
    """
    def inner(*args, **kwargs):
        import cProfile
        cProfile.runctx('func(self)', locals={'self':args[0], 'func':func}, globals=globals())
    return inner


class ElementTest(unittest.TestCase):
    def setUp(self):
        self.p = problem.Problem()
        self.n1 = Node((0, 0))
        self.n2 = Node((1000, 0))

    def test_should_have_same_axial_stiffness_with_no_displacements(self):
        self.rod = Rod(self.n1, self.n2)
        self.beam = Beam(self.n1, self.n2)
        for indices in ([0,0],[0,3],[3,0],[3,3]):
            i,j = indices
            rod_k00 = self.rod.stiffness_matrix_global()[i,j]
            beam_k00 = self.beam.stiffness_matrix_global()[i,j]

            self.assertEqual(rod_k00, beam_k00)

    def test_should_have_same_axial_stiffness_with_same_axial_displacements(self):
        self.rod = Rod(self.n1, self.n2)
        self.beam = Beam(self.n1, self.n2)
        self.n2.displacements = np.array([-10, 0, 0])

        for indices in ([0,0],[0,3],[3,0],[3,3]):
            i,j = indices
            rod_k00 = self.rod.stiffness_matrix_global()[i,j]
            beam_k00 = self.beam.stiffness_matrix_global()[i,j]

            self.assertEqual(rod_k00, beam_k00)

    def test_should_have_same_forces_with_same_axial_displacements(self):
        self.rod = Rod(self.n1, self.n2)
        self.beam = Beam(self.n1, self.n2)
        self.n2.displacements = np.array([-10, 0, 0])

        self.assertTrue(np.allclose(self.rod.get_forces(None), self.beam.get_forces(None)))

    def test_beam_with_preload_should_give_preload_with_no_load(self):
        self.beam = Beam(self.n1, self.n2)
        self.beam.preload = np.array([1000, 0, 0, -1000, 0, 0])

        self.assertTrue(np.allclose(-self.beam.preload, self.beam.get_forces_local_lin(None)))

    def test_strain_energy_should_be_force_times_distance(self):
        self.rod = Rod(self.n1, self.n2)
        self.beam = Beam(self.n1, self.n2)
        self.p.nodes.append(self.n1)
        self.p.nodes.append(self.n2)
        self.p.create_beam(self.n1, self.n2)
        self.n1.fix()
        f = 10
        self.n2.loads = np.array([0, -f, 0])
        self.p.reassign_dofs()
        self.p.remove_dofs()
        solver = solvers.LinearSolver(self.p)
        res = next(iter(solver.solve()))
        work = -self.n2.get_displacements(res.displacements)[1,0] * f * 0.5

        self.assertAlmostEqual(work, self.p.elements[0].get_strain_energy(res.displacements)[0], places=2)

    def test_strain_energy_should_be_uTku(self):
        self.rod = Rod(self.n1, self.n2)
        self.beam = Beam(self.n1, self.n2)
        self.p.nodes.append(self.n1)
        self.p.nodes.append(self.n2)
        self.p.create_beam(self.n1, self.n2)
        self.n1.fix()
        f = 5
        self.n2.loads = np.array([0, -f, 0])
        self.p.reassign_dofs()
        self.p.remove_dofs()
        solver = solvers.LinearSolver(self.p)
        res = next(iter(solver.solve()))

        e = self.p.elements[0]
        u = e.get_displacements(res.displacements).flatten()
        k = e.stiffness_matrix_global()
        work = e.get_strain_energy(res.displacements)[0]

        self.assertAlmostEqual(work, 0.5*u.T@k@u, places=2)

    def test_quad4_should_compute_monkeypatched(self):
        self.n3 = Node((1000, 1000))
        self.n4 = Node((0, 1000))
        self.quad = Quad4(self.n1, self.n2, self.n3, self.n4, 210e3, 0.3, 1)
        Quad4.stiffness_matrix_global = lambda *args,**kwargs: np.random.random((8,8))
        self.n1.pin()
        self.n2.pin()
        self.n3.loads = np.array([1000, 0, 0])
        self.p.nodes.extend((self.n1, self.n2, self.n3, self.n4))
        self.p.elements.append(self.quad)

        self.p.solve()

    def test_quad4_should_compute_real(self):
        self.n3 = Node((1000, 1000))
        self.n4 = Node((0, 1000))
        self.quad = Quad4(self.n1, self.n2, self.n3, self.n4, 210e3, 0.3, 1)
        self.n1.pin()
        self.n2.pin()
        self.n3.loads = np.array([1000, 0])
        self.p.nodes.extend((self.n1, self.n2, self.n3, self.n4))
        self.p.elements.append(self.quad)

        self.p.solve()

class ProblemTest(unittest.TestCase):
    def setUp(self) -> None:
        self.p = problem.Problem()
        self.n1 = Node((0, 0))
        self.n2 = Node((1000, 0))
        self.p.nodes.append(self.n1)
        self.p.nodes.append(self.n2)

    def _test_dynamic_timeint_pendulum(self): # Uncomment leading _ to run
        rod = self.p.create_rod(self.n1, self.n2)
        self.n1.pin()
        solver = solvers.DynamicSolver(None)
        solver.solve_whoknows(self.p)

    def test_prescribed_displacements(self):
        self.n3 = Node((2000, 0))
        self.p.nodes.append(self.n3)
        self.n1.fix()
        self.n3.pin()
        self.p.create_beam(self.n1, self.n2)
        self.p.create_beam(self.n2, self.n3)
        self.n3.prescribed_displacements = np.array([0, 100]).reshape((-1,1))
        self.p.solve()
        disp = self.p.displacements

        # ??? Should this not be 25 (1/4) if this is a parabola? Whatevs
        self.assertTrue(np.allclose(self.n2.get_displacements(disp)[0:2, 0], np.array([0, 31.25])),
                        f'{self.n2.get_displacements(disp)[0:2, 0]} != {np.array([0, 25])}')

    def test_load_cases(self):
        loads = np.zeros((3, 2))
        # Conceptually: (n, rows, cols) Two vectors with 3 rows and 1 column. Will go on node 2
        loads[:, 0] = np.array([0, -1000, 0])
        loads[:, 1] = np.array([0, -1500, 0])
        self.n2.loads = loads

        self.n1.fix()
        self.p.create_beam(self.n1, self.n2)
        self.p.solve()

        disp_lc1 = self.p.displacements[:, 0]
        disp_lc2 = self.p.displacements[:, 1]

        disp_lc1_z = disp_lc1[-2]
        disp_lc2_z = disp_lc2[-2]

        self.assertNotEqual(disp_lc1_z, 0.0)
        self.assertNotEqual(disp_lc2_z, 0.0)
        self.assertAlmostEqual(1.5, disp_lc2_z / disp_lc1_z, places=8)


    def test_constraint_penalty(self):
        n3 = Node((2000,0))
        n4 = Node((3000,0))
        self.p.nodes.append(n3)
        self.p.nodes.append(n4)

        for n1,n2 in zip(self.p.nodes, self.p.nodes[1:]):
            self.p.create_beam(n1, n2, E=2e5, A=1000)

        self.n1.fix()
        n4.loads = np.array([-1000, 0, 0])

        self.p.solve()
        self.assertAlmostEqual(-1000/(2e5*1000/3000), n4.displacements[0, 0], places=5)

        self.p.elements.append(PenaltyBeam(self.n2, n3))
        self.p.solve()
        self.assertAlmostEqual(-1000 / (2e5 * 1000 / 2000), n4.displacements[0, 0], places=5)

    def _test_constraint_penaltypreload(self):
        n3 = Node((2000,0))
        n4 = Node((3000,0))
        self.p.nodes.append(n3)
        self.p.nodes.append(n4)

        for n1,n2 in zip(self.p.nodes, self.p.nodes[1:]):
            self.p.create_beam(n1, n2, E=2e5, A=1000)

        self.n1.fix()
        n4.loads = np.array([-1000, 0, 0])

        self.p.solve()
        self.assertAlmostEqual(-1000/(2e5*1000/3000), n4.displacements[0, 0], places=5)

        #self.p.elements.append(PreloadPenaltyBeam(self.n2, n3)) # Not implemented
        self.p.solve()
        self.assertAlmostEqual(-1000/(2e5*1000/2000), n4.displacements[0, 0], places=5)


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

    @timeit
    def test_270_arc(self):
        # 5/11-23: 20.2 s
        # 17/11-23: 17s
        # 18/11-23: 11s
        # 19/11-23: 9s, 6s, back to 12?
        # 22/1-25: 2.1s, new cpu
        p = problem.Problem()
        start = np.deg2rad(225)
        end = np.deg2rad(-45)
        n = 31
        node_angles = np.linspace(start, end, n)
        node_points = 500 * np.array([np.cos(node_angles), np.sin(node_angles)]).T + np.array([0, 500])

        for r1, r2 in zip(node_points, node_points[1:]):
            p.create_beam(r1, r2)

        p.nodes[0].pin()
        p.nodes[-1].fix()
        p.nodes[n//2].loads = np.array([0, -200000, 0]).reshape((3,1))
        self.problem = p
        solver = solvers.NonLinearSolver(p)
        res = solver.solveall()

        self.assertAlmostEqual(-1254.63, res.displacements[-1].min(), delta=10)

class SampleProblems(unittest.TestCase):
    def setUp(self):
        self.p = problem.Problem()

    def curved_spring(self, ampl):
        length = 10.5 #mm

        # Half sine centered at (l/2, 0)
        xx = np.linspace(0, length, 21)
        yy = np.sin(xx / xx.max() * np.pi) * ampl


        E = 1500 # MPa
        h = 5
        t = 0.07
        I = 1/12 * h * t**3

        for x1, x2, y1, y2 in zip(xx, xx[1:], yy, yy[1:]):
            self.p.create_beam(np.array((x1,y1)) ,np.array((x2,y2)), E=E, A=t*h, I=I, z=t/2)

        self.p.nodes[0].roller()
        self.p.nodes[-1].pin()
        self.p.nodes[0].loads = np.array([0.01, 0, 0])

        res, = self.p.solve()
        print(f'Amplitude: {ampl} - Displacement: {res.displacements[0]}')

        #self.p.plot()
        #plt.show()

    def test_curved_spring(self):
        for i in range(5):
            self.p = problem.Problem()
            self.curved_spring(i)

    def test_cantilever_beam(self):
        P = 1000
        L = 1000
        E = 210e5
        I = 1000
        self.p.create_beams((0, 0), (L, 0), E=E, I=I)
        self.p.node_at((0,0)).fix()
        self.p.node_at((1000,0)).loads = np.array([0, -P, 0])

        self.p.solve()

        self.assertAlmostEqual(-P*L**3 / (3*E*I), self.p.node_at((1000,0)).displacements[1,0], places=5)

    def test_cantilever_bar_with_preload(self):
        n1 = Node((0, 0))
        n2 = Node((1000, 0))
        self.p.nodes.append(n1)
        self.p.nodes.append(n2)

        P = 1000
        L = 1000
        E = 210e5
        I = 1000
        A = 100
        beam = self.p.create_beam(n1, n2, E=E, I=I, A=A)
        n1.fix()
        beam.preload = np.array([P, 0, 0, -P, 0, 0]).reshape((6,1))

        self.p.solve()
        disp = self.p.displacements

        self.assertAlmostEqual(-P*L/(E*A), n2.get_displacements(disp)[0, 0], places=5)
        self.assertTrue(np.allclose(np.zeros(6), beam.get_forces_local_lin(disp).flatten()), f'Nonzero: {beam.get_forces_local_lin(disp).flatten()} ')

    def test_von_mises_truss(self):
        p = self.problem = self.p
        n1 = p.get_or_create_node((0,0))
        n2 = p.get_or_create_node((1000,200))
        p.create_beam(n1, n2, A=10)
        n1.pin()
        n2.roller90()
        n2.loads = np.array([0, -10000, 0])
        solver = solvers.NonLinearSolver(p)
        res = solver.solveall()
        self.assertAlmostEqual(-482.59459618768216, res.displacements[-1, 4], delta=2)

    def test_snapback_von_mises_truss(self):
        p = self.problem = self.p
        n1 = p.get_or_create_node((0, 0))
        n2 = p.get_or_create_node((1000, 200))
        n3 = p.get_or_create_node((1000, 600))
        p.create_beam(n1, n2, A=10)
        p.create_rod(n2, n3, A=0.05)
        n1.pin()
        n2.roller90()
        n3.glider()
        n3.loads = np.array([0, -4000, 0])
        solver = solvers.NonLinearSolver(p)
        res = solver.solveall()
        self.assertAlmostEqual(-442.62588512549337, res.displacements[-1, 4], delta=4)

    def _set_up_path_dependent_mises_truss(self):
        p = self.problem = self.p
        n1 = p.get_or_create_node((0, 0))
        n2 = p.get_or_create_node((-200, 1000))
        n3 = p.get_or_create_node((0, 2000))
        p.create_rod(n1, n2, A=10)
        p.create_rod(n2, n3, A=10)
        n1.fix()
        n2.constrained_dofs = [2] # Rotation lock
        n3.constrained_dofs = [0,2]

        p.elements.append(BoundarySpring(n3, np.array([0, 10000, 0])))

        # Case 1: Load on middle node (n2) first, then on n3. Should snap through
        for n in p.nodes:
            n.loads = np.zeros((3,2))

    def assertSmaller(self, a, b):
        self.assertTrue(a < b, f'Assertion failed: {a} not smaller than {b}')
        

    def test_path_dependent_mises_truss_displ_right(self):
        self._set_up_path_dependent_mises_truss()
        n1,n2,n3 = self.p.nodes

        n2.loads[:3, 0] = np.array([11000, 0, 0])
        n2.loads[:3, 1] = np.array([11000, 0, 0])

        n3.loads[:3, 0] = np.array([0, 0, 0])
        n3.loads[:3, 1] = np.array([0, -1000000, 0])

        solver = solvers.NonLinearSolver(self.p)
        res = list(solver.solve())[-1]
        disp = res.displacements[-2:-1, :].T

        self.assertGreater(n2.get_displacements(disp)[0, 0], 450)
        self.assertGreater(n2.get_displacements(disp)[1, 0], n2.get_displacements(disp)[0, 0])

    def test_path_dependent_mises_truss_displ_left(self):
        self._set_up_path_dependent_mises_truss()
        n1,n2,n3 = self.p.nodes

        n2.loads[:3, 0] = np.array([0, 0, 0])
        n2.loads[:3, 1] = np.array([0, 11000, 0])

        n3.loads[:3, 0] = np.array([0, -1000000, 0])
        n3.loads[:3, 1] = np.array([0, -1000000, 0])

        solver = solvers.NonLinearSolver(self.p)
        res = list(solver.solve())
        disp = np.vstack((res[0].displacements[-1,:], res[1].displacements[-1,:])).T

        self.assertSmaller(n2.get_displacements(disp)[0, 0], 0)
        self.assertSmaller(n2.get_displacements(disp)[0, 1], 0)
        self.assertGreater(n2.get_displacements(disp)[0, 1], n2.get_displacements(disp)[0, 0])






