import inspect
import numpy as np

class DSSModelObject:
    def __hash__(self):
        return id(self)

class Node(DSSModelObject):
    def __init__(self, xy):
        self._r = np.array(xy)

        self._elements = list()
        self.loads = np.zeros(2) # self.loads (global Fx, Fy, M) assigned on loading
        self.displacements = np.zeros(2)

        self._dofs = []  # self.dofs (dof1, dof2, dof3) assigned on creation
        self.constrained_dofs = []

    def add_element(self, beam):
        self._elements.append(beam)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value
        for e in self._elements:
            e.reinit()

    @property
    def dofs(self):
        return self._dofs

    @dofs.setter
    def dofs(self, value):
        if self._dofs is not None:
            self.loads = np.hstack((self.loads, np.zeros(len(value))))[:len(value)]
            self.displacements = np.hstack((self.displacements, np.zeros(len(value))))[:len(value)]
            while len(value) < len(self.constrained_dofs):
                self.constrained_dofs.pop()
        self._dofs = value


    def ndofs(self):
        return max(e.ndofs for e in self._elements)

    def connected_nodes(self):
        other_nodes = []
        for element in self._elements:
            for node in element.nodes:
                if not node == self:
                    other_nodes.append(node)
        return other_nodes

    def fix(self):
        self.constrained_dofs = [0,1,2]

    def pin(self):
        self.constrained_dofs = [0,1]

    def roller(self):
        self.constrained_dofs = [1]

    def roller90(self):
        self.constrained_dofs = [0]

    def lock(self):
        self.constrained_dofs = [2]

    def glider(self):
        self.constrained_dofs = [0,2]

    def copy(self):
        new_node = Node(self.r)
        new_node._dofs = self._dofs
        new_node.loads = self.loads
        new_node.constrained_dofs = self.constrained_dofs
        return new_node

    def __str__(self):
        return f'Node(({self._r[0]},{self._r[1]}))'

    def __hash__(self):
        return id(self)

class FiniteElement(DSSModelObject):
    ndofs = 3
    def __init__(self, nodes):
        self.nodes = nodes

        for node in self.nodes:
            node.add_element(self)

        self.stiffness_matrix_local = np.zeros((6,6))
        self._dofs = None # Cached
        self._ix = None # Cached

    @property
    def dofs(self):
        if self._dofs is None:
            self._dofs = np.hstack([n.dofs[:self.ndofs] for n in self.nodes])
            self._ix = np.ix_(self._dofs, self._dofs)
        return self._dofs

    def ix(self):
        if self._dofs is None:
            self._dofs = np.hstack([n.dofs[:self.ndofs] for n in self.nodes])
            self._ix = np.ix_(self._dofs, self._dofs)
        return self._ix

    def get_displacements(self):
        return np.hstack([n.displacements for n in self.nodes])

    def nonlin_update(self):
        pass

    def reinit(self):
        init_args = inspect.getfullargspec(self.__init__).args
        kwargs = {}
        for arg in init_args:
            if hasattr(self, arg):
                kwargs[arg] = getattr(self, arg)
            elif hasattr(self, '_' + arg):
                kwargs[arg] = getattr(self, '_' + arg)
            else:
                raise AttributeError(f'Cannot reinit {self.__class__.__name__}: Missing arg {arg}')
        self.__init__(**kwargs)

class FiniteElement2Node(FiniteElement):
    def __init__(self, node1:Node, node2:Node, A:float):
        super().__init__([node1, node2])
        self.node1 = node1
        self.node2 = node2
        self.A = A

        self._undeformed_length = np.linalg.norm(self.r2 - self.r1)
        self._forces_local = np.zeros(6)
        self._deformed_length = self._update_deformed_length()
        self._transform = self._update_transform()
        self._stiffness = self._update_stiffness()

    @property
    def r1(self):
        return self.node1.r

    @property
    def r2(self):
        return self.node2.r

    def nonlin_update(self):
        self._update_deformed_length()
        self._update_transform()
        self._update_forces_local()
        self._update_stiffness()

    def get_forces(self):
        return self._transform.T @ self._get_forces_local()

    def _update_transform(self):
        e1 = ((self.node2.r + self.node2.displacements[:2]) -
              (self.node1.r + self.node1.displacements[0:2])) / self._deformed_length
        e2 = [-e1[1], e1[0]]
        T = np.array([[e1[0], e1[1], 0, 0, 0, 0],
                      [e2[0], e2[1], 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, e1[0], e1[1], 0],
                      [0, 0, 0, e2[0], e2[1], 0],
                      [0, 0, 0, 0, 0, 1]])
        self._transform = T
        return self._transform

    def _update_stiffness(self):
        self._stiffness = self._transform.T @ (self.stiffness_matrix_local + self._get_stiffness_geometric()) @ self._transform
        return self._stiffness

    def stiffness_matrix_global(self):
        return self._stiffness

    def _get_stiffness_geometric(self):
        fx1,fy1,m1,fx2,fy2,m2 = self._get_forces_local()
        forces_permuted = np.array([-fy1, fx1, 0, -fy2, fx2, 0])
        G = np.array([0, -1/self._deformed_length, 0, 0, 1/self._deformed_length, 0])
        return np.outer(forces_permuted, G)

    def mass_matrix_global(self):
        T = self._transform
        density = 7.86e-9 # Density of steel in tonnes / mm^3
        half_mass = self.A * self._undeformed_length * density / 2
        rot_mass = 1/50 * half_mass * self._undeformed_length ** 2 # Felippa: IFEM Ch.31
        rot_mass = half_mass # This is likely better (in fact, should be negative but matrix must be positive)
        local = half_mass * np.array([[1, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 0, rot_mass, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0, rot_mass]])
        return T.T @ local @ T

    def _update_deformed_length(self):
        self._deformed_length = np.linalg.norm((self.node2.r + self.node2.displacements[:2] -
                                                self.node1.r - self.node1.displacements[:2]))
        return self._deformed_length

    def _get_forces_local(self):
        return self._forces_local

    def _update_forces_local(self):
        r1 = self.r1
        r2 = self.r2
        dl = self._deformed_length - self._undeformed_length

        tan_e = (r2 - r1)/self._undeformed_length
        tan_ed = (r2 + self.node2.displacements[0:2] -
                  r1 - self.node1.displacements[0:2])/self._deformed_length
        tan_1 = R(self.node1.displacements[2]) @ tan_e
        tan_2 = R(self.node2.displacements[2]) @ tan_e

        th1 = np.arcsin(tan_ed[0]*tan_1[1] - tan_ed[1]*tan_1[0])
        th2 = np.arcsin(tan_ed[0]*tan_2[1] - tan_ed[1]*tan_2[0])

        displacements_local = np.array([-dl/2, 0, th1, dl/2, 0, th2])
        self._forces_local = self.stiffness_matrix_local @ displacements_local
        return self._forces_local

class Beam(FiniteElement2Node):
    def __init__(self, node1:Node, node2:Node, E=2e5, A=1e5, I=1e5, z=None):
        super().__init__(node1, node2, A)

        self.E = E
        self.A = A
        self.I = I
        self.z = z if z else np.sqrt(I/A)/3

        length = self._undeformed_length
        kn = A*E/length * (E*I/length**3)**(-1)
        self.stiffness_matrix_local = E*I/length**3 * np.array(
             [[kn, 0, 0, -kn, 0, 0],
              [0, 12, 6*length, 0, -12, 6*length],
              [0, 6*length, 4*length**2, 0, -6*length, 2*length**2],
              [-kn, 0, 0, kn, 0, 0],
              [0, -12, -6*length, 0, 12, -6*length],
              [0, 6*length, 2*length**2, 0, -6*length, 4*length**2]])

        self._stiffness = self._update_stiffness()

    def get_strain_energy(self):
        """ Bending energy: integral (x=0, L, M(x)/(2EI), dx)
            Am assuming M(x) = M1(1-x/L) + M2(x/L) """

        c = 1/(6 * self.E * self.I)
        forces = self._get_forces_local()
        M1,M2 = forces[2], forces[5]
        bending_strain_energy = c*(M1**2 + M1*M2 + M2**2)*self._undeformed_length

        axial_strain = 1 - self._deformed_length/self._undeformed_length
        axial_stress = self.E * axial_strain
        axial_strain_energy = 0.5 * axial_stress * axial_strain * self.A * self._undeformed_length

        return axial_strain_energy + bending_strain_energy

    def clone(self, newnode1, newnode2):
        return Beam(newnode1, newnode2, self.E, self.A, self.I, self.z)

class Rod(FiniteElement2Node):
    def __init__(self, r1, r2, E=2e5, A=1e5, *args, **kwargs):
        super().__init__(r1, r2, A)

        self.E = E
        self.A = A
        length = np.linalg.norm(self.node2.r - self.node1.r)
        self.kn = A*E/length
        self.stiffness_matrix_local = np.array([  [self.kn, 0, 0, -self.kn, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [-self.kn, 0, 0, self.kn, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0]])
        self._stiffness = self._update_stiffness()

    def get_strain_energy(self):
        strain = 1 - self._deformed_length() / self._undeformed_length
        stress = self.E * strain
        return np.abs(0.5 * stress * strain * self.A * self._undeformed_length)

    def clone(self, newnode1, newnode2):
        return Rod(newnode1, newnode2, self.E, self.A)

class Quad4(FiniteElement):
    ndofs = 2
    def __init__(self, node1, node2, node3, node4, E, v, t):
        super().__init__([node1, node2, node3, node4])
        self.nodes = [node1, node2, node3, node4]
        self.E = E
        self.v = v
        self.t = t
        self._r = np.array([node.r for node in self.nodes])

    def stiffness_matrix_global(self):
        a = 1 / np.sqrt(3)
        integration_points = [[-a, a], [a, a], [-a, -a], [a, -a]]

        k = np.zeros((8, 8))
        material_stiffness = self.E / (1 - self.v**2) * np.array([[1, self.v,   0],
                                                                  [-self.v, 1,  0],
                                                                  [0, 0, 1+self.v]])
        for r in integration_points:
            B = self.strain_displ(r)
            k = k + B.T @ material_stiffness @ B * np.linalg.det(self.jacobian(r)) * self.t

        return k

    @staticmethod
    def shape_functions(pt):
        xi, eta = pt
        return 0.25 * np.array([(1 - xi) * (1 - eta),
                                (1 + xi) * (1 - eta),
                                (1 + xi) * (1 + eta),
                                (1 - xi) * (1 + eta)])

    @staticmethod
    def shape_functions_deriv(pt):
        xi, eta = pt
        return 0.25 * np.array([[-(1 - eta), 1 - eta,  1 + eta, -(1 + eta)],
                                [-(1 - xi), -(1 + xi), 1 + xi,  1 - xi]])

    def strain_displ(self, pt):
        dNdx, dNdy = np.linalg.solve(self.jacobian(pt), self.shape_functions_deriv(pt))
        return np.array([[dNdx[0], 0, dNdx[1], 0, dNdx[2], 0, dNdx[3], 0],
                         [0, dNdy[0], 0, dNdy[1], 0, dNdy[2], 0, dNdy[3]],
                         [dNdy[0], dNdx[0], dNdy[1], dNdx[1], dNdy[2], dNdx[2], dNdy[3], dNdx[3]]])

    def jacobian(self, pt):
        return Quad4.shape_functions_deriv(pt) @ self._r


def beta(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, s, 0, 0, 0, 0],
                     [-s, c, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, c, s, 0],
                     [0, 0, 0, -s, c, 0],
                     [0, 0, 0, 0, 0, 1]])

def R(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, -s],
                     [s, c]])