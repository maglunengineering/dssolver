import numpy as np

class DSSModelObject:
    def __hash__(self):
        return id(self)

class Node(DSSModelObject):
    def __init__(self, xy):
        self._r = np.array(xy)

        self._elements = list()
        self.loads = np.zeros(3) # self.loads (global Fx, Fy, M) assigned on loading
        self.displacements = np.zeros(3)

        self.number = None  # (node id) assigned on creation
        self.dofs = []  # self.dofs (dof1, dof2, dof3) assigned on creation
        self.constrained_dofs = []  # 'fixed', 'pinned', 'roller', 'locked', 'glider'

    def add_element(self, beam):
        self._elements.append(beam)

    @property
    def r(self):
        return self._r

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
        new_node.dofs = self.dofs
        new_node.loads = self.loads
        new_node.constrained_dofs = self.constrained_dofs
        return new_node

    def __str__(self):
        return f'Node(({self._r[0]},{self._r[1]}))'

    def __hash__(self):
        return id(self)

class FiniteElement(DSSModelObject):
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
            self._dofs = np.hstack([n.dofs for n in self.nodes])
            self._ix = np.ix_(self._dofs, self._dofs)
        return self._dofs

    def ix(self):
        if self._dofs is None:
            self._dofs = np.hstack([n.dofs for n in self.nodes])
            self._ix = np.ix_(self._dofs, self._dofs)
        return self._ix

    def get_displacements(self):
        return np.hstack([n.displacements for n in self.nodes])

    def nonlin_update(self):
        pass

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
        self._update_stiffness()
        self._update_forces_local()

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
        self._update_forces_local() # TODO: Remove but slightly breaks the mises truss tests
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
        """
        Bending energy is
        integral between (x=0, L) of M(x)/(2EI) dx
        Am assuming
        M(x) = M1(1-x/L) + M2(x/L)
        """
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

    def draw_on_canvas(self, canvas, **kwargs):
        if not self.settings['Displaced']:
            r1 = self.node1.r
            r2 = self.node2.r
        else:
            r1 = self.node1.r + self.node1.displacements[0:2]
            r2 = self.node2.r + self.node2.displacements[0:2]

        length = np.linalg.norm(self.r2 - self.r1)
        dirvec = (r2 - r1) / length
        normal = np.array([-dirvec[1], dirvec[0]]) * (length / self._deformed_length())**2
        num_steps = 21
        steplen = length * 1 / num_steps

        pt1 = r1
        pt2 = pt1 + normal * steplen * 0.5 + dirvec * steplen * 0.5
        pts = [pt1, pt2]
        sign = -1
        for i in range(num_steps - 1):
            new_pt = pts[-1] + sign * normal * steplen + dirvec * steplen
            pts.append(new_pt)
            sign *= -1
        pts.append(r2)

        if not self.settings['Displaced']:
            for pt1, pt2 in zip(pts, pts[1:]):
                canvas.draw_line(pt1, pt2)
        else:
            for pt1, pt2 in zip(pts, pts[1:]):
                canvas.draw_line(pt1, pt2, fill='red', dash=(1,), **kwargs)

    def clone(self, newnode1, newnode2):
        return Rod(newnode1, newnode2, self.E, self.A)

class Quad4(FiniteElement):
    def __init__(self, node1, node2, node3, node4, E, v, t):
        super().__init__([node1, node2, node3, node4])
        self.nodes = [node1, node2, node3, node4]
        self.E = E
        self.v = v
        self.t = t

    def stiffness_matrix_global(self):
        return np.zeros((12,12))

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