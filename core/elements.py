import inspect
from enum import IntFlag, auto
import numpy as np

class DSSModelObject:
    def __hash__(self):
        return id(self)

class Node(DSSModelObject):
    def __init__(self, xy):
        self._r = np.array(xy)

        self._elements = list()
        self._loads = np.zeros(3, dtype=float) # self.loads (global Fx, Fy, M) assigned on loading
        self.displacements = np.zeros(3, dtype=float)

        self._dofs = None
        self.constrained_dofs = []

    @property
    def loads(self):
        return self._loads

    @loads.setter
    def loads(self, value):
        self._loads = value

    def get_displacements(self, displacements):
        if self._dofs is None or displacements is None:
            return np.zeros(3)
        return displacements[..., self._dofs]

    def add_element(self, beam):
        if beam not in self._elements:
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
        if self._dofs is None:
            return np.arange(self.ndofs())
        return self._dofs

    @dofs.setter
    def dofs(self, value):
        new_ndofs = len(value)
        old_ndofs = len(self._dofs) if self._dofs is not None else new_ndofs


        new_shape = np.array(self.loads.shape)
        new_shape[-1] = new_ndofs

        if new_ndofs > old_ndofs:
            new_loads = np.zeros(new_shape)
            new_disp = np.zeros(new_shape)
            new_loads[..., :old_ndofs] = self.loads
            new_disp[..., :old_ndofs] = self.displacements
            self.loads = new_loads
            self.displacements = new_disp
        else:
            self.loads = self.loads[..., :old_ndofs]
            self.displacements = self.displacements[..., :old_ndofs]

        while new_ndofs < len(self.constrained_dofs):
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


class ElementBehavior(IntFlag):
    DISABLED = 1
    ITERABLE = 2
    NONLIN_GEOM = 4
    #NONLIN_MATR = 8

    def without(self, behavior):
        return ElementBehavior(self - behavior) if behavior in self else self

    def including(self, behavior):
        return ElementBehavior(self + behavior) if not behavior in self else self


class FiniteElement(DSSModelObject):
    ndofs = 3
    behavior = ElementBehavior(0)

    def __init__(self, nodes):
        self.nodes = nodes

        for node in self.nodes:
            node.add_element(self)

        self.stiffness_matrix_local = np.zeros((6,6))
        self.preload: np.ndarray = np.zeros(1)
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

    def stiffness_matrix_global(self, displacements) -> np.ndarray:
        raise NotImplementedError(self.__class__.__name__)

    def get_displacements(self, displacements):
        return displacements[..., self.dofs]

    def nonlin_update(self, behavior, i_lc, displacements):
        pass

    def do_iterate(self, global_K:np.ndarray, global_u:np.ndarray, global_f:np.ndarray) -> bool:
        """
        :return: False if iteration should be done again, True if iteration is finished
        """
        if not ElementBehavior.ITERABLE in self.behavior:
            raise NotImplementedError(f'Called iterate on non-iterable {self.__class__.__name__}')
        return True

    def reinit(self):
        init_args = inspect.getfullargspec(self.__init__).args
        kwargs = {}
        for arg in init_args:
            if arg == 'self':
                continue
            elif hasattr(self, arg):
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

        # Don't initialize with displaced node as we'll get "nonlinear" behavior
        assert np.allclose(node1.displacements, 0)
        assert np.allclose(node2.displacements, 0)

        self._undeformed_length = np.linalg.norm(self.r2 - self.r1)

    @property
    def stiffness_matrix_local(self):
        return self._stif_local

    @stiffness_matrix_local.setter
    def stiffness_matrix_local(self, value):
        self._stif_local = value

    @property
    def r1(self):
        return self.node1.r

    @property
    def r2(self):
        return self.node2.r

    def nonlin_update(self, behavior, i_lc, displacements):
        pass

    def get_external_forces(self, displacements):
        T_T = np.swapaxes(self._get_transform(displacements), -1, -2) # Transpose that works for both (m,n,n) and (n,n)
        f = self._get_forces_local(displacements)[..., np.newaxis]
        return (T_T @ f)[..., 0] # ((m), n, n) @ ((m), n, n) -> ((m), n)

    def _get_transform(self, displacements):
        if displacements is not None:
            deformed_length = self._get_deformed_length(displacements)
            e1 = ((self.node2.r + self.node2.get_displacements(displacements)[..., :2]) -
                  (self.node1.r + self.node1.get_displacements(displacements)[..., :2])) / deformed_length
        else:
            e1 = (self.node2.r - self.node1.r) / self._undeformed_length
        e2 = e1[..., ::-1] * np.array([-1, 1])
        if e1.ndim > 1:
            T = np.zeros((*e1.shape[:-1], 6, 6))
        else:
            T = np.zeros((6,6))

        T[...,0,0] = e1[..., 0]
        T[...,0,1] = e1[..., 1]
        T[...,1,0] = e2[..., 0]
        T[...,1,1] = e2[..., 1]
        T[...,3,3] = e1[..., 0]
        T[...,3,4] = e1[..., 1]
        T[...,4,3] = e2[..., 0]
        T[...,4,4] = e2[..., 1]
        T[...,2,2] = 1.0
        T[...,5,5] = 1.0

        return T

    def stiffness_matrix_global(self, displacements, geostiff=True) -> np.ndarray:
        T = self._get_transform(displacements)
        K = self.stiffness_matrix_local
        if geostiff and displacements is not None:
            K = (K + self._get_stiffness_geometric(displacements))
        return np.swapaxes(T, -1, -2) @ K @ T

    def _get_stiffness_geometric(self, displacements):
        deformed_length = self._get_deformed_length(displacements)
        #fx1,fy1,m1,fx2,fy2,m2 = self._get_forces_local(displacements)
        forces_permuted = self._get_forces_local(displacements)[..., [1, 0, 2, 4, 3, 5]] * np.array([-1, 1, 0, -1, 1, 0])
        o = np.zeros_like(deformed_length) # Zero

        # 1d:
        # Outer product of n 6-vectors:
        # (n,6), (n,6) -> (n, 1, 6) * (n, 6, 1) -> (n, 6, 6)

        G = np.array([o, -1/deformed_length, o, o, 1/deformed_length, o]).T
        return forces_permuted[..., np.newaxis, :] * G[0, ..., :, np.newaxis]

    def mass_matrix_global(self) -> np.ndarray:
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

    def _get_deformed_length(self, displacements):
        r1 = self.node1.r + self.node1.get_displacements(displacements)[..., :2]
        r2 = self.node2.r + self.node2.get_displacements(displacements)[..., :2]
        return np.linalg.norm(r2 - r1, axis=-1, keepdims=True)

    def get_forces_local_lin(self, displacements):
        disp_local = self.get_displacements(displacements) @ np.swapaxes(self._get_transform(displacements), -1, -2)
        return disp_local @ self.stiffness_matrix_local - self.preload

    def _get_forces_local(self, displacements):
        r1 = self.r1
        r2 = self.r2
        u1 = self.node1.get_displacements(displacements)
        u2 = self.node2.get_displacements(displacements)
        uu = np.stack((u1,u2), axis=0)
        # ^ np.stack creates a new axis on the very left. Saves some calls (esp to R) below which is slow
        # In those cases, [0] is at node1 and [1] is at node 2.

        tan_ed = (r2 - r1 + (u2 - u1)[..., 0:2])
        deformed_length = np.linalg.norm(tan_ed, axis=-1, keepdims=True)
        dl = deformed_length - self._undeformed_length
        tan_e0 = (r2 - r1)/self._undeformed_length
        tan_ed = tan_ed/deformed_length
        tan_rd = R(uu[..., 2:3]) @ tan_e0 # R is slow as hell. Try to reformulate?
        th = np.arcsin(tan_ed[..., 0:1]*tan_rd[..., 1:2] - tan_ed[..., 1:2]*tan_rd[..., 0:1]) # Indexers like 0:1 are to get a single value but keep the dimension
        displacements_local = np.zeros((*displacements.shape[:-1], 6))
        displacements_local[..., [0]] = -dl/2
        displacements_local[..., [2]] = th[0]
        displacements_local[..., [3]] = dl/2
        displacements_local[..., [5]] = th[1]
        return displacements_local @ self.stiffness_matrix_local

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

    def get_strain_energy(self, displacements):
        """ Bending energy: integral (x=0, L, M(x)/(2EI), dx)
            Am assuming M(x) = M1(1-x/L) + M2(x/L) """

        deformed_length = self._get_deformed_length(displacements)
        c = 1/(6 * self.E * self.I)
        forces = self.get_forces_local_lin(displacements)
        M1,M2 = forces[..., 2], forces[..., 5]
        bending_strain_energy = c*(M1**2 + M1*M2 + M2**2)*self._undeformed_length

        axial_strain = 1 - deformed_length/self._undeformed_length
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

    def get_strain_energy(self):
        strain = 1 - self._deformed_length() / self._undeformed_length
        stress = self.E * strain
        return np.abs(0.5 * stress * strain * self.A * self._undeformed_length)

    def clone(self, newnode1, newnode2):
        return Rod(newnode1, newnode2, self.E, self.A)

class BoundarySpring(FiniteElement):
    def __init__(self, node, stiffness:np.ndarray):
        super().__init__([node])
        self.stiffness_matrix_local = np.diag(np.asarray(stiffness))

    def stiffness_matrix_global(self, displacements) -> np.ndarray:
        return self.stiffness_matrix_local

    def get_external_forces(self, displacements):
        return self.get_displacements(displacements) @ self.stiffness_matrix_local


class PenaltyBeam(FiniteElement):
    behavior = FiniteElement.behavior.including(ElementBehavior.ITERABLE)
    def __init__(self, node1, node2):
        super().__init__((node1, node2))
        self.stiffness_matrix_local = np.zeros((6,6))
        self._is_calibrated = False


    def stiffness_matrix_global(self, displacements) -> np.ndarray:
        return self.stiffness_matrix_local

    def do_iterate(self, global_K:np.ndarray, global_u:np.ndarray, global_f:np.ndarray) -> bool:
        """
        :return: False if iteration should be done again, True if iteration is finished
        """
        # IFEM 9.2.3: 10**(k+p/2) where k is order of max stiffness and p is machine prec
        if self._is_calibrated:
            return True
        stiff = 10 ** (np.log10(global_K.max()) + 7)
        self.stiffness_matrix_local = np.array([
            [stiff, 0, 0, -stiff, 0, 0],
            [0, stiff, 0, 0, -stiff, 0],
            [0, 0, stiff, 0, 0, -stiff],
            [-stiff, 0, 0, stiff, 0, 0],
            [0, -stiff, 0, 0, stiff, 0],
            [0, 0, -stiff, 0, 0, stiff]
        ])
        self._is_calibrated = True
        return False # Change needs to propagate



class Quad4(FiniteElement):
    ndofs = 2
    def __init__(self, node1, node2, node3, node4, E, v, t):
        super().__init__([node1, node2, node3, node4])
        self.nodes = [node1, node2, node3, node4]
        self.E = E
        self.v = v
        self.t = t
        self._r = np.array([node.r for node in self.nodes])

    def stiffness_matrix_global(self, displacements) -> np.ndarray:
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
    T = np.zeros((*np.shape(angle)[:-1], 2, 2))
    s, c = np.sin(angle), np.cos(angle)
    T[..., [0], [0]] = c
    T[..., [0], [1]] = -s
    T[..., [1], [0]] = s
    T[..., [1], [1]] = c
    return T