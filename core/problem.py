from typing import List, Iterable, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from core.elements import FiniteElement, Node, Rod, Beam, ElementBehavior
import core.results as results

np.set_printoptions(suppress=True)


class Problem:
    def __init__(self):
        self.nodes:List[Node] = []
        self.elements:List[FiniteElement] = []

        self.constrained_dofs = []
        self.forces = None  # Forces (at all nodes, incl removed dofs)
        self.displacements = None  # Assigned at solution ( self.solve() )

    def create_beam(self, node1:Node, node2:Node, E=2e5, A=1e5, I=1e5, z=None):
        if isinstance(node1, np.ndarray):
            node1 = self.get_or_create_node(node1)
        if isinstance(node2, np.ndarray):
            node2 = self.get_or_create_node(node2)
        element = Beam(node1, node2, E, A, I, z)
        self.elements.append(element)

        return element

    def create_rod(self, node1, node2, E=2e5, A=1e5, *args, **kwargs):
        if isinstance(node1, np.ndarray):
            node1 = self.get_or_create_node(node1)
        if isinstance(node2, np.ndarray):
            node2 = self.get_or_create_node(node2)
        rod = Rod(node1, node2, E, A)

        self.elements.append(rod)

    def create_beams(self, r1, r2, E=2e5, A=1e5, I=1e5, z=None, n=4):
        rr = np.linspace(r1, r2, int(n+1))

        for ri, rj in zip(rr, rr[1:]):
            node1 = self.get_or_create_node(ri)
            node2 = self.get_or_create_node(rj)
            self.create_beam(node1, node2, E, A, I, z)

    def get_or_create_node(self, r) -> Node:
        for node in self.nodes:
            if np.allclose(r, node.r):
                return node
        else:
            new_node = Node(r)
            self.nodes.append(new_node)
            return new_node

    def node_at(self, r) -> Node:
        r = np.asarray(r)
        for node in self.nodes:
            if np.allclose(node.r, r):
                return node

    def reassign_dofs(self):
        i = 0
        for node in self.nodes:
            ndofs = node.ndofs()
            node.dofs = np.arange(i, i+ndofs)
            i += ndofs

    def upd_obj_displacements(self):
        for node in self.nodes:
            node.displacements = self.displacements[node.dofs]

    def remove_dofs(self):  # Interpret boundary conditions
        self.constrained_dofs = []
        for node in self.nodes:
            self.constrained_dofs.extend(node.dofs[node.constrained_dofs])

    def nonlin_update(self):
        for e in self.elements:
            e.nonlin_update(ElementBehavior.NONLIN_GEOM)

    def model_size(self):
        xy = self.nodal_coordinates
        if not np.any(xy):
            return 1
        else:
            model_size = np.sqrt( (np.max(xy[:,0]) - np.min(xy[:,0]))**2 + (np.max(xy[:,1]) - np.min(xy[:,1]))**2)
            return model_size

    def free_dofs(self) -> np.ndarray:
        return np.delete(np.arange(sum(n.ndofs() for n in self.nodes)), self.constrained_dofs)

    def M(self, reduced=False):
        return self.assemble(lambda e: e.mass_matrix_global(), reduced)

    def K(self, reduced=False):
        return self.assemble(lambda e: e.stiffness_matrix_global(), reduced)

    def assemble_loads(self, min_max_dim=0):
        max_dim = max(min_max_dim, max(node.loads.shape[0] if len(node.loads.shape) > 1 else 1 for node in self.nodes))
        return np.hstack([np.broadcast_to(node.loads, (max_dim, node.ndofs(), 1)) for node in self.nodes])

    def assemble_displacements(self, min_max_dim=0):
        max_dim = max(min_max_dim, max(node.displacements.shape[0] if len(node.displacements.shape) > 1 else 1 for node in self.nodes))
        return np.hstack([np.broadcast_to(node.displacements, (max_dim, node.ndofs(), 1)) for node in self.nodes])

    def assemble_preload(self, min_max_dim=0):
        max_dim = max(min_max_dim, max(node.loads.shape[0] if len(node.loads.shape) > 1 else 1 for node in self.nodes))
        preload = np.zeros((max_dim, sum(n.ndofs() for n in self.nodes), 1))
        for el in self.elements:
            preload[:, el.dofs, :] += el.preload
        return preload

    def assemble(self, elem_func, reduced=False):
        if not self.constrained_dofs:
            self.remove_dofs()

        num_dofs = sum(n.ndofs() for n in self.nodes)

        matrix = np.zeros((num_dofs, num_dofs))
        for e in self.elements:
            contrib = elem_func(e)
            matrix[e.ix()] += contrib

        if not reduced:
            return matrix
        else:
            free_dofs = self.free_dofs()
            return matrix[np.ix_(free_dofs, free_dofs)]

    def solve(self) -> Iterable[Optional[results.ResultsStaticLinear]]:
        self.reassign_dofs()
        self.remove_dofs()
        free_dofs = self.free_dofs()
        constrained_dofs = self.constrained_dofs

        iterables = [e for e in self.elements if ElementBehavior.ITERABLE in e.behavior]
        max_iter = 8
        itercnt = 0

        while True:
            K = self.K()
            K11 = K[np.ix_(free_dofs, free_dofs)]
            K12 = K[np.ix_(free_dofs, constrained_dofs)]
            K21 = K[np.ix_(constrained_dofs, free_dofs)]
            K22 = K[np.ix_(constrained_dofs, constrained_dofs)]

            # Assemble displacements
            forces = self.assemble_loads()
            displacements = self.assemble_displacements(forces.shape[0])
            preload = self.assemble_preload(forces.shape[0])

            displacements[:, free_dofs, :] = np.linalg.solve(K11, (forces+preload)[:, free_dofs, :] - K12 @ displacements[:, constrained_dofs, :])
            forces[:, constrained_dofs, :] = K21 @ displacements[:, free_dofs, :] + K22 @ displacements[:, constrained_dofs, :]

            for node in self.nodes:
                node.loads = (forces - preload)[:, node.dofs, :]
                node.displacements = displacements[:, node.dofs, :]
            self.displacements = displacements

            if not iterables:
                break
            elif itercnt >= max_iter:
                break
            else:
                if all(iterable.do_iterate(K, displacements, forces) for iterable in iterables):
                    break


        return [results.ResultsStaticLinear(self, displacements)]

    def plot(self):
        nodal_coordinates = np.array([0,0])

        plt.figure()
        for node in self.nodes:
            nodal_coordinates = np.vstack((nodal_coordinates, node.r))

        for beam in self.elements:
            plt.plot(*np.array([beam.r1, beam.r2]).T, color='b')

        k = 10
        plt.xlim((np.min(nodal_coordinates[:, 0]) - self.model_size() / k,
                  np.max(nodal_coordinates[:, 0]) + self.model_size() / k))
        plt.ylim((np.min(nodal_coordinates[:, 1]) - self.model_size() / k,
                  np.max(nodal_coordinates[:, 1]) + self.model_size() / k))

        try:
            plt.arrow(*node.r,
                      *node.loads[0:2]/np.linalg.norm(node.loads[0:2])*self.model_size()/10,
                      head_width=20 )
        except:
            pass

    @property
    def nodal_coordinates(self):
        nodal_coordinates = np.array([node.r for node in self.nodes])
        return nodal_coordinates

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
