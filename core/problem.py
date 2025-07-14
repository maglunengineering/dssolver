import typing
from typing import List, Iterable, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from core.elements import FiniteElement, Node, Rod, Beam, ElementBehavior
import core.results as results

np.set_printoptions(suppress=True)
WithDofs = typing.Union[FiniteElement, Node]
T = typing.TypeVar('T', bound=WithDofs)


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

    def remove_dofs(self):  # Interpret boundary conditions
        self.constrained_dofs = []
        for node in self.nodes:
            self.constrained_dofs.extend(node.dofs[node.constrained_dofs])

    def nonlin_update(self, i_lc, displacements):
        for e in self.elements:
            e.nonlin_update(ElementBehavior.NONLIN_GEOM, i_lc, displacements)

    def model_size(self) -> float:
        xy = self.nodal_coordinates
        if not np.any(xy):
            return 1.0
        else:
            model_size = np.sqrt( (np.max(xy[:,0]) - np.min(xy[:,0]))**2 + (np.max(xy[:,1]) - np.min(xy[:,1]))**2)
            return model_size

    def free_dofs(self) -> np.ndarray:
        return np.delete(np.arange(sum(n.ndofs() for n in self.nodes)), self.constrained_dofs)

    def M(self):
        return self.assemble_matrix(lambda e: e.mass_matrix_global())

    def K(self, displacements):
        return self.assemble_matrix(lambda e: e.stiffness_matrix_global(displacements), displacements.shape[-1])

    def assemble_vector(self, collection:Iterable[T], func:Callable[[T], np.ndarray], ndofs:int):
        shape = np.asarray(func(next(iter(collection))).shape)
        shape[-1] = ndofs
        assembly = np.zeros(shape)
        for item in collection:
            assembly[..., item.dofs] += func(item)
        return assembly

    def assemble_matrix(self, elem_func, ndofs):
        if not self.constrained_dofs:
            self.remove_dofs()

        shape = np.asarray(elem_func(self.elements[0]).shape)
        shape[-1] = ndofs
        shape[-2] = ndofs
        matrix = np.zeros(shape)

        for e in self.elements:
            contrib = elem_func(e)
            ix = e.ix()
            matrix[..., ix[0], ix[1]] += contrib

        return matrix

    def solve(self) -> Optional[results.ResultsStaticLinear]:
        self.reassign_dofs()
        self.remove_dofs()
        free_dofs = self.free_dofs()
        constrained_dofs = self.constrained_dofs
        ndofs = sum(node.ndofs() for node in self.nodes)

        iterables = [e for e in self.elements if ElementBehavior.ITERABLE in e.behavior]
        max_iter = 8
        itercnt = 0

        while True:
            K = self.K(np.zeros(ndofs))
            K11 = K[np.ix_(free_dofs, free_dofs)]
            K12 = K[np.ix_(free_dofs, constrained_dofs)]
            K21 = K[np.ix_(constrained_dofs, free_dofs)]
            K22 = K[np.ix_(constrained_dofs, constrained_dofs)]

            # Assemble displacements
            forces = self.assemble_vector(self.nodes, lambda n: n.loads, ndofs)
            displacements = self.assemble_vector(self.nodes, lambda n:n.displacements, ndofs)
            preload = self.assemble_vector(self.elements, lambda e:e.preload, ndofs)
            if forces.shape != displacements.shape or forces.shape != preload.shape:
                forces,displacements,preload = [np.array(a) for a in np.broadcast_arrays(forces, displacements, preload)]

            displacements[..., free_dofs] = np.linalg.solve(K11, ((forces+preload)[..., free_dofs].T - K12 @ displacements[..., constrained_dofs].T)).T
            forces[..., constrained_dofs] = (K21 @ displacements[..., free_dofs].T + K22 @ displacements[..., constrained_dofs].T).T

            for node in self.nodes:
                node.loads = (forces - preload)[..., node.dofs]
                node.displacements = displacements[..., node.dofs]
            self.displacements = displacements

            if not iterables:
                break
            elif itercnt >= max_iter:
                break
            elif all(iterable.do_iterate(K, displacements, forces) for iterable in iterables):
                break

            itercnt += 1

        return results.ResultsStaticLinear(self, forces, displacements)

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
