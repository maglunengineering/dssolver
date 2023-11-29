from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from core.elements import FiniteElement, Node, Rod, Beam, beta
import core.results as results

np.set_printoptions(suppress=True)


class Problem:
    def __init__(self):
        self.nodes:List[Node] = list()
        self.elements:List[FiniteElement] = list()

        self.constrained_dofs = []
        self.forces = None  # Forces (at all nodes, incl removed dofs)
        self.displacements = None  # Assigned at solution ( self.solve() )

        # For incremental analysis
        self.incremental_loads = None
        self.incremental_displacements = None

    def create_beam(self, node1:Node, node2:Node, E=2e5, A=1e5, I=1e5, z=None):
        """
        drawnode:
        0 - Don't draw any nodes
        1 - Draw node at r1
        2 - Draw node at r2
        3 - Draw nodes at r1 and r2
        """
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

    def get_or_create_node(self, r):
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
        print('No node at {}'.format(r))

    def reassign_dofs(self):
        i = 0
        for node in self.nodes:
            ndofs = node.ndofs()
            node.dofs = np.arange(i, i+ndofs)
            i += ndofs

    def upd_obj_displacements(self):
        for node in self.nodes:
            node.displacements = self.displacements[node.dofs]

    def auto_rotation_lock(self):
        """
        Rotation locks all nodes where only Rod elements meet. Useful for truss analysis.
        """
        for node in self.nodes:
            if all(type(element)==Rod for element in node._elements) and 2 not in node.constrained_dofs:
                node.constrained_dofs.append(2)


    def remove_dofs(self):  # Interpret boundary conditions
        self.constrained_dofs = []
        for node in self.nodes:
            self.constrained_dofs.extend(node.dofs[node.constrained_dofs])

    def nonlin_update(self):
        for e in self.elements:
            e.nonlin_update()

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

    def solve(self) -> results.ResultsStaticLinear:
        self.reassign_dofs()
        self.remove_dofs()
        free_dofs = self.free_dofs()
        constrained_dofs = self.constrained_dofs

        K = self.K()
        K11 = K[np.ix_(free_dofs, free_dofs)]
        K12 = K[np.ix_(free_dofs, constrained_dofs)]
        K21 = K[np.ix_(constrained_dofs, free_dofs)]
        K22 = K[np.ix_(constrained_dofs, constrained_dofs)]

        # Assemble displacements
        displacements = np.hstack(tuple(node.displacements for node in self.nodes))
        forces = self.loads

        displacements[free_dofs] = np.linalg.solve(K11, forces[free_dofs] - K12 @ displacements[constrained_dofs])
        forces[constrained_dofs] = K21 @ displacements[free_dofs] + K22 @ displacements[constrained_dofs]

        for node in self.nodes:
            node.loads = forces[node.dofs]
            node.displacements = displacements[node.dofs]
        self.displacements = displacements

        return results.ResultsStaticLinear(self, displacements)

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

    def clone(self):
        node_clones = {}
        for node in self.nodes:
            copy = node.copy()
            node_clones[node] = copy

        cloned_elements = []
        for element in self.elements:
            node1 = node_clones[element.node1]
            node2 = node_clones[element.node2]
            cloned_elements.append(element.clone(node1, node2))

        clone = Problem()
        clone.elements = cloned_elements
        clone.nodes = list(node_clones.values())
        return clone

    @property
    def nodal_coordinates(self):
        nodal_coordinates = np.array([node.r for node in self.nodes])
        return nodal_coordinates

    @property
    def loads(self): # TODO : Should not be a property
        return np.hstack(tuple(node.loads for node in self.nodes))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
