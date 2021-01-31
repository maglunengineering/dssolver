from typing import Iterable,Dict,TypeVar,Callable
import numpy as np
from elements import *

T = TypeVar('T')

class ElementNodeMap:
    def __init__(self, elements:Iterable[FiniteElement], nodal_results:Dict[Node, np.ndarray]):
        self.elements = list(elements)
        self.nodal_results = nodal_results

    def __getitem__(self, element:FiniteElement):
        at_node_1 = self.nodal_results[element.node1]
        at_node_2 = self.nodal_results[element.node2]
        return np.hstack((at_node_1, at_node_2))



class Results:
    pass



class ResultsStaticLinear(Results):
    def __init__(self, nodes:Iterable[Node], elements:Iterable[FiniteElement], displacements:np.ndarray):
        self.node_displacements = dict()
        self.element_map = ElementNodeMap(elements, self.node_displacements)

        for node in nodes:
            self.node_displacements[node] = displacements[node.dofs]