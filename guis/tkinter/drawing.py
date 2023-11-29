import numpy as np
from core.elements import Node, FiniteElement
import extras


_register = {}


class NodeDrawer:
    settings = {'Loads': True,
                'Boundary conditions': True}


    def __init__(self):
        pass

    @staticmethod
    def draw_on_canvas(self, canvas: extras.DSSCanvas, **kwargs):
        canvas.draw_node(self.r, 2.5, **kwargs)

        # If lump force, draw an arrow
        if NodeDrawer.settings['Loads']:
            NodeDrawer.draw_loads(self, canvas, **kwargs)

        if NodeDrawer.settings['Boundary conditions']:
            NodeDrawer.draw_boundary_condition(self, canvas, **kwargs)

        return self.r

    @staticmethod
    def draw_loads(self, canvas: extras.DSSCanvas):
        scale = 100
        pos = self.r
        if np.any(np.round(self.loads[0:2])):
            arrow_start = pos
            arrow_end = pos + self.loads[0:2] / np.linalg.norm(self.loads[0:2]) * scale
            canvas.draw_line(arrow_start, arrow_end,
                             arrow='last', fill='blue', tag='mech')
            canvas.draw_text((arrow_start + arrow_end) / 2,
                             '{}'.format(self.loads[0:2]),
                             anchor='sw', tag='mech')

        # If moment, draw a circular arrow
        if len(self.loads) >= 3 and self.loads[2] != 0:
            sign = np.sign(self.loads[2])
            arc_start = pos + np.array([0, -scale / 2]) * sign
            arc_mid = pos + np.array([scale / 2, 0]) * sign
            arc_end = pos + np.array([0, scale / 2]) * sign

            arrow = 'first' if sign == 1 else 'last'
            canvas.draw_arc(arc_start, arc_mid, arc_end,
                            smooth=True,
                            arrow=arrow, fill='blue', tag='mech')
            canvas.draw_text(arc_start,
                             text='{}'.format(np.round(self.loads[2], 0)),
                             anchor='ne', tag='mech')

    @staticmethod
    def draw_boundary_condition(self, canvas: extras.DSSCanvas):
        scale = 50
        linewidth = 2
        pos = self.r + self.displacements[0:2]

        if self.constrained_dofs == [0,1,2]:
            angle_vector = sum(n.r - self.r for n in self.connected_nodes())
            angle = np.arctan2(*angle_vector[::-1])
            c, s = np.cos(-angle), np.sin(-angle)
            rotation = np.array([[c, -s], [s, c]])

            canvas.draw_line((pos + rotation @ [0, scale]), (pos + rotation @ [0, -scale]),
                             width=linewidth, fill='black', tag='bc')
            for offset in np.linspace(0, 2 * scale, 6):
                canvas.draw_line((pos + rotation @ [0, -scale + offset]),
                                 (pos + rotation @ [0, -scale + offset] + rotation @ [-scale / 2, scale / 2]),
                                 width=linewidth, fill='black', tag='bc')

        elif self.constrained_dofs == [0,1] or self.constrained_dofs == [1]:
            k = 1.5  # constant - triangle diameter

            canvas.draw_oval((pos - scale / 4), (pos + scale / 5))
            canvas.draw_line(pos, (pos + np.array([-np.sin(np.deg2rad(30)),
                                                   np.cos(np.deg2rad(30))]) * k * scale),
                             width=linewidth, fill='black', tag='bc')
            canvas.draw_line(pos, (pos + np.array([np.sin(np.deg2rad(30)),
                                                   np.cos(np.deg2rad(30))]) * k * scale),
                             width=linewidth, fill='black', tag='bc')

            canvas.draw_line((pos + (np.array([-np.sin(np.deg2rad(30)),
                                               np.cos(np.deg2rad(30))])
                                     + np.array([-1.4 / (k * scale), 0])
                                     ) * k * scale),
                             (pos + (np.array([np.sin(np.deg2rad(30)),
                                               np.cos(np.deg2rad(30))])
                                     + np.array([1.4 / (k * scale), 0])
                                     ) * k * scale),
                             width=linewidth, fill='black', tag='bc')
            if self.constrained_dofs == [1]:
                canvas.draw_line((pos + np.array([-np.sin(np.deg2rad(30)),
                                                  np.cos(np.deg2rad(30))]) * k * scale
                                  + np.array([-scale / 2, scale / 4])),
                                 (pos + np.array([np.sin(np.deg2rad(30)),
                                                  np.cos(np.deg2rad(30))]) * k * scale)
                                 + np.array([scale / 2, scale / 4]),
                                 width=linewidth, fill='black', tag='bc')

        elif self.constrained_dofs == [0]:
            k = 1.5  # constant - triangle diameter

            canvas.draw_oval((pos - scale / 4), (pos + scale / 5))
            canvas.draw_line(pos, (pos + np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))]) * k * scale),
                             width=linewidth, fill='black', tag='bc')
            canvas.draw_line(pos, (pos + np.array([np.cos(np.deg2rad(30)), -np.sin(np.deg2rad(30))]) * k * scale),
                             width=linewidth, fill='black', tag='bc')

            canvas.draw_line((pos + (np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
                                     + np.array([-1.4 / (k * scale), 0])
                                     ) * k * scale),
                             (pos + (np.array([np.cos(np.deg2rad(30)),
                                               -np.sin(np.deg2rad(30))])
                                     + np.array([1.4 / (k * scale), 0])
                                     ) * k * scale),
                             width=linewidth, fill='black', tag='bc')

            canvas.draw_line((pos + np.array([np.cos(np.deg2rad(30)),
                                              -np.sin(np.deg2rad(30))]) * k * scale
                              + np.array([scale / 4, 2 * scale])),
                             (pos + np.array([np.cos(np.deg2rad(30)),
                                              -np.sin(np.deg2rad(30))]) * k * scale)
                             + np.array([scale / 4, -scale / 2]),
                             width=linewidth, fill='black', tag='bc')


        elif self.constrained_dofs == [2]:
            canvas.draw_oval((pos + np.array([-scale, -scale])),
                             (pos - np.array([-scale, -scale])),
                             width=linewidth, tag='bc')
            canvas.draw_line(pos, (pos + np.array([scale / 2, -scale]) * 1.4),
                             width=linewidth, fill='black', tag='bc')

        elif self.constrained_dofs == [0,2]:
            angle = 0  # Could be pi
            c, s = np.cos(-angle), np.sin(-angle)
            rotation = np.array([[c, -s], [s, c]])

            canvas.draw_line((pos + rotation @ [0, scale]), (pos + rotation @ [0, -scale]),
                             width=linewidth, fill='black', tag='bc')
            canvas.draw_oval((pos + rotation @ [0, -scale / 4]), (pos + rotation @ [scale / 2, -3 * scale / 4]))
            canvas.draw_oval((pos + rotation @ [0, scale / 4]), (pos + rotation @ [scale / 2, 3 * scale / 4]))
            canvas.draw_line((pos + rotation @ [scale / 2, 0] + rotation @ [0, scale]),
                             (pos + rotation @ [scale / 2, 0] + rotation @ [0, -scale]),
                             width=linewidth, fill='black', tag='bc')

            for offset in np.linspace(0, 2 * scale, 6):
                canvas.draw_line((pos + rotation @ [scale / 2, -scale + offset]),
                                 (pos + rotation @ [scale / 2, -scale + offset]
                                  + rotation @ [scale / 2, scale / 2]),
                                 width=linewidth, fill='black', tag='bc')


class ElementDrawer:
    settings = {'Displaced': True}

    def __init__(self):
        pass

    @staticmethod
    def draw_on_canvas(self, canvas, **kwargs):
        nodes = list(self.nodes)
        for node1, node2 in zip(self.nodes, self.nodes[1:]):
            canvas.draw_line(node1.r, node2.r, **kwargs)
            if ElementDrawer.settings['Displaced']:
                canvas.draw_line(node1.r + node1.displacements[0:2],
                                 node2.r + node2.displacements[0:2],
                                 fill='red', dash=(1,), **kwargs)
        if len(self.nodes) > 2:
            canvas.draw_line(self.nodes[-1].r, self.nodes[0].r, **kwargs)
            if ElementDrawer.settings['Displaced']:
                canvas.draw_line(self.nodes[-1].r + self.nodes[-1].displacements[0:2],
                                 self.nodes[0].r + self.nodes[0].displacements[0:2],
                                 fill='red', dash=(1,), **kwargs)

        return 1/len(self.nodes) * sum(node.r for node in self.nodes)


_register[Node] = NodeDrawer
_register[FiniteElement] = ElementDrawer

def get_drawer(obj):
    for T in type(obj).__mro__:
        if T in _register:
            return _register[T]
