import tkinter as tk
import numpy as np

from core import elements, solvers

class Tool:
    def __init__(self, canvas, root):
        self._canvas = canvas
        self._root = root

    def activate(self):
        pass

    def deactivate(self):
        pass

    def on_click(self, event):
        pass

    def on_rclick(self, event):
        pass

class ToolSelect(Tool):
    def __init__(self, gui, canvas, root):
        super().__init__(canvas, root)
        self._gui: 'DSSGUI' = gui
        self._closest_node_label = None
        self._bcm:tk.Menu = None
        self._rcm:tk.Menu = None
        self._lm:tk.Menu = None

        self._r1 = None
        self._r2 = None  # For self.start_or_end_beam()

        self.build_bc_menu()

        self._bm = tk.Menu(self._rcm, tearoff=0)  # Beam menu
        self._rcm.add_cascade(label='Start/end element', menu=self._bm)
        self._bm.add_command(label='Start element at {}'.format((None, None)),
                             command=lambda: self.start_or_end_beam
                             (r=(self._canvas.transformation_matrix @ [self._click_x, self._click_y, 1])[0:2]))
        self._bm.add_command(label='Start element at closest',
                             command=lambda: self.start_or_end_beam
                             (r=self._closest_node_label))

    def activate(self):
        self._canvas.bind('<Button-1>', self.on_click)
        self._canvas.bind('<Button-3>', self.on_rclick)
        self._canvas.bind('<B2-Motion>', self._canvas.move)
        def on_mousewheel(event):
            if event.delta > 0:
                self._canvas.scaleup(event)
            else:
                self._canvas.scaledown(event)
        self._canvas.bind('<MouseWheel>', on_mousewheel)
        #self.canvas.bind('<Double-Button-1>', self.canvas.scaleup)
        #self.canvas.bind('<Double-Button-2>', self.canvas.scaledown)
        self._canvas.bind('<ButtonRelease-2>', self._canvas.on_mbuttonup)

    def on_click(self, event):
        obj = self._canvas.get_closest((event.x, event.y))
        print(f'Selected obj: {obj}')
        if obj:
            self._gui.set_selection(obj)
        self.set_closest_node((event.x, event.y))
        print("Clicked at canvas", [event.x, event.y], 'problem',
              (self._canvas.transformation_matrix @ [event.x, event.y, 1])[0:2])
        print('Closest node', self._closest_node_label)
        print('r1, r2', self._r1, self._r2)

    def on_rclick(self, event):
        self._click_x = event.x  # Canvas coordinates
        self._click_y = event.y
        self.set_closest_node((event.x, event.y))

        self._bcm.entryconfigure(0, label='Fix node at {}'.format(self._closest_node_label))
        self._bcm.entryconfigure(1, label='Pin node at {}'.format(self._closest_node_label))
        self._bcm.entryconfigure(2, label='Roller node at {}'.format(self._closest_node_label))
        self._bcm.entryconfigure(3, label='Rotation lock node at {}'.format(self._closest_node_label))
        self._bcm.entryconfigure(4, label='Glider node at {}'.format(self._closest_node_label))

        self._bm.entryconfigure(0, label='Start element at {}'.format((self._canvas.transformation_matrix @ [event.x, event.y, 1])[0:2]))
        self._bm.entryconfigure(1, label='Start element at closest: {}'.format(self._closest_node_label))

        self._lm.entryconfigure(0, label='Apply point load at {}'.format(self._closest_node_label))

        #if self.r1 is not None and np.any(self.r2) is None:  # End distr loads or elements
        #    self.bm.entryconfigure(0, label='End element at {}'.format((self.canvas.transformation_matrix@[event.x, event.y, 1])[0:2]))
        #    self.bm.entryconfigure(1, label='End element at closest: {}'.format(self.closest_node_label))

        self._rcm.grab_release()
        self._rcm.tk_popup(event.x_root, event.y_root, 0)

    def set_closest_node(self, xy):
        event_r = np.array(xy)  # xy: Canvas coordinate
        event_r_ = (self._canvas.transformation_matrix @ np.hstack((event_r, 1)))[0:2]  # Problem coordinate
        event_to_node = event_r_ - self._gui.problem.nodal_coordinates
        event_to_node_norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=event_to_node)
        node_id = np.where(event_to_node_norm == np.min(event_to_node_norm))[0][0]

        self._closest_node_label = np.array(self._gui.problem.nodes[node_id].r)  # Problem coordinates

    def build_bc_menu(self):
        """
        Coordinates on these labels are updated when the left mouse button
        is clicked on the canvas. (See self.rightclickmenu )
        """
        self._rcm = tk.Menu(self._root, tearoff=0)

        self._lm = tk.Menu(self._rcm, tearoff=0)  # Load menu
        self._rcm.add_cascade(label='Apply load', menu=self._lm)

        self._lm.add_command(label='Apply point load at {}'.format(self._closest_node_label),
                             command=lambda: self._gui.create_load_dlg(self._gui.problem.node_at(self._closest_node_label)))

        self._bcm = tk.Menu(self._rcm, tearoff=0)  # Boundary condition menu
        self._rcm.add_cascade(label='Apply boundary condition', menu=self._bcm)
        self._bcm.add_command(label='Fix node at {}'.format(self._closest_node_label),
                              command=lambda: self._gui.problem.node_at(self._closest_node_label).fix())
        self._bcm.add_command(label='Pin node at {}'.format(self._closest_node_label),
                              command=lambda: self._gui.problem.node_at(self._closest_node_label).pin())
        self._bcm.add_command(label='Roller node at {}'.format(self._closest_node_label),
                              command=lambda: self._gui.problem.node_at(self._closest_node_label).roller())
        self._bcm.add_command(label='Rotation lock node at {}'.format(self._closest_node_label),
                              command=lambda: self._gui.problem.node_at(self._closest_node_label).lock())
        self._bcm.add_command(label='Glider node at {}'.format(self._closest_node_label),
                              command=lambda: self._gui.problem.node_at(self._closest_node_label).glider())

    def start_or_end_beam(self, r):  # r: Problem coordinates
        if self._r1 is None:  # If r1 does not exist
            self._r1 = np.array(r)

        elif self._r1 is not None and self._r2 is None:  # If r1 does exist and r2 does not exist
            self._r2 = np.array(r)
            self._gui.create_beam_dlg(self._r1, self._r2)
            self._r1 = self._r2 = None


class ToolDispl(Tool):
    def __init__(self, gui, canvas, root):
        super().__init__(canvas, root)
        self._gui:'DSSGUI' = gui

        self._dragging = False
        self._snap_obj = None
        self._last_click_xy = np.empty(2)
        self._solver = solvers.LinearSolver(gui)

        self.displace = False


    def activate(self):
        self._canvas.bind('<Button-1>', self.on_click)
        self._canvas.bind('<Button-3>', self.on_rclick)
        self._canvas.bind('<B1-Motion>', self._drag)

        def on_mousewheel(event):
            if event.delta > 0:
                self._canvas.scaleup(event)
            else:
                self._canvas.scaledown(event)

        self._canvas.bind('<MouseWheel>', on_mousewheel)
        self._canvas.bind('<ButtonRelease-1>', self.on_lbuttonup)

    def _drag(self, event):
        if not self._dragging or not self._snap_obj:
            return

        pt = self._canvas.canvas_to_problem((event.x, event.y))
        if self.displace:
            self._snap_obj.displacements = np.array([*(pt - self._last_click_xy), 0])
        else:
            self._snap_obj._r = pt
        self._gui.draw_canvas()
        self._solver.solveall()

    def on_click(self, event):
        self._dragging = True
        self._last_click_xy = self._canvas.canvas_to_problem((event.x, event.y))
        if not self._snap_obj:
            obj = self._canvas.get_closest((event.x, event.y))
            if isinstance(obj, elements.Node):
                self._snap_obj = obj
                if self.displace:
                    self._snap_obj.constrained_dofs = [0, 1]

    def on_lbuttonup(self, event):
        self._dragging = False
        self._last_click_xy = np.empty(2)
        self._snap_obj = None
