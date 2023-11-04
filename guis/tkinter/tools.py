import tkinter as tk
import numpy as np

class Tool:
    def __init__(self, canvas, root):
        self.canvas = canvas
        self.root = root

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
        self.gui:'DSSGUI' = gui
        self.closest_node_label = None
        self.bcm:tk.Menu = None
        self.rcm:tk.Menu = None
        self.lm:tk.Menu = None

        self.r1 = None
        self.r2 = None  # For self.start_or_end_beam()

        self.build_bc_menu()

        self.bm = tk.Menu(self.rcm, tearoff=0)  # Beam menu
        self.rcm.add_cascade(label='Start/end element', menu=self.bm)
        self.bm.add_command(label='Start element at {}'.format((None, None)),
                             command=lambda: self.start_or_end_beam
                             (r=(self.canvas.transformation_matrix@[self.click_x, self.click_y,1])[0:2]))
        self.bm.add_command(label='Start element at closest',
                            command=lambda: self.start_or_end_beam
                             (r=self.closest_node_label))

    def activate(self):
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Button-3>', self.on_rclick)
        self.canvas.bind('<B1-Motion>', self.canvas.move)
        self.canvas.bind('<Double-Button-1>', self.canvas.scaleup)
        self.canvas.bind('<Double-Button-2>', self.canvas.scaledown)
        self.canvas.bind('<ButtonRelease-1>', self.canvas.on_lbuttonup)

    def on_click(self, event):
        self.set_closest_node((event.x, event.y))
        print("Clicked at canvas", [event.x, event.y], 'problem',
              (self.canvas.transformation_matrix @ [event.x, event.y, 1])[0:2])
        print('Closest node', self.closest_node_label)
        print('r1, r2', self.r1, self.r2)

    def on_rclick(self, event):
        self.click_x = event.x  # Canvas coordinates
        self.click_y = event.y
        self.set_closest_node((event.x, event.y))

        self.bcm.entryconfigure(0, label='Fix node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(1, label='Pin node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(2, label='Roller node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(3, label='Rotation lock node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(4, label='Glider node at {}'.format(self.closest_node_label))

        self.bm.entryconfigure(0, label='Start element at {}'.format((self.canvas.transformation_matrix@[event.x, event.y, 1])[0:2]))
        self.bm.entryconfigure(1, label='Start element at closest: {}'.format(self.closest_node_label))

        self.lm.entryconfigure(0, label='Apply point load at {}'.format(self.closest_node_label))

        #if self.r1 is not None and np.any(self.r2) is None:  # End distr loads or elements
        #    self.bm.entryconfigure(0, label='End element at {}'.format((self.canvas.transformation_matrix@[event.x, event.y, 1])[0:2]))
        #    self.bm.entryconfigure(1, label='End element at closest: {}'.format(self.closest_node_label))

        self.rcm.grab_release()
        self.rcm.tk_popup(event.x_root, event.y_root, 0)

    def set_closest_node(self, xy):
        event_r = np.array(xy)  # xy: Canvas coordinate
        event_r_ = (self.canvas.transformation_matrix @ np.hstack((event_r, 1)))[0:2]  # Problem coordinate
        event_to_node = event_r_ - self.gui.problem.nodal_coordinates
        event_to_node_norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=event_to_node)
        node_id = np.where(event_to_node_norm == np.min(event_to_node_norm))[0][0]

        self.closest_node_label = np.array(self.gui.problem.nodes[node_id].r)  # Problem coordinates

    def build_bc_menu(self):
        """
        Coordinates on these labels are updated when the left mouse button
        is clicked on the canvas. (See self.rightclickmenu )
        """
        self.rcm = tk.Menu(self.root, tearoff=0)

        self.lm = tk.Menu(self.rcm, tearoff=0)  # Load menu
        self.rcm.add_cascade(label='Apply load', menu=self.lm)

        self.lm.add_command(label='Apply point load at {}'.format(self.closest_node_label),
                            command=lambda: self.gui.create_load_dlg(self.gui.problem.node_at(self.closest_node_label)))

        self.bcm = tk.Menu(self.rcm, tearoff=0)  # Boundary condition menu
        self.rcm.add_cascade(label='Apply boundary condition', menu=self.bcm)
        self.bcm.add_command(label='Fix node at {}'.format(self.closest_node_label),
                             command=lambda: self.gui.problem.node_at(self.closest_node_label).fix())
        self.bcm.add_command(label='Pin node at {}'.format(self.closest_node_label),
                             command=lambda: self.gui.problem.node_at(self.closest_node_label).pin())
        self.bcm.add_command(label='Roller node at {}'.format(self.closest_node_label),
                             command=lambda: self.gui.problem.node_at(self.closest_node_label).roller())
        self.bcm.add_command(label='Rotation lock node at {}'.format(self.closest_node_label),
                             command=lambda: self.gui.problem.node_at(self.closest_node_label).lock())
        self.bcm.add_command(label='Glider node at {}'.format(self.closest_node_label),
                             command=lambda: self.gui.problem.node_at(self.closest_node_label).glider())

    def start_or_end_beam(self, r):  # r: Problem coordinates
        if self.r1 is None:  # If r1 does not exist
            self.r1 = np.array(r)

        elif self.r1 is not None and self.r2 is None:  # If r1 does exist and r2 does not exist
            self.r2 = np.array(r)
            self.gui.create_beam_dlg(self.r1, self.r2)
            self.r1 = self.r2 = None