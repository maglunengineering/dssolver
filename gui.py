import tkinter as tk
from tkinter import messagebox
from elements import *
from numpy.linalg import inv
from extras import ResizingCanvas

class Window:
    def __init__(self, root, problem=None, *args, **kwargs):
        self.root = root
        self.root.minsize(width=512, height=512)
        self.problem = problem

        self.mainframe = tk.Frame(self.root, bg='white')
        self.mainframe.pack(fill=tk.BOTH, expand=True)
        self.mainframe.winfo_toplevel().title('DSSolver')
        self.canvas = tk.Canvas()  # Changed later on
        self.rcm = None
        self.bcm = None
        self.bm = None
        self.lm = None
        self.click_x = None
        self.click_y = None
        self.r1 = None
        self.r2 = None  # For self.start_or_end_beam()
        self.closest_node_label = None
        self.dx = -50; self.dy = 100
        self.kx = 1; self.ky = -1
        self.transformation_matrix = np.array([[self.kx,0,self.dx],[0,self.ky,self.dy],[0,0,1]], dtype=float)
        self.tx = 0; self.ty = 0  # Translation
        self.prev_x = None; self.prev_y = None
        self.zoom = 1
        self.ratio = 1


        self.build_grid()
        self.build_menu()
        self.build_banner()
        self.build_canvas()
        self.build_bc_menu()  # Outsourced

        self.displaced_plot = False

        self.problem.create_node((0,0), draw=True)
        self.draw_canvas()

    # Def building functions
    def build_grid(self):
        self.mainframe.columnconfigure(0, weight=1)  #
        self.mainframe.rowconfigure(0, weight=0)  # 'DSSolver' banner
        self.mainframe.rowconfigure(1, weight=1)  # Canvas (resizable)
        self.mainframe.rowconfigure(2, weight=0)  # Output console?

    def build_banner(self):
        banner = tk.Label(self.mainframe, bg='white', text='DSSolver')
        banner.grid(row=0, column=0)

    def build_menu(self):
        topmenu = tk.Menu(self.root)
        self.root.config(menu=topmenu)


        menu_file = tk.Menu(topmenu)
        topmenu.add_cascade(label='File', menu=menu_file)
        menu_file.add_command(label='New case', command=lambda: self.new_problem())


        menu_edit = tk.Menu(topmenu)
        topmenu.add_cascade(label='Edit', menu=menu_edit)
        menu_edit.add_command(label='Create beams',
                              command=lambda: BeamInputMenu(self, self.root, self.problem))
        menu_edit.add_command(label='Redraw',
                              command=lambda: self.draw_canvas())

        topmenu.add_command(label='Solve',
                            command=lambda: self.problem.solve())

        menu_stdcases = tk.Menu(topmenu)
        topmenu.add_cascade(label='Standard load cases', menu = menu_stdcases)
        menu_stdcases.add_command(label='Cantilever beam',
                            command=lambda: self._selftest(1))
        menu_stdcases.add_command(label='Simply supported beam',
                            command=lambda: self._selftest(2))
        menu_stdcases.add_command(label='L beam(s)',
                            command=lambda: self._selftest(3))
        menu_stdcases.add_command(label='Fanned out cantilever beams',
                            command=lambda: self._selftest(4))


        show_menu = tk.Menu(topmenu)
        topmenu.add_cascade(label='Show/hide', menu=show_menu)

        show_menu.add_command(label='Displaced shape',
                            command=lambda: self.draw_displaced())

        topmenu.add_command(label='Autoscale', command=lambda: self.autoscale())
        topmenu.add_command(label='Move to origin', command= lambda: self.move_to())

    def build_bc_menu(self):
        self.rcm = tk.Menu(root, tearoff=0)
        self.lm = tk.Menu(self.rcm, tearoff=0)  # Load menu
        self.rcm.add_cascade(label='Apply load', menu=self.lm)
        self.lm.add_command(label='Apply point load at {}'.format(self.closest_node_label),
                            command=lambda: LoadInputMenu(self, self.root, self.problem))
        self.lm.add_command(label='Apply distributed load from {}'.format(self.closest_node_label),
                            command=lambda: self.start_or_end_distr_load(r=self.closest_node_label))

        self.bcm = tk.Menu(self.rcm, tearoff=0)  # Boundary condition menu
        self.rcm.add_cascade(label='Apply boundary condition', menu=self.bcm)
        self.bcm.add_command(label='Fix node at {}'.format(self.closest_node_label),
                             command=lambda: self.boundary_condition('fix'))
        self.bcm.add_command(label='Pin node at {}'.format(self.closest_node_label),
                             command=lambda: self.boundary_condition('pin'))
        self.bcm.add_command(label='Roller node at {}'.format(self.closest_node_label),
                             command=lambda: self.boundary_condition('roller'))

        '''
        All these labels are updated when the left mouse button 
        is clicked on the canvas. (See self.rightclickmenu )
        '''

        self.bm = tk.Menu(self.rcm, tearoff=0)  # Beam menu
        self.rcm.add_cascade(label='Start/end element', menu=self.bm)
        self.bm.add_cascade(label='Start element at {}'.format((None, None)),
                             command=lambda: self.start_or_end_beam
                             (r=(self.transformation_matrix@[self.click_x, self.click_y,1])[0:2]))
        self.bm.add_cascade(label='Start element at closest',
                            command=lambda: self.start_or_end_beam
                             (r=self.closest_node_label))

    def build_canvas(self):
        self.canvas = ResizingCanvas(self.mainframe, bg='white', highlightthickness=0)
        self.canvas.grid(sticky='nsew')
        self.canvas.grid(row=1)

        self.canvas.bind('<Button-1>', self._printcoords)
        self.canvas.bind('<Button-3>', self.rightclickmenu)
        self.canvas.bind('<Double-Button-1>', self.scaleup)
        self.canvas.bind('<Double-Button-2>', self.scaledown)
        self.canvas.bind('<B1-Motion>', self.move)
        self.canvas.bind('<ButtonRelease-1>', self.reset_prev)

    def rightclickmenu(self, event):
        self.click_x = event.x  # Canvas coordinates
        self.click_y = event.y
        self._closest_node((event.x, event.y))

        self.bcm.entryconfigure(0, label='Fix node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(1, label='Pin node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(2, label='Roller node at {}'.format(self.closest_node_label))

        self.bm.entryconfigure(0, label='Start element at {}'.format((self.transformation_matrix@[event.x, event.y, 1])[0:2]))
        self.bm.entryconfigure(1, label='Start element at closest: {}'.format(self.closest_node_label))

        self.lm.entryconfigure(0, label='Apply point load at {}'.format(self.closest_node_label))
        self.lm.entryconfigure(1, label='Apply distributed load from {}'.format(self.closest_node_label))

        if self.r1 is not None and np.any(self.r2) is None:
            self.bm.entryconfigure(0, label='End element at {}'.format((self.transformation_matrix@[event.x, event.y, 1])[0:2]))
            self.bm.entryconfigure(1, label='End element at closest: {}'.format(self.closest_node_label))
            self.lm.entryconfigure(1, label='End distributed load at {}'.format(self.closest_node_label))

        self.rcm.grab_release()
        self.rcm.tk_popup(event.x_root, event.y_root, 0)

    # Drawing functions
    def draw_canvas(self):
        self.canvas.delete('all')  # Clear the canvas
        if self.displaced_plot:
            self.displaced_plot = False
            self.draw_displaced()
            # This is ugly as hell but works as intended


        linewidth = 2.0
        scale = 50
        for node in self.problem.nodes:

            node_r = (inv(self.transformation_matrix) @ np.hstack((node.r, 1)))[0:2]
            if node.draw:
                node_radius = 2.5
                self.canvas.create_oval(*np.hstack((node_r - node_radius, node_r + node_radius)),
                                                   fill='black', tag='mech')

            if np.any(np.round(node.loads[0:2])):
                # If lump force, draw force arrow
                arrow_start = node_r - node.loads[0:2] / np.linalg.norm(node.loads[0:2]) * scale * np.array([1,1])
                arrow_end = node_r
                self.canvas.create_line(*arrow_start, *arrow_end,
                                        arrow='last', fill='blue', tag='mech')
                self.canvas.create_text(*arrow_start,
                                        text='{}'.format(node.loads[0:2]),
                                        anchor='s', tag='mech')

            if np.alen(node.loads) >= 3 and node.loads[2] != 0:
                # If lump moment, draw moment arrow

                arc_start = node_r + [0, -scale/2] * np.sign(node.loads[2])
                arc_mid = node_r + [scale/2, 0] * np.sign(node.loads[2])
                arc_end = node_r + [0, scale/2] * np.sign(node.loads[2])

                self.canvas.create_line(*arc_start, *arc_mid, *arc_end,
                                        smooth = True,
                                        arrow='last', fill='blue', tag='mech')
                self.canvas.create_text(*arc_start,
                                        text='{}'.format(node.loads[2]),
                                        anchor='s', tag='mech')


            if node.boundary_condition:
                # If boundary condition, draw boundary condition symbol
                self.draw_boundary_condition(node.boundary_condition,
                                             position=node_r,
                                             draw_angle=np.average([beam.angle for beam in node.beams]))

        for beam in self.problem.beams:
            beam_r1 = (inv(self.transformation_matrix) @ np.hstack((beam.r1, 1)))[0:2]
            beam_r2 = (inv(self.transformation_matrix) @ np.hstack((beam.r2, 1)))[0:2]
            if isinstance(beam, Beam):
                # Draw the beam
                self.canvas.create_line(*np.hstack((beam_r1, beam_r2)),
                                        width=linewidth, tag = 'mech')
                if beam.distributed_load:
                    # Draw a distributed load
                    angle = beam.angle
                    c, s = np.cos(-angle), np.sin(-angle)
                    rotation = np.array([[c, -s], [s, c]])
                    p1 = beam_r1 + rotation@[0, -scale/2]
                    p2 = beam_r1 + rotation@[0, -scale]
                    p3 = beam_r2 + rotation@[0, -scale/2]
                    p4 = beam_r2 + rotation@[0, -scale]

                    self.canvas.create_line(*p1, *p3)
                    self.canvas.create_line(*p2, *p4)
                    for x0,y0 in zip(np.linspace(p2[0], p4[0], 3, endpoint=True),
                                      np.linspace(p2[1], p4[1], 3, endpoint=True)):
                        x1, y1 = np.array([x0, y0]) + rotation @ [0, scale/2]
                        arrow = 'first' if (beam.beta @ beam.member_loads)[2] > 0 else 'last'
                        self.canvas.create_line(x0, y0, x1, y1,
                                                arrow=arrow)
                        # Bugs out / draws incorrectly / /// if beam is at 90 degrees




            elif isinstance(beam, Rod):
                self.canvas.create_line(*np.hstack((beam_r1, beam_r2)),
                                        width=linewidth/2, tag='mech')

    def draw_boundary_condition(self, bc_type, position, draw_angle=0):
        """
        :param bc_type: 'fixed', 'pinned', 'roller', 'fix' or 'pin'
        """
        scale = 20
        linewidth = 3


        if bc_type == 'fixed':
            r0 = position  # np.array
            angle = draw_angle
            c, s = np.cos(-angle), np.sin(-angle)
            rotation = np.array([[c, -s], [s, c]])

            self.canvas.create_line(*(r0 + rotation@[0, scale]), *(r0 + rotation@[0, -scale]),
                                                          width=linewidth, fill='black', tag='bc')
            for offset in np.linspace(0, 2*scale, 6):
                self.canvas.create_line(*(r0 + rotation@[0, -scale+offset]),
                                        *(r0 + rotation@[0, -scale+offset] + rotation@[-scale/2, scale/2]),
                                        width=linewidth, fill='black', tag='bc')

        elif bc_type == 'pinned' or bc_type == 'roller':
            r0 = position  # np.array
            k = 1.5  # constant - triangle diameter

            self.canvas.create_oval(*(r0 - scale/4), *(r0 + scale/5))
            self.canvas.create_line(*r0, *(r0 + np.array([-np.sin(np.deg2rad(30)),
                                                          np.cos(np.deg2rad(30))]) * k*scale),
                                    width=linewidth, fill='black', tag='bc')
            self.canvas.create_line(*r0, *(r0 + np.array([np.sin(np.deg2rad(30)),
                                                          np.cos(np.deg2rad(30))])*k*scale),
                                    width=linewidth, fill='black', tag='bc')

            self.canvas.create_line(*(r0 + (np.array([-np.sin(np.deg2rad(30)),
                                                          np.cos(np.deg2rad(30))])
                                            + np.array([-1.4/(k*scale), 0])
                                            ) * k * scale),
                                    *(r0 + (np.array([np.sin(np.deg2rad(30)),
                                                     np.cos(np.deg2rad(30))])
                                            + np.array([1.4/(k*scale), 0])
                                            ) * k * scale),
                                    width=linewidth, fill='black', tag='bc')
            if bc_type == 'roller':
                self.canvas.create_line(*(r0 + np.array([-np.sin(np.deg2rad(30)),
                                                         np.cos(np.deg2rad(30))])*k*scale
                                                        + np.array([-scale/2, scale/4])),
                                        *(r0 + np.array([np.sin(np.deg2rad(30)),
                                                         np.cos(np.deg2rad(30))])*k*scale)
                                                        + np.array([scale/2, scale/4]),
                                        width=linewidth, fill='black', tag='bc')

    def draw_displaced(self):

        node_radius = 2.5
        for node in self.problem.nodes:
            if node.draw:
                node.r_ = (inv(self.transformation_matrix) @ [node.r[0], node.r[1], 1])[0:2]
                self.canvas.create_oval(*np.hstack((node.r_ - node_radius + node.displacements[0:2]*[-1,1],
                                                    node.r_ + node_radius + node.displacements[0:2]*[-1,1])),
                                        fill='red', tag='mech_disp')

                if np.any(np.round(node.loads[0:2])):  # If node is loaded
                    scale = 50
                    arrow_start = node.r_ - node.loads[0:2]/np.linalg.norm(node.loads[0:2])*scale*[1, 1]
                    arrow_end = node.r_
                    self.canvas.create_line(*arrow_start + node.displacements[0:2]*[-1,1],
                                            *arrow_end + node.displacements[0:2]*[-1,1],
                                            arrow='last', fill='blue', tag='mech_disp')

                self.canvas.create_text(*node.r_ + node.displacements[0:2],
                                        text='{}'.format(np.round(node.displacements, 1)),
                                        anchor='sw', tag='mech_disp')

        for beam in self.problem.beams:
            beam.r1_ = (inv(self.transformation_matrix) @ [beam.r1[0], beam.r1[1], 1])[0:2]
            beam.r2_ = (inv(self.transformation_matrix) @ [beam.r2[0], beam.r2[1], 1])[0:2]
            self.canvas.create_line(*np.hstack((beam.r1_ + beam.nodes[0].displacements[0:2]*[-1,1],
                                                beam.r2_ + beam.nodes[1].displacements[0:2]*[-1,1])),
                                        fill='red', tag='mech_disp', dash=(1,))

        if hasattr(self, 'displaced_plot'):
            if self.displaced_plot:
                self.canvas.delete('mech_disp')
        self.displaced_plot = not self.displaced_plot

    # Scaling and moving functions
    def scaleup(self, event):
        self.zoom *= 0.8
        self.transformation_matrix[0:2, 0:2] = self.transformation_matrix[0:2, 0:2]*0.8
        self.draw_canvas()

    def scaledown(self, event):
        self.zoom *= 1.2
        self.transformation_matrix[0:2, 0:2] = self.transformation_matrix[0:2, 0:2]*1.2
        self.draw_canvas()

    def autoscale(self):
        canvas_size = np.sqrt(self.canvas.width**2 + self.canvas.height**2)
        model_size = self.problem.model_size()
        self.ratio = 1/2 * canvas_size / model_size  # Big if model size is small
        self.transformation_matrix[0:2,0:2] = np.array([[1,0],[0,-1]]) / self.ratio


        self.zoom = self.ratio
        self.move_to((self.canvas.width/2, self.canvas.height/2))
        self.draw_canvas()

    def move(self, event):
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = event.x
            self.prev_y = event.y

        self.transformation_matrix[0:,2] = self.transformation_matrix[:,2] + np.array([-self.tx, self.ty, 0])

        self.tx = (event.x - self.prev_x) * self.transformation_matrix[0,0]
        self.ty = (event.y - self.prev_y) * self.transformation_matrix[0,0]
        self.prev_x = event.x
        self.prev_y = event.y
        self.draw_canvas()

    def move_to(self, xy=(200, -200)):
        # Moves the problem csys origin to canvas csys (xy)
        x,y = xy
        self.transformation_matrix[0:3,2] = np.array([-self.transformation_matrix[0,0]*(x-1),
                                                      -self.transformation_matrix[1,1]*(y-1),
                                                      1])
        self.draw_canvas()

    # Mechanical functions
    def start_or_end_beam(self, r):  # r: Problem coordinates
        if self.r1 is None:  # If r1 does not exist
            self.r1 = np.array(r)

        elif self.r1 is not None and self.r2 is None:  # If r1 does exist and r2 does not exist
            self.r2 = np.array(r)
            BeamInputMenu(self, self.root, self.problem,
                          def_r1=self.r1,
                          def_r2=self.r2)
            self.r1 = self.r2 = None

    def start_or_end_distr_load(self, r):
        if self.r1 is None:  # If r1 does not exist
            self.r1 = np.array(r)

        elif self.r1 is not None and self.r2 is None:  # If r1 does exist and r2 does not exist
            self.r2 = np.array(r)
            DistrLoadInputMenu(self, self.root, self.problem,
                               self.r1,
                               self.r2)
            self.r1 = self.r2 = None

    def _closest_node(self, xy):
        event_r = np.array(xy)  # xy: Canvas coordinate
        event_r_ = (self.transformation_matrix @ np.hstack((event_r, 1)))[0:2]  # Problem coordinate
        event_to_node = event_r_ - self.problem.nodal_coordinates
        event_to_node_norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=event_to_node)
        node_id = np.where(event_to_node_norm == np.min(event_to_node_norm))[0][0]

        self.closest_node_label = np.array(self.problem.nodes[node_id].r)  # Problem coordinates

    def boundary_condition(self, bc):
        """
        :param bc: 'fixed', 'fix', 'pinned', 'pin' or 'roller'
        """
        if bc == 'fix':
            self.problem.fix(self.problem.node_at(self.closest_node_label))
        elif bc == 'pin':
            self.problem.pin(self.problem.node_at(self.closest_node_label))
        elif bc == 'roller':
            self.problem.roller(self.problem.node_at(self.closest_node_label))
        else:
            pass
        self.draw_canvas()

    def reset_prev(self, event):
        self.prev_x = None
        self.prev_y = None

    def new_problem(self):
        self.problem = Problem()

        self.rcm = None
        self.bcm = None
        self.bm = None
        self.click_x = None
        self.click_y = None
        self.r1 = None
        self.r2 = None  # For self.start_or_end_beam()
        self.closest_node_label = None
        self.dx = -50
        self.dy = 100
        self.kx = 1
        self.ky = -1
        self.transformation_matrix = np.array([[self.kx, 0, self.dx], [0, self.ky, self.dy], [0, 0, 1]], dtype=float)
        self.tx = 0
        self.ty = 0  # Translation
        self.prev_x = None
        self.prev_y = None
        self.zoom = 1
        self.ratio = 1

        self.build_grid()
        self.build_menu()
        self.build_banner()
        self.build_canvas()
        self.build_bc_menu()

        self.displaced_plot = False

        self.problem.create_node((0, 0), draw=True)
        self.draw_canvas()

    def _printcoords(self, event):
        self._closest_node((event.x, event.y))
        print("Clicked at canvas", [event.x, event.y], 'problem', (self.transformation_matrix@[event.x,event.y,1])[0:2])
        print('Closest node', self.closest_node_label)
        print('r1, r2', self.r1, self.r2)

    def _selftest(self, loadcase = 1):
        if loadcase == 1:  # Cantilever beam, point load
            self.problem.create_beams((0,0), (707,-707), n=5)
            self.problem.fix(self.problem.node_at((0,0)))
            #self.problem.load_members_distr((0,0),(707,-707),1)

            self.draw_canvas()

        if loadcase == 2:  # Simply supported beam, no load
            self.problem.create_beams((0,0), (1000,0))
            self.problem.pin(self.problem.node_at((0,0)))
            self.problem.roller(self.problem.node_at((1000,0)))

            self.draw_canvas()

        if loadcase == 3:  # L beam(s), point load
            self.problem.create_beams((0,0), (1000,0))
            self.problem.create_beams((1000,0), (1000,-1000))
            self.problem.fix(self.problem.node_at((0,0)))

            self.draw_canvas()

        if loadcase == 4:  # Fanned out cantilever beams with load=10 distr loads
            for point in ((1000,0),(707,-707),(0,-1000),(-707,-707),(-1000,0)):
                self.problem.create_beams((0,0),point, n=2)
                self.problem.load_members_distr((0,0),point, load=10)

            self.problem.fix(self.problem.node_at((0,0)))


            self.draw_canvas()

        self.autoscale()


class LoadInputMenu:
    def __init__(self, window, root, problem):
        """
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        top = self.top = tk.Toplevel(root)
        self.top.winfo_toplevel().title('Apply load')

        self.window = window
        self.root = root
        self.problem = problem

        self.label = tk.Label(top, text='Apply load at node')
        self.label.grid(row=0, column=0)

        entrykeys = ['Fx', 'Fy', 'M']
        entries = [self.e_fx, self.e_fy, self.e_m] = [tk.Entry(top) for _ in range(3)]

        for idx, entry in enumerate(entries):
            entry.grid(row=idx+1, column=1)

        for idx, key in enumerate(entrykeys):
            tk.Label(top, text=key).grid(row=idx+1)

        for idx, entry in enumerate((self.e_fx, self.e_fy, self.e_m)):
            try:
                entry.insert(0, self.problem.nodes[self.problem.node_at(self.window.closest_node_label)].loads[idx])
            except:
                entry.insert(0, 0)

        self.e_fx.focus_set()

        self.b = tk.Button(top, text='Apply load', command=self.cleanup)
        self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))
        self.b.grid(row=4, column=0)

    def cleanup(self):
        self.problem.load_node( self.problem.node_at(self.window.closest_node_label),
                                load = (eval(self.e_fx.get()),
                                        eval(self.e_fy.get()),
                                        eval(self.e_m.get()))
                                )

        self.window.draw_canvas()

        self.top.destroy()

class DistrLoadInputMenu:
    def __init__(self, window, root, problem, r1 = np.array([0,0]), r2 = np.array([0,0])):
        """
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        top = self.top = tk.Toplevel(root)
        self.top.winfo_toplevel().title('Apply distributed load')

        self.window = window
        self.root = root
        self.problem = problem

        self.label = tk.Label(top, text='Apply distributed load')
        self.label.grid(row=0, column=0)

        entrykeys = ['Starting node', 'Ending node', 'Load magnitude']
        entries = [self.e_r1, self.e_r2, self.e_load] = [tk.Entry(top) for _ in range(3)]

        for idx, entry in enumerate(entries):
            entry.grid(row=idx+1, column=1)

        for idx, key in enumerate(entrykeys):
            tk.Label(top, text=key).grid(row=idx+1)

        self.e_r1.insert(0, '{},{}'.format(r1[0], r1[1]))
        self.e_r2.insert(0, '{},{}'.format(r2[0], r2[1]))
        self.e_load.insert(0, '0')

        self.e_r2.focus_set()

        self.b = tk.Button(top, text='Apply load', command=self.cleanup)
        self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))
        self.b.grid(row=4, column=0)

    def cleanup(self):
        r1 = np.array(eval(self.e_r1.get()))
        r2 = np.array(eval(self.e_r2.get()))
        load = float(eval(self.e_load.get()))
        self.problem.load_members_distr(r1=r1, r2=r2, load=load)


        self.window.draw_canvas()

        self.top.destroy()

class BeamInputMenu:
    def __init__(self, window, root, problem, def_r1=np.array([0,0]), def_r2=np.array([1000,0])):
        """
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """

        top = self.top = tk.Toplevel(root)
        self.top.winfo_toplevel().title('Create element(s)')
        self.window = window
        self.root = root
        self.problem = problem

        self.label = tk.Label(top, text='Create element(s) between:')
        self.label.grid(row=0, column=0)


        entrykeys = ['Node 1', 'Node 2', 'Area', 'Elastic modulus', '2nd mmt of area', 'No. of elements']
        entries = [self.e_r1, self.e_r2, self.e_A, self.e_E, self.e_I, self.e_n] = \
            [tk.Entry(top) for _ in range(6)]

        for idx, entry in enumerate(entries):
            entry.grid(row=idx+1, column=1, columnspan=2)

        for idx, key in enumerate(entrykeys):
            tk.Label(top, text=key).grid(row=idx+1)

        defaults = (1e5, 2e5, 1e5, 10)
        for value, entry in zip(defaults, (self.e_A, self.e_E, self.e_I, self.e_n)):
            entry.insert(0,int(value))

        self.e_r1.insert(0, '{},{}'.format(def_r1[0], def_r1[1]))
        self.e_r2.insert(0, '{},{}'.format(def_r2[0], def_r2[1]))
        self.e_r1.focus_set()


        self.b = tk.Button(top, text='Create element(s)', command=self.cleanup)
        self.b.grid(row=8, column=0)
        self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))

        # Adding 'beam' or 'rod' radio buttons
        self.k = tk.IntVar()

        b1 = tk.Radiobutton(self.top, text='Beam', variable=self.k, value=1,
                            command=lambda: self.check_choice())#.grid(row=8, column=1)
        b1.grid(row=8, column=1)
        b2 = tk.Radiobutton(self.top, text='Rod', variable=self.k, value=2,
                            command=lambda: self.check_choice())#.grid(row=8, column=2)
        b2.grid(row=8, column=2)
        self.k.set(1)
        self.check_choice()
        print(self.k.get())

    def check_choice(self):
        if self.k.get() == 2:
            self.e_I.config(state=tk.DISABLED)
            self.e_n.delete(0, tk.END)
            self.e_n.insert(0,'1')
            self.e_n.config(state=tk.DISABLED)

        elif self.k.get() == 1:
            self.e_I.config(state=tk.NORMAL)
            self.e_n.config(state=tk.NORMAL)

    def cleanup(self):
        r1 = np.array(eval(self.e_r1.get()))
        r2 = np.array(eval(self.e_r2.get()))
        A = eval(self.e_A.get())
        E = eval(self.e_E.get())
        I = eval(self.e_I.get())
        n = eval(self.e_n.get())

        if self.k.get() == 1:  # Beam
            self.problem.create_beams(r1, r2, A=A, E=E, I=I, n=n)
            self.window.draw_canvas()
            if len(self.problem.beams) == n:
                self.window.autoscale()

        elif self.k.get() == 2:  # Rod
            self.problem.create_rod(r1, r2, A=A, E=E)
            self.window.draw_canvas()
            if len(self.problem.beams) == 1:
                self.window.autoscale()

        self.top.destroy()

class NodeInputMenu:
    def __init__(self, root, problem):
        top = self.top = tk.Toplevel(root)
        self.root = root
        self.problem = problem

        self.label = tk.Label(top, text='Create node at:')
        self.label.pack()

        self.entry = tk.Entry(top)
        self.entry.pack()

        self.b = tk.Button(top, text='Create node', command=self.cleanup)
        self.b.pack()

    def cleanup(self):
        value = np.array(eval(self.entry.get()))
        self.problem.create_node(value)
        self.top.destroy()
        return value


if __name__ == '__main__':
    p = Problem()

    root = tk.Tk()
    w = Window(root, problem=p)

    root.mainloop()