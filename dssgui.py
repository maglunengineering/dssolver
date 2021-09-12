import sys
import os
import pickle
from tkinter import filedialog

from problem import *
import elements
from elements import *
from extras import *
import plugins
import solvers
import report_plugin
from results import ResultsViewer

np.set_printoptions(precision=2, suppress=True)
inv = np.linalg.inv

class DSS:
    def __init__(self, root:tk.Tk, problem=None, *args, **kwargs):
        self.root = root
        self.root.minsize(width=1024, height=640)
        self.icon = icon if icon else None
        self.root.iconbitmap(self.icon)
        self.problem = problem

        self.mainframe = tk.Frame(self.root, bg='white')
        self.mainframe.pack(fill=tk.BOTH, expand=True)
        #self.mainframe.grid(row=0, column=0, sticky='NSEW')
        self.mainframe.winfo_toplevel().title('DSSolver')
        self.topmenu = None
        self.canvas = None  # Changed later on
        self.rcm = None
        self.listbox_results = None
        self.bcm = None
        self.bm = None
        self.lm = None
        self.rsm = None
        self.click_x = None
        self.click_y = None
        self.r1 = None
        self.r2 = None  # For self.start_or_end_beam()
        self.closest_node_label = None
        # [problem coords] = [[T]] @ [canvas coords]
        self.tx = 0; self.ty = 0  # Translation
        self.prev_x = None; self.prev_y = None
        self.zoom = 1
        self.ratio = 1

        self.linewidth = 2.0
        self.scale = 50
        self.node_radius = 2.5

        self.settings = {'shear': tk.BooleanVar(),
                         'moment': tk.BooleanVar(),
                         'highlight':tk.IntVar()}

        for var in ('shear', 'moment'):
            self.settings[var].set(False)

        for var in self.settings.keys():
            self.settings[var].trace_add('write', self.draw_canvas)

        self.plugins: Dict[type, DSSPlugin] = {}
        if 'plugins' in kwargs:
            plugins: Iterable[DSSPlugin] = kwargs['plugins']
            for plugin in plugins:
                plugin.load_plugin(self)

        self.menus = {}

        self.build_grid()
        self.build_menu()
        self.build_banner()
        self.build_rsmenu()
        self.build_canvas()
        self.build_bc_menu()  # Outsourced

        if not self.problem.nodes:
            self.problem.get_or_create_node((0,0), draw=True)
        self.draw_canvas()

        for plugin in self.plugins.values():
            plugin.on_after_dss_built()

    # Building functions
    def build_grid(self):
        self.mainframe.columnconfigure(0, weight=1)  #
        self.mainframe.columnconfigure(1, weight=0)  # Quick menu
        self.mainframe.rowconfigure(0, weight=0)  # 'DSSolver' banner
        self.mainframe.rowconfigure(1, weight=1)  # Canvas (resizable)
        self.mainframe.rowconfigure(2, weight=0)  # Output console?

    def build_banner(self):
        self.banner = tk.Label(self.mainframe, bg='white', text='DSSolver')
        self.banner.grid(row=0, column=0)

    def build_menu(self):
        self.topmenu = tk.Menu(self.root)
        topmenu = self.topmenu
        self.root.config(menu=topmenu)

        menu_file = tk.Menu(topmenu)
        self.menus['File'] = menu_file
        topmenu.add_cascade(label='File', menu=menu_file)
        menu_file.add_command(label='Open', command=lambda: self.open_problem())
        menu_file.add_command(label='Save as', command=lambda: self.save_problem())
        menu_file.add_separator()
        menu_file.add_command(label='New problem ', command=lambda: self.new_problem())

        menu_edit = tk.Menu(topmenu)
        self.menus['Edit'] = menu_edit
        topmenu.add_cascade(label='Edit', menu=menu_edit)
        menu_edit.add_command(label='Create element(s)',
                              command=lambda: BeamInputMenu(self, self.root, self.problem))
        menu_edit.add_command(label='Auto rotation lock',
                              command=lambda: self.problem.auto_rotation_lock())
        menu_edit.add_separator()
        menu_edit.add_command(label='Redraw canvas',
                              command=lambda: self.draw_canvas())

        menu_solve = tk.Menu(topmenu)
        self.menus['Solve'] = menu_solve
        topmenu.add_cascade(label='Solve', menu=menu_solve)

        def callback_factory(this, *args):
            return lambda : this.call_and_add_to_results(*args)

        menu_plugins = tk.Menu(topmenu)
        self.topmenu.add_cascade(label='Plugins', menu=menu_plugins)
        self.menus['Plugins'] = menu_plugins
        for cls, instance in self.plugins.items():
            menu = menu_plugins if not issubclass(cls, solvers.Solver) else menu_solve
            for key,value in instance.get_functions().items():
                menu.add_command(label = key,
                                 command = callback_factory(self, value))

        menu_results = tk.Menu(topmenu)
        self.topmenu.add_cascade(label='Results', menu=menu_results)
        self.menus['Results'] = menu_results

        topmenu.add_separator()

        topmenu.add_command(label='Autoscale', command=lambda: self.autoscale() )
        topmenu.add_command(label='Help', command=lambda: HelpBox(self.root))

        #topmenu.add_command(label='Func', command=lambda: self.upd_rsmenu())

    def call_and_add_to_results(self, func:Callable):
        results = func()
        #menu_results = self.menus['Results']
        #menu_results.add_command(label=results.__class__.__name__,
        #                 command = lambda : ResultsViewer(self.root, results, icon=self.icon))
        self.listbox_results.add(results)

    def build_bc_menu(self):
        """
        Coordinates on these labels are updated when the left mouse button
        is clicked on the canvas. (See self.rightclickmenu )
        """
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
        self.bcm.add_command(label='Rotation lock node at {}'.format(self.closest_node_label),
                             command=lambda: self.boundary_condition('locked'))
        self.bcm.add_command(label='Glider node at {}'.format(self.closest_node_label),
                             command=lambda: self.boundary_condition('glider'))

        self.bm = tk.Menu(self.rcm, tearoff=0)  # Beam menu
        self.rcm.add_cascade(label='Start/end element', menu=self.bm)
        self.bm.add_command(label='Start element at {}'.format((None, None)),
                             command=lambda: self.start_or_end_beam
                             (r=(self.canvas.transformation_matrix@[self.click_x, self.click_y,1])[0:2]))
        self.bm.add_command(label='Start element at closest',
                            command=lambda: self.start_or_end_beam
                             (r=self.closest_node_label))

        self.rcm.add_command(label='Query node at {}'.format(self.closest_node_label),
                             command=lambda: self.query_node(self.closest_node_label))

    def build_canvas(self):
        self.canvas = DSSCanvas(self.mainframe, bg='white', highlightthickness=0)
        self.canvas.dss = self # TODO: Remove
        self.canvas.grid(row=1, column=0, sticky='nsew')

        self.canvas.bind('<Button-1>', self._printcoords)
        self.canvas.bind('<Button-3>', self.rightclickmenu)

    def build_rsmenu(self):
        color1 = 'gray74'
        color2 = 'gray82'
        self.rsm = tk.Frame(self.mainframe, bg=color1, width=256)
        self.rsm.grid(row=1, column=1, sticky='nsew')

        self.listbox_results = DSSListbox(self.rsm)
        self.listbox_results.grid(row=1, column=0)
        self.listbox_results.bind('<Double-Button-1>', self.view_results)

        rsm_settings = tk.Frame(self.rsm, bg=color2, width=255)
        rsm_settings.grid(row=3, column=0, sticky='nw')
        rsm_shm_label = tk.Label(rsm_settings, text='Show/hide', bg=color2)
        rsm_shm_label.grid(row=0, column=0, columnspan=2, sticky='ew')

        buttons = ['Shear \n diagram', 'Moment \n diagram']

        vars = [self.settings['shear'], self.settings['moment']]

        i = 0
        for b,v in zip(buttons, vars):
            button = tk.Checkbutton(rsm_settings, text=b, variable=v, bg=color2,
                                    highlightthickness=0, justify=tk.LEFT)
            button.grid(row=int(i/2+1), column=i%2, sticky='wns')
            i += 1

        plugin_settings_frame = DSSSettingsFrame(rsm_settings)
        plugin_settings_frame.grid(row=3, columnspan=2, sticky='ew')
        for cls in self.plugins.keys():
            plugin_settings_frame.add_settings(cls)

    def rightclickmenu(self, event):
        self.click_x = event.x  # Canvas coordinates
        self.click_y = event.y
        self._closest_node((event.x, event.y))

        self.bcm.entryconfigure(0, label='Fix node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(1, label='Pin node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(2, label='Roller node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(3, label='Rotation lock node at {}'.format(self.closest_node_label))
        self.bcm.entryconfigure(4, label='Glider node at {}'.format(self.closest_node_label))

        self.bm.entryconfigure(0, label='Start element at {}'.format((self.canvas.transformation_matrix@[event.x, event.y, 1])[0:2]))
        self.bm.entryconfigure(1, label='Start element at closest: {}'.format(self.closest_node_label))

        self.lm.entryconfigure(0, label='Apply point load at {}'.format(self.closest_node_label))
        self.lm.entryconfigure(1, label='Apply distributed load from {}'.format(self.closest_node_label))

        self.rcm.entryconfigure(3, label='Query node at {}'.format(self.closest_node_label))

        if self.r1 is not None and np.any(self.r2) is None:  # End distr loads or elements
            self.bm.entryconfigure(0, label='End element at {}'.format((self.canvas.transformation_matrix@[event.x, event.y, 1])[0:2]))
            self.bm.entryconfigure(1, label='End element at closest: {}'.format(self.closest_node_label))
            self.lm.entryconfigure(1, label='End distributed load at {}'.format(self.closest_node_label))

        self.rcm.grab_release()
        self.rcm.tk_popup(event.x_root, event.y_root, 0)

    def view_results(self, *args):
        result = self.listbox_results.get_selected()
        ResultsViewer(self.root, result, icon=self.icon)

    # Drawing functions
    def draw_canvas(self, *args, **kwargs):
        self.canvas.delete('all')  # Clear the canvas

        self.draw_nodes()
        self.draw_elements()

        if self.settings['highlight'].get() and self.rsm_view_elements:
            self.draw_highlight(element_id =self.settings['highlight'].get() - 1)
        elif self.settings['highlight'].get() and not self.rsm_view_elements:
            self.draw_highlight(node_id =self.settings['highlight'].get() - 1)

        if self.problem.solved:
            if self.settings['shear'].get():
                self.draw_shear_diagram()

            if self.settings['moment'].get():
                self.draw_moment_diagram()

        self.draw_csys()

    def draw_elements(self):
        for element in self.problem.elements:
            element.draw_on_canvas(self.canvas)

    def draw_nodes(self):
        for node in self.problem.nodes:
            node.draw_on_canvas(self.canvas)

    def draw_shear_diagram(self):
        max_shear = np.max(np.abs(np.array([self.problem.forces[:, 1], self.problem.forces[:, 4]])))
        scale = 100/max_shear
        for beam in self.problem.elements:
            s,c = np.sin(beam.angle), np.cos(beam.angle)
            R = np.array([[c, -s], [s, c]])

            v1 = (beam.transform(beam.angle)@beam.forces)[1]
            v2 = -(beam.transform(beam.angle)@beam.forces)[4]

            p1 = (inv(self.canvas.transformation_matrix) @ np.hstack((beam.r1, 1)))[0:2] + R.T @ np.array([0, -scale])*v1
            p2 = (inv(self.canvas.transformation_matrix) @ np.hstack((beam.r2, 1)))[0:2] + R.T @ np.array([0, -scale])*v2
            self.canvas.create_line(*p1, *p2,
                                    tag='sheardiagram')
            self.canvas.create_text(*p1, text='{}'.format(np.round(v1,2)), anchor='sw')

    def draw_moment_diagram(self):
        max_moment = np.max(np.abs(np.array([self.problem.forces[:,2], self.problem.forces[:,5]])))
        scale = 100/max_moment
        for beam in self.problem.elements:
            s,c = np.sin(beam.angle), np.cos(beam.angle)
            R = np.array([[c, -s], [s, c]])

            v1 = beam.forces[2]
            v2 = -beam.forces[5]

            p1 = (inv(self.canvas.transformation_matrix) @ np.hstack((beam.r1, 1)))[0:2] + R.T @ np.array([0, -scale])*v1
            p2 = (inv(self.canvas.transformation_matrix) @ np.hstack((beam.r2, 1)))[0:2] + R.T @ np.array([0, -scale])*v2
            self.canvas.create_line(*p1, *p2,
                                    tag='momentdiagram')
            self.canvas.create_text(*p1, text='{}'.format(np.round(v1, 2)), anchor='sw')

    def draw_csys(self):
        self.canvas.create_line(10, self.canvas.height-10,
                                110, self.canvas.height-10,
                                arrow='last')
        self.canvas.create_text(110, self.canvas.height-10, text='x', anchor='sw')
        self.canvas.create_line(10, self.canvas.height-10,
                                10, self.canvas.height-110,
                                arrow='last')
        self.canvas.create_text(10, self.canvas.height-110, text='y', anchor='sw')

    def draw_highlight(self, node_id=None, element_id=None):
        """
        :param node_id: node.number
        :param element_id: element.number
        :return:  """

        self.canvas.delete('highlight')

        if element_id is not None:
            element = self.problem.elements[element_id]
            beam_r1 = (inv(self.canvas.transformation_matrix)@np.hstack((element.r1, 1)))[0:2]
            beam_r2 = (inv(self.canvas.transformation_matrix)@np.hstack((element.r2, 1)))[0:2]

            self.canvas.create_text(*beam_r1, text='r1', anchor='sw')
            self.canvas.create_text(*beam_r2, text='r2', anchor='sw')

            if type(element) == Beam:
                self.canvas.create_line(*np.hstack((beam_r1, beam_r2)),
                                        width=self.linewidth, tag='highlight', fill='red')

            elif type(element) == Rod:
                self.canvas.create_line(*np.hstack((beam_r1, beam_r2)),
                                        width=self.linewidth/4, tag='highlight', fill='red')

        elif node_id is not None:
            node = self.problem.nodes[node_id]
            node_r = (inv(self.canvas.transformation_matrix) @ np.hstack((node.r, 1)))[0:2]
            node_radius = self.node_radius * 1.5
            self.canvas.create_oval(*np.hstack((node_r - node_radius, node_r + node_radius)),
                                    fill='red', tag='mech')

    def query_node(self, node):
        self.problem.nodes[self.problem.node_at(node)].draw = not self.problem.nodes[self.problem.node_at(node)].draw
        self.draw_canvas()
        pass

    # Scaling and moving functions
    def autoscale(self):
        # TODO: Replace with DSSCanvas.autoscale when objects are added to the canvas
        x_ = (np.max(self.problem.nodal_coordinates[:, 0]) + np.min(self.problem.nodal_coordinates[:, 0]))/2
        y_ = (np.max(self.problem.nodal_coordinates[:, 1]) + np.min(self.problem.nodal_coordinates[:, 1]))/2

        w = self.canvas.width
        h = self.canvas.height

        a = self.problem.model_size()*0.7

        r0 = np.array([x_, y_, 1]) - np.array([a/2, a/2, 0])  # Problem bounding square SW corner
        r0_ = np.array([w/5, 4*h/5, 1])  # Canvas subsquare SW corner

        r = np.array([x_, y_, 1])  # Problem midpoint
        r_ = np.array([w/5, 4*h/5, 1]) + np.array([h/3, -h/3, 0])  # Canvas subsquare midpoint

        r1 = np.array([x_, y_, 1]) + np.array([0, a/2, 0])  # Problem bounding square N centerpoint
        r1_ = r_ + np.array([0, -h/3, 0])  # Canvas subsquare N centerpoint

        R = np.array([r0, r, r1]).T
        R_ = np.array([r0_, r_, r1_]).T
        # R and R_ must be invertible, or T will not be invertible

        self.canvas.transformation_matrix = R@inv(R_)
        self.draw_canvas()

    def move_to(self, xy=(50, -150)):
        # Moves the problem csys origin to canvas csys (xy)
        x,y = xy
        self.canvas.transformation_matrix[0:3,2] = np.array([self.canvas.transformation_matrix[0,0]*(x-1),
                                                      self.canvas.transformation_matrix[1,1]*(y-1),
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
        event_r_ = (self.canvas.transformation_matrix @ np.hstack((event_r, 1)))[0:2]  # Problem coordinate
        event_to_node = event_r_ - self.problem.nodal_coordinates
        event_to_node_norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=event_to_node)
        node_id = np.where(event_to_node_norm == np.min(event_to_node_norm))[0][0]

        self.closest_node_label = np.array(self.problem.nodes[node_id].r)  # Problem coordinates

    def boundary_condition(self, bc):
        """
        :param bc: 'fixed', 'fix', 'pinned', 'locked', 'pin', 'roller', 'lock'
        """
        if bc == 'fix':
            self.problem.fix(self.problem.node_at(self.closest_node_label))
        elif bc == 'pin':
            self.problem.pin(self.problem.node_at(self.closest_node_label))
        elif bc == 'roller':
            self.problem.roller(self.problem.node_at(self.closest_node_label))
        elif bc == 'locked':
            self.problem.lock(self.problem.node_at(self.closest_node_label))
        elif bc == 'glider':
            self.problem.glider(self.problem.node_at(self.closest_node_label))
        else:
            pass
        self.draw_canvas()

    def reset_prev(self, event):
        self.prev_x = None
        self.prev_y = None

    def new_problem(self):
        self.problem = Problem()
        self.draw_canvas()

    def _printcoords(self, event):
        self._closest_node((event.x, event.y))
        print("Clicked at canvas", [event.x, event.y], 'problem', (self.canvas.transformation_matrix@[event.x,event.y,1])[0:2])
        print('Closest node', self.closest_node_label)
        print('r1, r2', self.r1, self.r2)

    # Open and save
    def save_problem(self):
        filename = tk.filedialog.asksaveasfilename()
        with open(filename, 'wb') as file:
            pickle.dump(self.problem, file, pickle.HIGHEST_PROTOCOL)

    def open_problem(self):
        self.new_problem()
        filename = tk.filedialog.askopenfilename()

        with open(filename, 'rb') as file:
            self.problem = pickle.load(file)

        self.autoscale()

class DSSInputMenu:
    def __init__(self, window, root, problem, *args, **kwargs):
        self.top = tk.Toplevel(root)
        self.top.iconbitmap(window.icon)

        self.window = window
        self.root = root
        self.problem = problem

class LoadInputMenu(DSSInputMenu):
    def __init__(self, window, root, problem):
        """
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        super().__init__(window, root, problem)
        self.top.winfo_toplevel().title('Apply load')
        self.label = tk.Label(self.top, text='Apply load at node')
        self.label.grid(row=0, column=0)

        entrykeys = ['Fx', 'Fy', 'M']
        entries = [self.e_fx, self.e_fy, self.e_m] = [tk.Entry(self.top) for _ in range(3)]

        for idx, entry in enumerate(entries):
            entry.grid(row=idx+1, column=1)

        for idx, key in enumerate(entrykeys):
            tk.Label(self.top, text=key).grid(row=idx+1)

        for idx, entry in enumerate((self.e_fx, self.e_fy, self.e_m)):
            try:
                entry.insert(0, self.problem.node_at(self.window.closest_node_label).loads[idx])
            except:
                entry.insert(0, 0)

        self.e_fx.focus_set()

        self.b = tk.Button(self.top, text='Apply load', command=self.cleanup)
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

class DistrLoadInputMenu(DSSInputMenu):
    def __init__(self, window, root, problem, r1 = np.array([0,0]), r2 = np.array([0,0])):
        """
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        super().__init__(window, root, problem)
        self.top.winfo_toplevel().title('Apply distributed load')

        self.label = tk.Label(self.top, text='Apply distributed load')
        self.label.grid(row=0, column=0)

        entrykeys = ['Starting node', 'Ending node', 'Load magnitude']
        entries = [self.e_r1, self.e_r2, self.e_load] = [tk.Entry(self.top) for _ in range(3)]

        for idx, entry in enumerate(entries):
            entry.grid(row=idx+1, column=1)

        for idx, key in enumerate(entrykeys):
            tk.Label(self.top, text=key).grid(row=idx+1)

        self.e_r1.insert(0, '{},{}'.format(r1[0], r1[1]))
        self.e_r2.insert(0, '{},{}'.format(r2[0], r2[1]))
        self.e_load.insert(0, '0')

        self.e_r2.focus_set()

        self.b = tk.Button(self.top, text='Apply load', command=self.cleanup)
        self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))
        self.b.grid(row=4, column=0)

    def cleanup(self):
        r1 = np.array(eval(self.e_r1.get()))
        r2 = np.array(eval(self.e_r2.get()))
        load = float(eval(self.e_load.get()))
        self.problem.load_members_distr(r1=r1, r2=r2, load=load)

        self.window.draw_canvas()
        self.top.destroy()

class BeamInputMenu(DSSInputMenu):
    def __init__(self, window, root, problem, def_r1=np.array([0,0]), def_r2=np.array([1000,0])):
        """
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        super().__init__(window, root, problem)

        self.top.winfo_toplevel().title('Create element(s)')
        self.top.columnconfigure(1, weight=1)

        self.label = tk.Label(self.top, text='Create element(s) between:')
        self.label.grid(row=0, column=0)

        self.nodeframe = tk.Frame(self.top, width=300)
        self.nodeframe.grid(row=1, column=0, columnspan=3, sticky='ew')
        self.nodeframe.grid_columnconfigure(0, weight=1)

        self.morenodes = tk.Button(self.top,
                                   text='+1 \n node',
                                   command=lambda: self.more_nodes())
        self.morenodes.grid(row=0, column=2)

        self.n = 0
        self.more_nodes()
        self.more_nodes()

        entrykeys = ['Area', 'Elastic modulus', '2nd mmt of area',
                     'Half section height', 'No. of elements']
        entries = [self.e_A, self.e_E, self.e_I, self.e_z, self.e_n] = \
            [tk.Entry(self.top) for _ in range(5)]

        for idx, entry in enumerate(entries):
            entry.grid(row=idx+2, column=1, columnspan=2)

        for idx, key in enumerate(entrykeys):
            tk.Label(self.top, text=key).grid(row=idx+2)

        defaults = (1e5, 2e5, 1e5, 4)
        for value, entry in zip(defaults, (self.e_A, self.e_E, self.e_I, self.e_n)):
            entry.insert(0,int(value))

        self.e_r1.insert(0, '{},{}'.format(def_r1[0], def_r1[1]))
        self.e_r2.insert(0, '{},{}'.format(def_r2[0], def_r2[1]))
        self.e_z.insert(0, '158')
        self.e_r1.focus_set()

        self.secmgr = tk.Button(self.top,
                                text='Section \n manager',
                                command=lambda: SectionManager(bip=self,
                                                               window=self.window,
                                                               root=self.root))
        self.secmgr.grid(row=0, column=1, columnspan=1, sticky='e')
        #self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))



        self.b = tk.Button(self.top, text='Create element(s)', command=self.cleanup)
        self.b.grid(row=9, column=0)
        self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))

        # Adding 'beam' or 'rod' radio buttons
        self.k = tk.IntVar()

        b1 = tk.Radiobutton(self.top, text='Beam', variable=self.k, value=1,
                            command=lambda: self.check_choice())#.grid(row=8, column=1)
        b1.grid(row=9, column=1)
        b2 = tk.Radiobutton(self.top, text='Rod', variable=self.k, value=2,
                            command=lambda: self.check_choice())#.grid(row=8, column=2)
        b2.grid(row=9, column=2)
        self.k.set(1)
        self.check_choice()

    def check_choice(self):
        if self.k.get() == 2:
            self.e_I.config(state=tk.DISABLED)
            self.e_z.config(state=tk.DISABLED)
            self.e_n.delete(0, tk.END)
            self.e_n.insert(0,'1')
            self.e_n.config(state=tk.DISABLED)

        elif self.k.get() == 1:
            self.e_I.config(state=tk.NORMAL)
            self.e_z.config(state=tk.NORMAL)
            self.e_n.config(state=tk.NORMAL)

    def more_nodes(self):
        self.__setattr__('e_r{}'.format(self.n+1), tk.Entry(self.nodeframe))
        self.__getattribute__('e_r{}'.format(self.n+1)).grid(row=self.n, column=1, columnspan=2, sticky='e')
        #.grid(row=self.n, column=1, columnspan=2))
        tk.Label(self.nodeframe, text='Node {}'.format(self.n+1)).grid(row=self.n, column=0, sticky='ew')
        self.n += 1

    def cleanup(self):
        A = float(self.e_A.get())
        E = float(self.e_E.get())
        I = float(self.e_I.get())
        z = float(self.e_z.get())
        n = float(self.e_n.get())

        if self.k.get() == 1: # Beam
            func = self.problem.create_beams

        elif self.k.get() == 2: # Rod
            func = self.problem.create_rod

        for k in range(self.n - 1):
            ri = self.__getattribute__('e_r{}'.format(k+1)).get()
            ri = np.array(ri.split(','), dtype=float)
            rj = self.__getattribute__('e_r{}'.format(k+2)).get()
            rj = np.array(rj.split(','), dtype=float)
            print('Element', ri, rj)
            func(ri, rj, A=A, E=E, I=I, z=z, n=n)


        self.window.autoscale()
        self.top.destroy()

class SectionManager(DSSInputMenu):
    def __init__(self, bip, window, root):
        """
        :param bip: The BeamInputMenu window (passed as 'self' from class BeamInputMenu)
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        super().__init__(window, root, problem=None)
        self.top.winfo_toplevel().title('Section manager')
        self.bip = bip

        self.A = None
        self.I = None

        self.sec = tk.StringVar()
        self.sec.set('Rectangular')
        self.sec.trace('w', self.change_dropdown)
        self.sections = ['Rectangular', 'Circular', 'I-beam']
        self.dropdown = tk.OptionMenu(self.top, self.sec, *self.sections)
        self.dropdown.grid(row=0, column=1, rowspan=2)

        entrykeys = ['DIM1', 'DIM2', 'DIM3', 'DIM4']
        entries = [self.dim1, self.dim2, self.dim3, self.dim4] = [tk.Entry(self.top) for _ in range(4)]
        for idx, entry in enumerate(entries):
            entry.grid(row=idx+2, column=1)

        for idx, key in enumerate(entrykeys):
            tk.Label(self.top, text=key).grid(row=idx+2)

        self.dim3.insert(0, 0)
        self.dim4.insert(0, 0)

        self.valuelabel = tk.Label(self.top, text='Placeholder, area: A, I:I')
        self.valuelabel.grid(row=6, column=0, columnspan=2)

        self.b = tk.Button(self.top, text='Ok', command=self.cleanup)
        self.b.grid(row=7, column=0, columnspan=2)
        self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))

        file = os.getcwd() + '/gfx/dim_Rectangular.gif'
        self.photo = tk.PhotoImage(file=file)
        self.photolabel = tk.Label(self.top, image=self.photo)
        self.photolabel.photo = self.photo
        self.photolabel.grid(row=1, column=2, rowspan=7)

    def change_dropdown(self, *args):
        file = os.getcwd() + '/gfx/dim_{}.gif'.format(self.sec.get())
        self.photo = tk.PhotoImage(file=file)
        self.photolabel.configure(image=self.photo)

        if self.sec.get() == 'Circular':
            self.dim2.insert(0, 0)
            self.dim3.config(state=tk.DISABLED)
            self.dim4.config(state=tk.DISABLED)
        else:
            self.dim3.config(state=tk.NORMAL)
            self.dim4.config(state=tk.NORMAL)

    def cleanup(self, *args):
        dim1 = eval(self.dim1.get())
        dim2 = eval(self.dim2.get())
        dim3 = eval(self.dim3.get())
        dim4 = eval(self.dim4.get())
        if self.sec.get() == 'Rectangular':
            self.I = 1/12 * (dim1 * dim2**3 - dim3 * dim4**3)
            self.A = (dim1*dim2) - (dim3*dim4)
            self.z = (dim2/2)
        elif self.sec.get() == 'Circular':
            self.I = np.pi/4 * ((dim1/2)**4 - (dim2/2)**4)
            self.A = np.pi * (dim1**2 - dim2**2)
            self.z = dim1/2
        elif self.sec.get() == 'I-beam':
            self.I = 1/12 * ((dim1 * dim3**3)
                            -((dim1-dim4) * (dim3-2*dim2)**3))
            self.A = (dim1*dim3) - (dim3-2*dim2)*(dim1-dim4)
            self.z = dim3/2

        self.bip.e_A.delete(0, tk.END)
        self.bip.e_I.delete(0, tk.END)
        self.bip.e_z.delete(0, tk.END)
        self.bip.e_A.insert(0, str(self.A))
        self.bip.e_I.insert(0, str(self.I))
        self.bip.e_z.insert(0, str(self.z))

        self.top.destroy()

class BeamManager:
    def __init__(self, window, root, element_id):
        """
        :param bip: The BeamInputMenu window (passed as 'self' from class BeamInputMenu)
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        top = self.top = tk.Toplevel(root)
        self.top.winfo_toplevel().title('Properties: Element no. {}'.format(element_id))
        self.top.iconbitmap(window.icon)

        self.window = window
        self.root = root

        entrykeys = ['Area', 'Elastic modulus', '2nd mmt of area']
        entries = [self.e_A, self.e_E, self.e_I] = \
            [tk.Entry(top) for _ in range(3)]

        self.e_A.insert(0, self.problem.elements[element_id].A)

class HelpBox:
    def __init__(self, root):
        top = self.top = tk.Toplevel(root)
        self.top.iconbitmap(icon)
        self.root = root

        self.textbox = tk.Text(top)
        self.top.columnconfigure(0, weight=1)
        self.top.columnconfigure(1, weight=0)
        self.top.rowconfigure(0, weight=1)

        self.textbox.grid(row=0, column=0, sticky='nsew')

        self.scr = tk.Scrollbar(top)
        self.scr.grid(row=0, column=1, sticky='ns')

        questions = ['Q{}: Why is this table of contents not clickable?\n',
                     'Q{}: Key bindings\n',
                     'Q{}: What is the difference between Beam and Rod elements?\n',
                     'Q{}: Can I do truss analysis using DSSolver?\n',
                     'Q{}: Rod elements make my analysis fail\n',
                     'Q{}: What units does DSSolver use?\n' ,
                     'Q{}: How many Beam elements should I use?\n',
                     'Q{}: Data view descriptions and sign conventions\n']

        answers = ['A{}: It will be someday, probably.\n',
                   """A{}: Double click mouse-1 to zoom in. Double click mousewheel to zoom out. 
Drag and drop to move the view.\n""",
                   """A{}: A Beam element is a classical beam which can withstand normal, shearing and bending 
forces. A Rod element is only capable of withstanding normal forces. See Q/A's below for more info on Rod elements.\n""",
                   """A{}: Yes - just use Rod elements. Due to how Rod elements work, you will have to 
use the lock rotation boundary condition on all nodes that only connect Rod elements. 
Use the "Auto rotation lock" (in the Edit menu) to automatically rotation lock all nodes 
that only connect Rod elements.  
You should also fix (not pin/roller) the truss where it is supported. See 'Rod 
elements make my analysis fail' for more info.\n""",
                   """A{}: Just like Beam elements, Rod elements have 3 degrees of freedom pr node (two 
translations and one rotation) to ensure compatibility with beam elements. However, Rod 
elements only have stiffness against axial deformations. All nodes in a system must have 
stiffness against all deformations (x, y, rotation) or the system stiffness matrix will be 
singular. Therefore, at both endnodes of a Rod element, there must either be (1) a Beam 
element, which provides stiffness against all deformations, (2) a fixed support, which 
removes all degrees of freedom at that node, or (3) a differently oriented Rod element 
and a rotation lock support, which in sum will have stiffness against both translations 
and remove the rotation degree of freedom.\n""",
                   """A{}: DSSolver is unitless, meaning you can use any units you want as long as they are 
internally consistent. All outputs will be in the same units. Some systems of consistent 
units are SI (meters, newtons, newtonmeters, pascals), SI (mm) (millimeters, newtons, 
newtonmillimeters, megapascals) and imperial (inches) (inches, pounds, pound-inches, 
pounds pr square inch). More systems can be found on Google.\n""",
                   """A{}: The number of beam elements only affects the drawing of displaced shapes and shear 
force- and moment diagrams, not the accuracy of the analysis. Displaced shapes and shear force- 
and moment diagrams are drawn with one straight line segment pr element. The number of elements 
also affect the solution time, but you are probably not analyzing systems of such size that 
solution time is a concern with this software.\n""",
                   """A{}: /n -All vectors written on the graphics view refer to the global system coordinate system. /n
- The distributed load sign 'convention' is a little arbitrary, but the drawn direction drawn is what it is. /n
- Displacements and forces for nodes in the right hand menu are given in the global csys. /n
- Displacements, forces and stresses for elements in the right hand menu are given in the local (element) csys, that 
is (axial, transversal, rotational)./n
- Stresses for elements are given as (max axial, min axial, avg shear)./n
- Moment diagrams seem to always be drawn on the tensile side, but this is purely an accident and may not always be the 
case.
 
"""]

        for idx, Q in enumerate(questions):
            self.textbox.insert(tk.INSERT, Q.format(idx).replace('\n', ''))
            self.textbox.insert(tk.INSERT, '\n')

        self.textbox.insert(tk.INSERT, '\n \n')

        for idx, (Q,A) in enumerate(zip(questions,answers)):
            self.textbox.insert(tk.INSERT, Q.format(idx).replace('\n', ''))
            self.textbox.insert(tk.INSERT, '\n')
            self.textbox.insert(tk.INSERT, A.format(idx).replace('\n', '').replace('/n','\n'))
            self.textbox.insert(tk.INSERT, '\n\n')

if __name__ == '__main__':
    #self.icon = 'dss_icon.ico' if _platform == 'win32' or _platform == 'win64' else '@dss_icon.xbm'
    if sys.platform == 'win32' or sys.platform == 'win64':
        icon = os.getcwd() + '/gfx/dss_icon.ico'
    else:
        icon = '@' + os.getcwd() + 'dss_icon.xbm'

    # Load plugin_types
    plugin_list = []
    for module in (plugins, elements, solvers, report_plugin):
        plugin_classes = (cls for cls in module.__dict__.values() if
                          isinstance(cls, type) and issubclass(cls, plugins.DSSPlugin))
        for cls in plugin_classes:
            plugin_list.append(cls)
            #cls.load_plugin(dss)
            # instance = cls.create_instance(dss)
            # instance.load_plugin()





    p = Problem()
    root = tk.Tk()
    dss = DSS(root, problem=p, plugins=plugin_list)
    dss.autoscale()

    root.mainloop()

