import sys
import os
import pickle
import importlib
import tkinter as tk
from typing import Callable, Iterable, Dict

import numpy as np

from core import problem, elements, settings
from guis.tkinter import extras, plugin_base
from plugins import solvers
import tools

from results_viewer import ResultsViewer

np.set_printoptions(precision=2, suppress=True)
inv = np.linalg.inv

class DSSGUI:
    def __init__(self, root:tk.Tk, problem=None, *args, **kwargs):
        self.root = root
        self.root.minsize(width=1024, height=640)
        self.icon = icon if icon else None
        self.root.iconbitmap(self.icon)
        self.problem = problem

        self.mainframe = tk.Frame(self.root, bg='white')
        self.mainframe.pack(fill=tk.BOTH, expand=True)
        self.mainframe.winfo_toplevel().title('DSSolver')
        self.topmenu = None
        self.rsm_settings = None
        self.sel_obj_settings:extras.DSSSettingsFrame = None
        self.selected_object = None
        self.listbox_results = None
        self.rsm = None
        self.canvas:extras.DSSCanvas = extras.DSSCanvas(self.mainframe, bg='white', highlightthickness=0)
        self.canvas.dss = self # TODO: Remove
        self.canvas.grid(row=0, column=0, sticky='nsew')

        settings.add_setting('dssgui.running_animation', True)

        self.menus = {}

        self.build_grid()
        self.build_menu(kwargs.get('plugins', {}))
        self.build_rsmenu()
        #self.build_bc_menu()  # Outsourced

        self.canvas.set_tool(tools.ToolSelect(self, self.canvas, root))

        if not self.problem.nodes:
            self.canvas.add_object(self.problem.get_or_create_node((0,0)))
        self.draw_canvas()

        self.plugins: Dict[type, plugin_base.DSSPlugin] = {}
        plugins = kwargs.get('plugins', {})
        for plugin in plugins:
            instance = plugin(self)
            instance.load_plugin()

    # Building functions
    def build_grid(self):
        self.mainframe.columnconfigure(0, weight=1)  #
        self.mainframe.columnconfigure(1, weight=1)  # Quick menu
        self.mainframe.rowconfigure(0, weight=1)  # Canvas (resizable)

    def build_menu(self, plugins):
        self.topmenu = tk.Menu(self.root)
        topmenu = self.topmenu
        self.root.config(menu=topmenu)

        self.add_topmenu_item('File', 'Open', self.open_problem)
        self.add_topmenu_item('File', 'Save as', self.save_problem)
        self.menus['File'].add_separator()
        self.add_topmenu_item('File', 'New problem', self.new_problem)

        self.add_topmenu_item('Edit', 'Create element(s)', lambda: BeamInputMenu(self, self.problem))
        self.add_topmenu_item('Edit', 'Auto rotation lock', self.problem.auto_rotation_lock)
        self.menus['Edit'].add_separator()
        self.add_topmenu_item('Edit', 'Redraw canvas', self.draw_canvas)

        menu_solve = tk.Menu(topmenu)
        self.menus['Solve'] = menu_solve
        topmenu.add_cascade(label='Solve', menu=menu_solve)

        def callback_factory(this, *args):
            return lambda : this.call_and_add_to_results(*args)

        menu_plugins = tk.Menu(topmenu)
        #for cls, instance in plugins.items():
        for plugin in plugins:
            instance = plugin(self)
            if isinstance(instance, solvers.Solver) and type(instance) != solvers.Solver:
                self.add_topmenu_item('Solve', plugin.__name__, callback_factory(self, instance.solve))

        topmenu.add_separator()
        topmenu.add_command(label='Autoscale', command=lambda: self.autoscale() )

    def add_topmenu_item(self, menu_title:str, cmd_title:str, cmd:Callable):
        if menu_title not in self.menus:
            self.menus[menu_title] = tk.Menu(self.topmenu)
            self.topmenu.add_cascade(label=menu_title, menu=self.menus[menu_title])

        if cmd_title and cmd:
            menu = self.menus[menu_title]
            menu.add_command(label=cmd_title, command=cmd)


    def call_and_add_to_results(self, func:Callable):
        for x in func():
            if x:
                results = x
                self.listbox_results.add(results)
                if settings.get_setting('dssgui.running_animation', True):
                    self.draw_canvas()
                break
            self.draw_canvas()
            if settings.get_setting('dssgui.running_animation', True):
                self.canvas.update()


    def build_rsmenu(self):
        color1 = 'gray74'
        color2 = 'gray82'
        self.rsm = tk.Frame(self.mainframe, bg=color1, width=400)
        self.rsm.grid(row=0, column=1, sticky='nsew')

        self.listbox_results = extras.DSSListbox(self.rsm)
        self.listbox_results.grid(row=1, column=0)
        self.listbox_results.bind('<Double-Button-1>', self.view_results)

        self.rsm_settings = tk.Frame(self.rsm, bg=color2, width=400)
        self.rsm_settings.grid(row=3, column=0, sticky='nw')
        rsm_shm_label = tk.Label(self.rsm_settings, text='Settings', bg=color2)
        rsm_shm_label.grid(row=0, column=1, columnspan=3, sticky='ew')

        rsm_shm_label2 = tk.Label(self.rsm_settings, text='Selected object', bg=color2)
        rsm_shm_label2.grid(row=4, column=1, columnspan=3, sticky='ew')

        self.set_settings_default()

    def set_settings_default(self):
        if self.sel_obj_settings:
            self.sel_obj_settings.destroy()
        plugin_settings_frame = extras.DSSSettingsFrame.from_settings(self.rsm_settings, 'dssgui')
        plugin_settings_frame.grid(row=3, columnspan=2, sticky='ew')

    def set_settings(self, obj):
        if self.sel_obj_settings:
            self.sel_obj_settings.destroy()
        self.sel_obj_settings = extras.DSSSettingsFrame.from_object(self.rsm_settings, obj)
        self.sel_obj_settings.grid(row=5, columnspan=2, sticky='nsew')
        self.root.update()

    def set_selection(self, obj):
        self.set_settings(obj)
        self.canvas.set_selection(obj)
        self.draw_canvas()

    def view_results(self, *args):
        result = self.listbox_results.get_selected()
        ResultsViewer(self.root, result, icon=self.icon)

    # Drawing functions
    def draw_canvas(self, *args, **kwargs):
        self.canvas.delete('all')  # Clear the canvas

        self.canvas.redraw()
        self.draw_csys()

    def update_canvas(self):
        for obj in self.problem.nodes:
            self.canvas.add_object(obj)
        for obj in self.problem.elements:
            self.canvas.add_object(obj)

    def draw_csys(self):
        self.canvas.create_line(10, self.canvas.height-10,
                                110, self.canvas.height-10,
                                arrow='last')
        self.canvas.create_text(110, self.canvas.height-10, text='x', anchor='sw')
        self.canvas.create_line(10, self.canvas.height-10,
                                10, self.canvas.height-110,
                                arrow='last')
        self.canvas.create_text(10, self.canvas.height-110, text='y', anchor='sw')

    # Scaling and moving functions
    def autoscale(self):
        self.update_canvas()
        self.canvas.autoscale()

    def move_to(self, xy=(50, -150)):
        # Moves the problem csys origin to canvas csys (xy)
        x,y = xy
        self.canvas.transformation_matrix[0:3,2] = np.array([self.canvas.transformation_matrix[0, 0] * (x - 1),
                                                                           self.canvas.transformation_matrix[1,1] * (y-1),
                                                                           1])
        self.draw_canvas()

    def new_problem(self):
        self.problem = problem.Problem()
        self.canvas.clear()
        self.draw_canvas()

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

    def node_from_pt(self, pt):
        for node in self.problem.nodes:
            if np.allclose(np.r, pt):
                return node
        return None

    def create_beam_dlg(self, r1, r2):
        return BeamInputMenu(self, self.problem, r1, r2)

    def create_load_dlg(self, node):
        LoadInputMenu(self, self.problem, node)

class DSSInputMenu:
    def __init__(self, window, problem, *args, **kwargs):
        self.top = tk.Toplevel(root)
        self.top.iconbitmap(window.icon)

        self.window = window
        self.problem = problem

class LoadInputMenu(DSSInputMenu):
    def __init__(self, window, problem, node):
        """
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        super().__init__(window, problem)
        self.top.winfo_toplevel().title('Apply load')
        self.label = tk.Label(self.top, text='Apply load at node')
        self.label.grid(row=0, column=0)
        self.node = node

        entrykeys = ['Fx', 'Fy', 'M']
        entries = [self.e_fx, self.e_fy, self.e_m] = [tk.Entry(self.top) for _ in range(3)]

        for idx, entry in enumerate(entries):
            entry.grid(row=idx+1, column=1)

        for idx, key in enumerate(entrykeys):
            tk.Label(self.top, text=key).grid(row=idx + 1)

        for idx, entry in enumerate((self.e_fx, self.e_fy, self.e_m)):
            try:
                entry.insert(0, node.loads[idx])
            except:
                entry.insert(0, 0)

        self.e_fx.focus_set()

        self.b = tk.Button(self.top, text='Apply load', command=self.cleanup)
        self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))
        self.b.grid(row=4, column=0)

    def cleanup(self):
        self.node.loads = np.array([float(self.e_fx.get()), float(self.e_fy.get()), float(self.e_m.get())])

        self.window.update_canvas()
        self.window.draw_canvas()

        self.top.destroy()

class BeamInputMenu(DSSInputMenu):
    def __init__(self, window, problem, def_r1=np.array([0, 0]), def_r2=np.array([1000, 0])):
        """
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        super().__init__(window, problem)

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
            tk.Label(self.top, text=key).grid(row=idx + 2)

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
                                                               window=self.window
                                                                      ))
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
        tk.Label(self.nodeframe, text='Node {}'.format(self.n + 1)).grid(row=self.n, column=0, sticky='ew')
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
        self.window.update_canvas()


class SectionManager(DSSInputMenu):
    def __init__(self, bip, window):
        """
        :param bip: The BeamInputMenu window (passed as 'self' from class BeamInputMenu)
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """
        super().__init__(window, problem=None)
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
            tk.Label(self.top, text=key).grid(row=idx + 2)

        self.dim3.insert(0, 0)
        self.dim4.insert(0, 0)

        self.valuelabel = tk.Label(self.top, text='Placeholder, area: A, I:I')
        self.valuelabel.grid(row=6, column=0, columnspan=2)

        self.b = tk.Button(self.top, text='Ok', command=self.cleanup)
        self.b.grid(row=7, column=0, columnspan=2)
        self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))

        file = os.path.join(os.getcwd(), '..', 'gfx', 'dim_Rectangular.gif')
        self.photo = tk.PhotoImage(file=file)
        self.photolabel = tk.Label(self.top, image=self.photo)
        self.photolabel.photo = self.photo
        self.photolabel.grid(row=1, column=2, rowspan=7)

    def change_dropdown(self, *args):
        file = os.path.join(os.getcwd(), '..', 'gfx', f'dim_{self.sec.get()}.gif')
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
            self.I = np.pi / 4 * ((dim1 / 2) ** 4 - (dim2 / 2) ** 4)
            self.A = np.pi * (dim1 ** 2 - dim2 ** 2)
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

if __name__ == '__main__':
    #self.icon = 'dss_icon.ico' if _platform == 'win32' or _platform == 'win64' else '@dss_icon.xbm'
    ext = 'ico' if sys.platform.startswith('win') else '.xbm'
    icon = os.path.join(os.getcwd(), '..', 'gfx', f'dss_icon.{ext}')

    # Load plugin_types
    plugin_list = []
    modules = [elements, solvers]
    for module_name in os.listdir('plugins'):
        if module_name.endswith('.py'):
            module = importlib.import_module(f'plugins.{module_name[:-3]}')
            modules.append(module)

    for module in modules:
        plugin_classes = (cls for cls in module.__dict__.values() if
                          isinstance(cls, type) and issubclass(cls, plugin_base.DSSPlugin))
        for cls in plugin_classes:
            plugin_list.append(cls)

    p = problem.Problem()
    root = tk.Tk()
    dss = DSSGUI(root, problem=p, plugins=plugin_list)
    dss.autoscale()

    root.mainloop()

