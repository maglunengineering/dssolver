import tkinter as tk
from tkinter import messagebox
from elements import *
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
        self.click_x = None
        self.click_y = None
        self.r1 = None
        self.r2 = None  # For self.start_or_end_beam()
        self.closest_node_label = None


        self.build_grid()
        self.build_menu()
        self.build_banner()
        self.build_canvas()
        self.build_rightclickmenu()
        self.build_bc_menu()  # Outsourced

        self.displaced_plot = False



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
        #menu_edit.add_command(label='Create node',
        #                      command=lambda: NodeInputMenu(self.root, self.problem))
        menu_edit.add_command(label='Create beams',
                              command=lambda: BeamInputMenu(self, self.root, self.problem))
        menu_edit.add_command(label='Redraw',
                              command=lambda: self.draw_canvas())

        topmenu.add_command(label='Redraw',
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


        show_menu = tk.Menu(topmenu)
        topmenu.add_cascade(label='Show/hide', menu=show_menu)

        show_menu.add_command(label='Displaced shape',
                            command=lambda: self.draw_displaced())

    def build_rightclickmenu(self):
        self.rcm = tk.Menu(root, tearoff=0)

    def build_bc_menu(self):
        self.bcm = tk.Menu(self.rcm, tearoff=0)

        self.rcm.add_cascade(label='Apply boundary condition', menu=self.bcm)

        self.bcm.add_command(label='Pin node at {}'.format(self.closest_node_label))
        self.bcm.add_command(label='Fix node at {}'.format(self.closest_node_label))
        self.bcm.add_command(label='Roller node at {}'.format(self.closest_node_label))
        self.bcm.add_command(label='Apply load at {}'.format(self.closest_node_label))
        '''
        All these labels are updated, and commands are assigned, when the left mouse button 
        is clicked on the canvas. (See self.rightclickmenu )
        '''

        self.bm = tk.Menu(self.rcm, tearoff=0)
        self.rcm.add_cascade(label='Start/end beam', menu=self.bm)
        self.bm.add_cascade(label='Start beam at {}'.format((0, 0)),
                             command=lambda: self.start_or_end_beam(r=(self.click_x, self.click_y)))
        self.bm.add_cascade(label='Start beam at closest', #{}'.format(self.closest_node_label))
                            command=lambda: self.start_or_end_beam(r=self.closest_node_label))


    def build_canvas(self):
        self.canvas = ResizingCanvas(self.mainframe, bg='white', highlightthickness=0)
        #canvas.pack(fill='both', expand=True)
        self.canvas.grid(sticky='nsew')
        self.canvas.create_rectangle(5, 5, self.canvas.width-10, self.canvas.height-10,
                                     tag='bbox') # Canvas bounding box
        self.canvas.grid(row=1)

        self.canvas.bind('<Button-1>', self._printcoords)
        self.canvas.bind('<Button-3>', self.rightclickmenu)


    def rightclickmenu(self, event):
        self.click_x = event.x
        self.click_y = event.y
        self.bm.entryconfigure(0, label='Start beam at {}'.format([event.x, event.y]))
        self.bm.entryconfigure(1, label='Start beam at closest')#'{}'.format(self.closest_node_label))
        if np.any(self.r1) and not np.any(self.r2):
            self.bm.entryconfigure(0, label='End beam at {}'.format([event.x, event.y]))
            self.bm.entryconfigure(1, label='End beam at closest')#{}'.format(self.closest_node_label))

        try:

            self._closest_node((event.x, event.y))
            self.bcm.entryconfigure(0, label='Fix node at {}'.format(self.closest_node_label),
                                    command=lambda: self.boundary_condition('fix'))

            self.bcm.entryconfigure(1, label='Pin node at {}'.format(self.closest_node_label),
                                    command=lambda: self.boundary_condition('pin'))

            self.bcm.entryconfigure(2, label='Roller node at {}'.format(self.closest_node_label),
                                    command=lambda: self.boundary_condition('roller'))

            self.bcm.entryconfigure(3, label='Apply load at {}'.format(self.closest_node_label),
                                    command=lambda: LoadInputMenu(self, self.root, self.problem))

        except:
            pass

        finally:
            self.rcm.grab_release()
            self.rcm.tk_popup(event.x_root, event.y_root, 0)

    def _closest_node(self, xy):
        event_r = np.array(xy)
        event_to_node = event_r - self.problem.nodal_coordinates
        event_to_node_norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=event_to_node)
        node_id = np.where(event_to_node_norm == np.min(event_to_node_norm))[0][0] - 1

        self.closest_node_label = self.problem.nodes[node_id].r

    def start_or_end_beam(self, r):
        if not np.any(self.r1):  # If r1 does not exist
            self.r1 = np.array(r)
            self.bm.entryconfigure(0, label='End beam at {}'.format((self.click_x, self.click_y)))
        elif np.any(self.r1) and not np.any(self.r2):  # If r1 does exist and r2 does not exist
            self.r2 = np.array(r)
            BeamInputMenu(self, self.root, self.problem,
                          def_r1 = '{},{}'.format(self.r1[0], self.r1[1]),
                          def_r2 = '{},{}'.format(self.r2[0], self.r2[1]))
            self.r1 = self.r2 = None



    def draw_canvas(self):
        self.canvas.delete('all')  # Clear the canvas
        self.canvas.create_rectangle(5, 5, self.canvas.width - 10, self.canvas.height - 10,
                                     tag='bbox')  # Canvas bounding box

        #_node_radius = node_radius * np.array([-1, -1, 1, 1])
        linewidth = 3.0

        for node in self.problem.nodes:
            if node.draw:
                node_radius = 0
                self.canvas.create_oval(*np.hstack((node.r - node_radius, node.r + node_radius)),
                                                   fill='black', tag='mech')
            if np.any(np.round(node.loads[0:2])):  # If lump force, draw force arrow
                scale = 50
                arrow_start = node.r - node.loads[0:2] / np.linalg.norm(node.loads[0:2]) * scale * np.array([1,1])
                arrow_end = node.r
                self.canvas.create_line(*arrow_start, *arrow_end,
                                        arrow='last', fill='blue', tag='mech')
                self.canvas.create_text(*arrow_start,
                                        text='{}'.format(node.loads[0:2]),
                                        anchor='s', tag='mech')

            if np.alen(node.loads) >= 3 and node.loads[2] != 0:  # If lump moment, draw moment arrow
                scale = 50 * np.sign(node.loads[2])

                arc_start = node.r + [0, scale/2]
                arc_mid = node.r + [scale/2, 0]
                arc_end = node.r + [0, -scale/2]

                self.canvas.create_line(*arc_start, *arc_mid, *arc_end,
                                        smooth = True,
                                        arrow='first', fill='blue', tag='mech')
                self.canvas.create_text(*arc_start,
                                        text='{}'.format(node.loads[2]),
                                        anchor='s', tag='mech')


            if node.boundary_condition:
                self.draw_boundary_condition(node.boundary_condition,
                                             position=node.r, draw_angle=node.beams[0].angle)
                pass
                #print('Image')
                #self.canvas.image = tk.PhotoImage(file='bc_{}.gif'.format(node.boundary_condition))
                #self.canvas.create_image(*node.r, image=self.canvas.image)



        for beam in self.problem.beams:
            if isinstance(beam, Beam):
                self.canvas.create_line(*np.hstack((beam.r1, beam.r2)),
                                        width=linewidth, tag = 'mech')
            elif isinstance(beam, Rod):
                self.canvas.create_line(*np.hstack((beam.r1, beam.r2)),
                                        width=linewidth/2, tag='mech')

    def draw_displaced(self):

        node_radius = 2.5
        for node in self.problem.nodes:
            #node.displacements *= [1, -1, 1]
            if node.draw:
                self.canvas.create_oval(*np.hstack((node.r - node_radius + node.displacements[0:2],
                                                    node.r + node_radius + node.displacements[0:2])),
                                        fill='red', tag='mech_disp')

                if np.any(np.round(node.loads[0:2])):  # If node is loaded
                    scale = 50
                    arrow_start = node.r - node.loads[0:2]/np.linalg.norm(node.loads[0:2])*scale*[1, 1]
                    arrow_end = node.r
                    self.canvas.create_line(*arrow_start + node.displacements[0:2],
                                            *arrow_end + node.displacements[0:2],
                                            arrow='last', fill='blue', tag='mech_disp')

                self.canvas.create_text(*node.r + node.displacements[0:2],
                                        text='{}'.format(np.round(node.displacements, 1)),
                                        anchor='sw', tag='mech_disp')

        for beam in self.problem.beams:
            self.canvas.create_line(*np.hstack((beam.r1 + beam.nodes[0].displacements[0:2],
                                                beam.r2 + beam.nodes[1].displacements[0:2])),
                                        fill='red', tag='mech_disp', dash=(1,))

        if hasattr(self, 'displaced_plot'):
            if self.displaced_plot:
                self.canvas.delete('mech_disp')
        self.displaced_plot = not self.displaced_plot

    def draw_boundary_condition(self, bc_type, position, draw_angle=0):
        """
        :param bc_type: 'fixed', 'pinned', 'roller', 'fix' or 'pin'
        """
        scale = 20
        linewidth = 2
        if bc_type == 'fixed':
            r0 = position  # np.array
            angle = draw_angle
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]])

            self.canvas.create_line(*(r0 + rotation@[0, scale]), *(r0 + rotation@[0, -scale]),
                                                          width=linewidth, fill='black', tag='bc')
            for offset in np.linspace(0, 2*scale, 6):
                self.canvas.create_line(*((r0 + rotation@[0, -scale+offset])),
                                        *((r0 + rotation@[0, -scale+offset] + rotation@[-scale/2, scale/2])),
                                        width=linewidth, fill='black', tag='bc')

        elif bc_type == 'pinned' or bc_type == 'roller':
            r0 = position  # np.array
            k = 1.5  # constant

            self.canvas.create_oval(*(r0 - scale/4), *(r0 + scale/5))
            self.canvas.create_line(*r0, *(r0 + np.array([-np.sin(np.deg2rad(30)),
                                                          np.cos(np.deg2rad(30))]) * k*scale),
                                    width=linewidth, fill='black', tag='bc')
            self.canvas.create_line(*r0, *(r0 + np.array([np.sin(np.deg2rad(30)),
                                                          np.cos(np.deg2rad(30))])*k*scale),
                                    width=linewidth, fill='black', tag='bc')
            self.canvas.create_line(*(r0 + np.array([-np.sin(np.deg2rad(30)),
                                                          np.cos(np.deg2rad(30))]) * k*scale),
                                    *(r0 + np.array([np.sin(np.deg2rad(30)),
                                                     np.cos(np.deg2rad(30))])*k*scale),
                                    width=linewidth, fill='black', tag='bc')
            if bc_type == 'roller':
                self.canvas.create_line(*(r0 + np.array([-np.sin(np.deg2rad(30)),
                                                         np.cos(np.deg2rad(30))])*k*scale
                                                        + np.array([-scale/5, scale/5])),
                                        *(r0 + np.array([np.sin(np.deg2rad(30)),
                                                         np.cos(np.deg2rad(30))])*k*scale)
                                                        + np.array([scale/5, scale/5]),
                                        width=linewidth, fill='black', tag='bc')

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

    def new_problem(self):
        self.__delattr__('problem')
        self.problem = Problem()
        self.draw_canvas()

    def _printcoords(self, event):
        print("Clicked at", event.x, event.y)

    def _selftest(self, loadcase = 1):
        if loadcase == 1:  # Cantilever beam, point load
            self.problem.create_beams((20,120), (400,120))
            self.problem.fix(self.problem.node_at((20,120)))
            #self.problem.load_node(self.problem.node_at((400,120)), (0, 10000))

            self.draw_canvas()

        if loadcase == 2:  # Simply supported beam, no load
            self.problem.create_beams((20, 120), (400, 120))
            self.problem.pin(self.problem.node_at((20, 120)))
            self.problem.roller(self.problem.node_at((400, 120)))

            self.draw_canvas()

        if loadcase == 3:  # L beam(s), point load
            self.problem.create_beams((20, 120), (400, 120))
            self.problem.create_beams((400, 120), (400, 440))
            self.problem.fix(self.problem.node_at((20,120)))
            #self.problem.load_node(self.problem.node_at((400,440)), (10000,0,0))

            self.draw_canvas()




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
                # Doesn't work -- fix. If node is loaded, default value in dialog should not be zero
            except:
                entry.insert(0, 0)


        self.b = tk.Button(top, text='Apply load', command=self.cleanup)
        self.b.grid(row=4, column=0)

    def cleanup(self):
        self.problem.load_node( self.problem.node_at(self.window.closest_node_label),
                                load = (eval(self.e_fx.get()),
                                        eval(self.e_fy.get()),
                                        eval(self.e_m.get()))
                                )

        self.window.draw_canvas()

        self.top.destroy()


class BeamInputMenu:
    def __init__(self, window, root, problem, def_r1='20,20', def_r2='200,200'):
        """
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        """

        top = self.top = tk.Toplevel(root)
        self.top.winfo_toplevel().title('Create beam(s)')
        self.window = window
        self.root = root
        self.problem = problem

        self.label = tk.Label(top, text='Create beams between:')
        self.label.grid(row=0, column=0)


        entrykeys = ['Node 1', 'Node 2', 'Area', 'Elastic modulus', '2nd mmt of area', 'No. of elements']
        entries = [self.e_r1, self.e_r2, self.e_A, self.e_E, self.e_I, self.e_n] = \
            [tk.Entry(top) for _ in range(6)]

        for idx, entry in enumerate(entries):
            entry.grid(row=idx+1, column=1)

        for idx, key in enumerate(entrykeys):
            tk.Label(top, text=key).grid(row=idx+1)

        defaults = (1e5, 2e5, 1e5, 10)
        for value, entry in zip(defaults, (self.e_A, self.e_E, self.e_I, self.e_n)):
            entry.insert(0,int(value))

        self.e_r1.insert(0, def_r1)
        self.e_r2.insert(0, def_r2)
        self.e_r1.focus_set()


        self.b = tk.Button(top, text='Create beams', command=self.cleanup)
        self.b.grid(row=8, column=0)
        self.top.bind('<Return>', (lambda e, b=self.b: self.b.invoke()))

        """ # Adding 'beam' or 'rod' radio buttons
        elements = [('Beam', '1'), ('Rod', '2')]
        v = tk.StringVar(); v.set('1')
        for text, element in elements:
            radiobutton = tk.Radiobutton(top, text=text, value=element, variable=v)
        radiobutton.grid(row=8, column=1)
        """

    def cleanup(self):
        #try:
        r1 = np.array(eval(self.e_r1.get()))
        r2 = np.array(eval(self.e_r2.get()))
        A = eval(self.e_A.get())
        E = eval(self.e_E.get())
        I = eval(self.e_I.get())
        n = eval(self.e_n.get())

        self.problem.create_beams(r1, r2, A=A, E=E, I=I, n=n)
        self.window.draw_canvas()

        self.top.destroy()
        

        #except:
        #    msg = 'Node coordinates in wrong format or inputs missing?'
        #    tk.messagebox.showinfo('Error', msg)

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
        print(value)
        self.problem.create_node(value)
        self.top.destroy()
        return value


if __name__ == '__main__':
    p = Problem()

    root = tk.Tk()
    w = Window(root, problem=p)

    root.mainloop()