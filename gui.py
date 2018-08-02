import tkinter as tk
from tkinter import messagebox
from elements import *
from extras import ResizingCanvas

class Window:
    def __init__(self, root, problem=None, *args, **kwargs):
        self.root = root
        self.problem = problem
        self.mainframe = tk.Frame(self.root, bg='white')
        self.mainframe.pack(fill=tk.BOTH, expand=True)

        self.build_grid()
        self.build_menu()
        self.build_banner()
        self.build_canvas()

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
        menu_file.add_command(label='New case')# , command=)


        menu_edit = tk.Menu(topmenu)
        topmenu.add_cascade(label='Edit', menu=menu_edit)
        #menu_edit.add_command(label='Create node',
        #                      command=lambda: NodeInputMenu(self.root, self.problem))
        menu_edit.add_command(label='Create beams',
                              command=lambda: BeamInputMenu(self, self.root, self.problem))
        menu_edit.add_command(label='Redraw',
                              command=lambda: self._redraw_canvas())

        topmenu.add_command(label='Redraw',
                            command=lambda: self._redraw_canvas())
        topmenu.add_command(label='Self test',
                            command=lambda: self._selftest())

    def build_canvas(self):
        self.canvas = ResizingCanvas(self.mainframe, bg='white', highlightthickness=0)
        #canvas.pack(fill='both', expand=True)
        self.canvas.grid(sticky='nsew')
        self.canvas.create_rectangle(5, 5, self.canvas.width-10, self.canvas.height-10,
                                     tag='bbox') # Canvas bounding box
        self.canvas.grid(row=1)

        self.canvas.bind('<Button-1>', self._callback)

    def _redraw_canvas(self):
        self.canvas.delete('mech')  # Clear the canvas
        node_radius = 2.5
        #_node_radius = node_radius * np.array([-1, -1, 1, 1])

        for node in self.problem.nodes:
            self.canvas.create_oval(*np.hstack((node.r - node_radius, node.r + node_radius)),
                                               fill='black', tag='mech')

        for beam in self.problem.beams:
            self.canvas.create_line(*np.hstack((beam.r1, beam.r2)),
                                    tag='mech')


    def _callback(self, event):
        print("Clicked at", event.x, event.y)

    def _selftest(self):
        self._redraw_canvas()
        self.problem.create_beams((20, 120), (200, 120))
        self._redraw_canvas()
        tk.messagebox.showinfo('Messagebox', 'Halfway')
        self.problem.create_beams((200, 120), (200, 240))
        self._redraw_canvas()



class BeamInputMenu:
    def __init__(self, window, root, problem):
        '''
        :param window: The DSSolver main window (passed as 'self' from class Window)
        :param root: root is the root = tkinter.Tk() (passed as 'self.root')
        :param problem: Instance of the Problem class (passed as 'self.problem')
        '''

        top = self.top = tk.Toplevel(root)
        self.window = window
        self.root = root
        self.problem = problem

        self.label = tk.Label(top, text='Create beams between:')
        self.label.grid(row=0, column=0)


        entrykeys = ['Node 1', 'Node 2', 'Area', 'Elastic modulus', '2nd mmt of area', 'No. of beams']
        entries = [self.e_r1, self.e_r2, self.e_A, self.e_E, self.e_I, self.e_n] = \
            [tk.Entry(top) for _ in range(6)]

        for idx, entry in enumerate(entries):
            entry.grid(row=idx+1, column=1)

        for idx, key in enumerate(entrykeys):
            tk.Label(top, text=key).grid(row=idx+1)

        defaults = (1e5, 2e5, 1e5, 10)
        for value, entry in zip(defaults, (self.e_A, self.e_E, self.e_I, self.e_n)):
            entry.insert(0,int(value))

        self.e_r1.insert(0, '20,100')
        self.e_r2.insert(0, '200,100')  # Default nodes for easy debugging

        self.b = tk.Button(top, text='Create beams', command=self.cleanup)
        self.b.grid(row=8, column=0)

    def cleanup(self):
        #try:
        r1 = np.array(eval(self.e_r1.get()))
        r2 = np.array(eval(self.e_r2.get()))
        A = eval(self.e_A.get())
        E = eval(self.e_E.get())
        I = eval(self.e_I.get())
        n = eval(self.e_n.get())

        self.problem.create_beams(r1, r2, A=A, E=E, I=I, n=n)

        '''if self.problem.create_beams(r1, r2, A=A, E=E, I=I, n=n):  # Beam is new
            node_radius = 2.5

            self.window.canvas.create_oval(*np.hstack((r1-node_radius, r1+node_radius)),
                                           fill='black')
            self.window.canvas.create_oval(*np.hstack((r2-node_radius, r2+node_radius)),
                                           fill='black')
            self.window.canvas.create_line(*np.hstack((r1,r2)))
            
            ## Code for drawing beam on creation: replaced by Window._redraw_canvas()
        '''
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