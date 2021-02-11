import time
from typing import *
import numpy as np
import matplotlib.pyplot as plt
from elements import *
from extras import DSSCanvas, DSSSettingsFrame


T = TypeVar('T')

class ElementNodeMap:
    def __init__(self, elements:Iterable[FiniteElement2Node],
                 nodal_results:Dict[Node, np.ndarray]):
        self.elements = list(elements)
        self.nodal_results = nodal_results

    def __getitem__(self, element:FiniteElement2Node):
        at_node_1 = self.nodal_results[element.node1]
        at_node_2 = self.nodal_results[element.node2]
        return np.hstack((at_node_1, at_node_2))



class Results:
    def __init__(self, problem):
        self.problem = problem
        self.nodes = problem.nodes
        self.elements = problem.elements

    def get_objects(self) -> Iterable[DSSModelObject]:
        yield from self.nodes
        yield from self.elements

    def get_buttons(self):
        return {}

    def animate(self):
        pass

    def reset_animation(self):
        pass

class ResultsStaticLinear(Results):
    def __init__(self, problem, displacements:np.ndarray):
        super().__init__(problem)

        for node in self.nodes:
            node.displacements = displacements[node.dofs]

class ResultsStaticNonlinear(Results):
    def __init__(self, problem, displacements:Sized, forces:Sized):
        super().__init__(problem)

        self.forces = forces
        self.displacements = displacements
        self.num_steps = len(displacements)
        self.current_step = 0


    def set_displacements(self):
        displacements = self.displacements[self.current_step]
        for node in self.nodes:
            node.displacements = displacements[node.dofs]

    def reset_animation(self):
        self.current_step = 0

    def animate(self):
        if self.increment():
            #i = self.current_step
            #return int(self.step_lengths[i])
            return int(1000 * 2 / self.num_steps)
        else:
            return False

    def increment(self):
        if self.current_step < self.num_steps - 1:
            self.current_step += 1
        else:
            return False
        self.set_displacements()
        return True

    def decrement(self):
        if self.current_step > 0:
            self.current_step -= 1
        self.set_displacements()

    def quickplot(self):
        for node in self.nodes:
            if node.loads.any():
                break
        dof = node.dofs[np.abs(node.loads).argmax()]
        displ_history = self.displacements[:,dof]
        sign = np.sign(np.average(displ_history))
        load_history = self.forces

        plt.ylabel('Control parameter')
        plt.xlabel('Displacement')
        plt.title(f'Displacement vs control parameter at dof {dof}')
        plt.plot(sign * displ_history, load_history)
        plt.show()

    def get_buttons(self):
        return {'Increment' : self.increment,
                'Decrement' : self.decrement,
                'Quick plot' : self.quickplot}



class ResultsViewer:
    def __init__(self, root, results:Results, **kwargs):
        self.top = tk.Toplevel(root)
        if 'icon' in kwargs:
            self.top.iconbitmap(kwargs['icon'])
        self.top.winfo_toplevel().title('DSSolver Result Viewer')
        self.top.rowconfigure(0, weight=1)
        self.top.columnconfigure(0, weight=1)

        self.top.minsize(width=1024, height=640)
        self.top.maxsize(width=1024, height=640)

        self.canvas = DSSCanvas(self.top, bg='white')
        self.canvas.unbind_on_resize()
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)

        right_frame = tk.Frame(self.top)
        right_frame.grid(row=0, column=1)

        listbox = tk.Listbox(right_frame)
        listbox.grid(row=0)

        settings_frame = DSSSettingsFrame(right_frame)
        settings_frame.grid(row=1)

        def draw_func():
            delay = results.animate()
            if delay:
                self.canvas.redraw()
                self.canvas.update()
                self.top.update()
                self.canvas.after(delay, draw_func)
            else:
                results.reset_animation()


        btn_animate = tk.Button(right_frame, text='Animate',
                                command = draw_func)
        btn_animate.grid(row=2)

        i = 3
        for name,func in results.get_buttons().items():
            button = tk.Button(right_frame, text=name, command=func)
            button.bind('<Button-1>', lambda *args: self.canvas.redraw())
            button.grid(row=i)
            i += 1

        self.results = results
        classes = set()
        for item in self.results.get_objects():
            self.canvas.add_object(item)
            classes.add(item.__class__)

        for cls in classes:
            settings_frame.add_settings(cls)

        self.canvas.autoscale()

