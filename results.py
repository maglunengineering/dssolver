from typing import Dict,Iterable,TypeVar,Sized
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from elements import FiniteElement2Node, Node, DSSModelObject
from extras import DSSCanvas, DSSSettingsFrame, DSSListbox


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

        self.current_result_vector = 0
        self.num_result_vectors = 1

    def get_objects(self) -> Iterable[DSSModelObject]:
        yield from self.nodes
        yield from self.elements

    def get_buttons(self):
        return {}

    def get_text(self):
        return f'Current results vector: {self.current_result_vector}'

    def increment(self):
        if self.current_result_vector < self.num_result_vectors - 1:
            self.current_result_vector += 1
        else:
            self.current_result_vector = 0

        self.set_displacements()

    def decrement(self):
        if self.current_result_vector > 0:
            self.current_result_vector -= 1
        else:
            self.current_result_vector = self.num_result_vectors - 1

        self.set_displacements()

    def animate(self):
        pass

    def set_displacements(self):
        pass

    def reset_animation(self):
        pass

    def on_after_resultview_built(self, view):
        pass


class ResultsStaticLinear(Results):
    def __init__(self, problem, displacements:np.ndarray):
        super().__init__(problem)

        self.displacements = displacements
        for node in self.nodes:
            node.displacements = displacements[node.dofs]


class ResultsStaticNonlinear(Results):
    def __init__(self, problem, displacements:Sized, forces:Sized):
        super().__init__(problem)

        self.forces = forces
        self.displacements = displacements
        self.num_result_vectors = len(displacements)

    def set_displacements(self):
        displacements = self.displacements[self.current_result_vector]
        for node in self.nodes:
            node.displacements = displacements[node.dofs]

    def animate(self):
        interval = int(1000 * 2 / self.num_result_vectors)
        self.current_result_vector = 0
        for step in range(self.num_result_vectors):
            self.set_displacements()
            self.current_result_vector += 1
            yield interval
        yield False

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


class ResultsModal(Results):
    def __init__(self, problem, eigenvalues, eigenvectors):
        super().__init__(problem)

        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.num_result_vectors = len(eigenvalues)
        self.current_result_vector = 0
        self.scale = problem.model_size() / 5

    def set_displacements(self):
        eigenvector = self.eigenvectors[self.current_result_vector] * self.scale
        eigenvalue = self.eigenvalues[self.current_result_vector]
        for node in self.nodes:
            node.displacements = eigenvector[node.dofs]

    def on_after_resultview_built(self, view):
        for eigenvector in self.eigenvectors:
            view.listbox.add(eigenvector)
        self.set_displacements()

    def animate(self):
        # 50 steps
        oldscale = self.scale
        for sine in np.sin(np.linspace(-np.pi, np.pi, 51)):
            self.scale = sine * oldscale
            self.set_displacements()
            yield 20
        self.scale = oldscale
        self.set_displacements()
        yield False

    def get_text(self):
        return f'Eigenvalue no. {self.current_result_vector}: ' \
               f'{np.round(self.eigenvalues[self.current_result_vector], 2)}'

    def get_buttons(self):
        return {'Increment': self.increment,
                'Decrement': self.decrement}


class ResultsDynamicTimeIntegration(ResultsStaticNonlinear):
    def __init__(self, problem, displacements):
        super().__init__(problem, displacements, np.zeros_like(displacements))

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

        self.listbox = DSSListbox(right_frame)
        self.listbox.grid(row=0)

        settings_frame = DSSSettingsFrame(right_frame)
        settings_frame.grid(row=1)


        btn_animate = tk.Button(right_frame, text='Animate',
                                command = self.animate_func)
        btn_animate.grid(row=2)

        self.stringvar = tk.StringVar()

        self.results = results
        self.stringvar.set(results.get_text())
        classes = set()
        for item in self.results.get_objects():
            self.canvas.add_object(item)
            classes.add(item.__class__)

        for cls in classes:
            settings_frame.add_settings(cls)

        i = 3
        for name, func in results.get_buttons().items():
            button = tk.Button(right_frame, text=name, command=self.on_click_factory(func))
            button.grid(row=i)
            i += 1

        label = tk.Label(right_frame, textvariable=self.stringvar)
        label.grid(row=i)

        self.canvas.autoscale()
        results.on_after_resultview_built(self)

    def animate_func(self, *args):
        """
        Works with a generator formulation of the animate() function. animate() must yield a frame delay (in ms)
        while the animation runs, then yield False.
        """
        if args:
            iterator = args[0]
        else:
            iterator = self.results.animate()

        delay = next(iterator)
        if delay:
            self.canvas.redraw()
            self.canvas.update()
            self.stringvar.set(self.results.get_text())
            self.top.update()
            self.canvas.after(delay, self.animate_func, iterator)
        else:
            self.results.reset_animation()
            self.canvas.redraw()

    def on_click_factory(self, func):
        def return_func():
            func()
            self.canvas.redraw()
            self.stringvar.set(self.results.get_text())
        return return_func
