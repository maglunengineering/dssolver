import tkinter as tk
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from guis.tkinter.extras import DSSCanvas, DSSSettingsFrame, DSSListbox
from core.results import Results
from core import settings

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

        #settings_frame = DSSSettingsFrame(right_frame)
        #settings_frame.grid(row=1)

        btn_animate = tk.Button(right_frame, text='Animate', command = self.animate_func)
        btn_animate.grid(row=2)

        self.stringvar = tk.StringVar()

        self.results = results
        self._cur_lc = 0
        self._cur_hist = 0
        for item in self.results.get_objects():
            self.canvas.add_object(item)

        i = 3
        for func in (self.quickplot,):
            button = tk.Button(right_frame, text=func.__name__.capitalize(), command=self.on_click_factory(func))
            button.grid(row=i)
            i += 1            

        label = tk.Label(right_frame, textvariable=self.stringvar)
        label.grid(row=i)

        self.canvas.autoscale()
        results.on_after_resultview_built(self)

    def animate_func(self):
        animator = ResultAnimator(self.results, self.canvas, self._cur_lc)
        animator.add_hook(lambda i: self.stringvar.set(f'Current displacement set: {i}'))
        animator.start()

    def on_click_factory(self, func):
        def return_func():
            func()
            self.canvas.redraw()
            self.stringvar.set(f'Current displacement set: {self._cur_hist}')
        return return_func
    
    
    def quickplot(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig,ax = plt.subplots()

        for node in self.results.nodes:
            if node.loads.any():
                break
        
        dof = node.dofs[np.abs(node.loads).argmax()]
        displ_history = self.results.get_displacement_slice[self._cur_lc, :, dof]
        sign = np.sign(np.average(displ_history))
        load_history = self.results.get_force_slice[self._cur_lc, :]

        plt.ylabel('Control parameter')
        plt.xlabel('Displacement')
        plt.title(f'Displacement vs control parameter at dof {dof}')
        plt.plot(sign * displ_history, load_history)
        plt.show()



class ResultAnimator:
    def __init__(self, results:Results, canvas, i_lc):
        self.results = results
        self.canvas = canvas

        _, nhist, _ = results.get_size()
        self._i_lc = i_lc
        self._i_hist = 0
        self._n_hist = nhist[i_lc]

        self._delay = int(2000 / self._n_hist)
        self._running = False
        self._hooks = []

    def start(self):
        self._running = True
        self.canvas.after(0, self._run_animation)

    def stop(self):
        self._running = False

    def add_hook(self, hook:Callable[[int],None]):
        self._hooks.append(hook)

    def _run_animation(self):
        if self._i_hist == self._n_hist:
            self.stop()
            return
        
        self.canvas.redraw(displacements=self.results.get_displacement_slice[self._i_lc, self._i_hist, :])
        self._i_hist += 1
        self.canvas.update()
        if self._running:
            for hook in self._hooks:
                hook(self._i_hist)
            self.canvas.after(self._delay, self._run_animation)
