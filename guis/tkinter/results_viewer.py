import tkinter as tk
from typing import Callable
from guis.tkinter.extras import DSSCanvas, DSSSettingsFrame, DSSListbox
from core.results import Results

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
        classes = set()
        for item in self.results.get_objects():
            self.canvas.add_object(item)
            classes.add(item.__class__)

        i = 3
        for name, func in results.get_actions().items():
            button = tk.Button(right_frame, text=name, command=self.on_click_factory(func))
            button.grid(row=i)
            i += 1

        label = tk.Label(right_frame, textvariable=self.stringvar)
        label.grid(row=i)

        self.canvas.autoscale()
        results.on_after_resultview_built(self)

    def animate_func(self):
        animator = ResultAnimator(self.results, self.canvas)
        animator.add_hook(lambda i: self.stringvar.set(f'Current displacement set: {i}'))
        animator.start()
    def _iterate_results(self):
        interval = int(1000 * 2 / self.results.num_displ_sets)
        self.current_displ_set = 0
        for step in range(self.results.num_displ_sets):
            self.results.set_displacements()
            self.results.current_displ_set += 1
            yield interval
        yield False

    def on_click_factory(self, func):
        def return_func():
            func()
            self.canvas.redraw()
            self.stringvar.set(f'Current displacement set: {self.results.current_displ_set}')
        return return_func


class ResultAnimator:
    def __init__(self, results, canvas):
        self.results = results
        self.canvas = canvas
        self._delay = int(2000 / self.results.num_displ_sets)
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
        if self.results.increment() == 0:
            return
        self.canvas.redraw()
        if self._running:
            for hook in self._hooks:
                hook(self.results.current_displ_set)
            self.canvas.after(self._delay, self._run_animation)
