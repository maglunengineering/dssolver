import tkinter as tk
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

        settings_frame = DSSSettingsFrame(right_frame)
        settings_frame.grid(row=1)

        btn_animate = tk.Button(right_frame, text='Animate', command = self.animate_func)
        btn_animate.grid(row=2)

        self.stringvar = tk.StringVar()

        self.results = results
        self.stringvar.set(f'Current displacement set: {self.results.current_displ_set}')
        classes = set()
        for item in self.results.get_objects():
            self.canvas.add_object(item)
            classes.add(item.__class__)

        for cls in classes:
            settings_frame.add_settings(cls)

        i = 3
        for name, func in results.get_actions().items():
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
            self.stringvar.set(f'Current displacement set: {self.results.current_displ_set}')
            self.top.update()
            self.canvas.after(delay, self.animate_func, iterator)
        else:
            self.results.reset_animation()
            self.canvas.redraw()

    def on_click_factory(self, func):
        def return_func():
            func()
            self.canvas.redraw()
            self.stringvar.set(f'Current displacement set: {self.results.current_displ_set}')
        return return_func
