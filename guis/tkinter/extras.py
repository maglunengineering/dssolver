import collections
import tkinter as tk
import numpy as np
from typing import Iterable, Tuple, Callable
from numpy.linalg import solve
import drawing
from core import settings

# a subclass of Canvas for dealing with resizing of windows
# credit: ebarr @ StackOverflow


class ResizingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = 512  # self.winfo_reqheight()
        self.width = 768  # self.winfo_reqwidth()

    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        #self.scale("all",0,0,wscale,hscale)



class DSSCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.height = 512
        self.width = 768

        self.tx = 0
        self.ty = 0
        self.prev_x = None
        self.prev_y = None

        # Transforms from canvas/screen coordinates to problem space coordinates
        self.transformation_matrix = np.array([[1, 0, -50], [0, -1, 100], [0, 0, 1]], dtype=float)

        self.dss = None
        if 'dss' in kwargs:
            self.dss = kwargs['dss']

        self.objects = []
        self.snap_objs = {}
        self.bind_on_resize()
        self.selected_object = None

    def bind_on_resize(self):
        self.bind("<Configure>", self.on_resize)

    def unbind_on_resize(self):
        self.unbind("<Configure>")

    def on_resize(self, event):
        self.width = event.width
        self.height = event.height

        self.config(width=self.width, height=self.height)

    def set_tool(self, tool):
        self.tool = tool
        self.tool.activate()

    def set_selection(self, obj):
        self.selected_object = obj

    def draw_node(self, pt, radius, *args, **kwargs):
        pt_canvas = self.problem_to_canvas(pt)
        super().create_oval(*np.hstack((pt_canvas - radius, pt_canvas + radius)), *args, **kwargs)

    def draw_point(self, pt, radius, *args, **kwargs):
        pt_canvas = self.problem_to_canvas(pt)
        super().create_oval(*np.hstack((pt_canvas - radius, pt_canvas + radius)), *args, **kwargs)

    def draw_oval(self, pt1, pt2, *args, **kwargs):
        pt1_canvas = self.problem_to_canvas(pt1)
        pt2_canvas = self.problem_to_canvas(pt2)
        super().create_oval(*np.hstack((pt1_canvas, pt2_canvas)), *args, **kwargs)

    def draw_line(self, pt1, pt2, *args, **kwargs):
        r1 = self.problem_to_canvas(pt1)
        r2 = self.problem_to_canvas(pt2)
        super().create_line(*np.hstack((r1, r2)), *args, **kwargs)

    def draw_arc(self, arc_start, arc_mid, arc_end, **kwargs):
        arc_start = self.problem_to_canvas(arc_start)
        arc_mid = self.problem_to_canvas(arc_mid)
        arc_end = self.problem_to_canvas(arc_end)
        super().create_line(*arc_start, *arc_mid, *arc_end, **kwargs)

    def draw_polygon(self, pts, *args, **kwargs):
        canvas_pts = []
        for pt in pts:
            canvas_pt = np.linalg.solve(self.transformation_matrix, np.array([*pt, 1]))[:2]
            canvas_pts.extend(canvas_pt)
        self.create_polygon(*canvas_pts, *args, **kwargs)

    def draw_text(self, pt, text, *args, **kwargs):
        canvas_pt = self.problem_to_canvas(pt)
        self.create_text(*canvas_pt, text=text, *args, **kwargs)

    def problem_to_canvas(self, pt):
        return np.linalg.solve(self.transformation_matrix, np.array([*pt, 1]))[0:2]

    def canvas_to_problem(self, pt):
        return (self.transformation_matrix @ np.array([*pt, 1]))[0:2]

    def redraw(self):
        self.delete('all')
        self.snap_objs.clear()
        for obj in self.objects:
            snap_pt = drawing.get_drawer(obj).draw_on_canvas(obj, self)
            self.snap_objs[obj] = snap_pt
        if self.selected_object and self.selected_object in self.snap_objs:
            pt = self.snap_objs[self.selected_object]
            scale = 4*np.abs(self.transformation_matrix[0,0] * self.transformation_matrix[1,1])
            self.draw_oval(pt - scale*np.ones(2), pt + scale*np.ones(2), outline='red')

    def move(self, event):
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = event.x
            self.prev_y = event.y

        self.transformation_matrix[0:,2] = self.transformation_matrix[:,2] + np.array([-self.tx, self.ty, 0])

        self.tx = (event.x - self.prev_x) * self.transformation_matrix[0,0]
        self.ty = (event.y - self.prev_y) * self.transformation_matrix[0,0]
        self.prev_x = event.x
        self.prev_y = event.y

        self.redraw()

        if self.dss:
            self.dss.draw_canvas()

    def on_lbuttonup(self, event):
        self.prev_x = None
        self.prev_y = None

    def autoscale(self):
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf

        nodes = (obj for obj in self.objects if obj.__class__.__name__ == 'Node') # Whatever
        for node in nodes:
            if node.r[0] < xmin:
                xmin = node.r[0]
            if node.r[0] > xmax:
                xmax = node.r[0]
            if node.r[1] < ymin:
                ymin = node.r[1]
            if node.r[1] > ymax:
                ymax = node.r[1]

        if np.isclose(xmin, xmax) and np.isclose(ymin, ymax):
            self.transformation_matrix = np.eye(3)
            self.redraw()
            return

        w = self.width
        h = self.height

        x_ = (xmin + xmax) / 2
        y_ = (ymin + ymax) / 2
        a = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)

        r0 = np.array([x_, y_, 1]) - np.array([a/2, a/2, 0])  # Problem bounding square SW corner
        r0_ = np.array([w/5, 4*h/5, 1])  # Canvas subsquare SW corner

        r = np.array([x_, y_, 1])  # Problem midpoint
        r_ = np.array([w/5, 4*h/5, 1]) + np.array([h/3, -h/3, 0])  # Canvas subsquare midpoint

        r1 = np.array([x_, y_, 1]) + np.array([0, a/2, 0])  # Problem bounding square N centerpoint
        r1_ = r_ + np.array([0, -h/3, 0])  # Canvas subsquare N centerpoint

        R = np.array([r0, r, r1]).T
        R_ = np.array([r0_, r_, r1_]).T
        # R and R_ must be invertible, or T will not be invertible

        self.transformation_matrix = R@np.linalg.inv(R_)
        self.redraw()

    def add_object(self, obj):
        self.objects.append(obj)
        snap_pt = drawing.get_drawer(obj).draw_on_canvas(obj, self)
        self.snap_objs[obj] = snap_pt

    def get_closest(self, pt_canvas):
        pt_model = self.canvas_to_problem(pt_canvas)
        closest_obj = min_by(self.snap_objs.items(), lambda kvp: np.linalg.norm(kvp[1] - pt_model))
        if closest_obj:
            return closest_obj[0]

    def scaleup(self, event):
        self.transformation_matrix[0:2, 0:2] = self.transformation_matrix[0:2, 0:2]*0.8
        self.redraw()

    def scaledown(self, event):
        self.transformation_matrix[0:2, 0:2] = self.transformation_matrix[0:2, 0:2]*1.2
        self.redraw()


class DSSListbox(tk.Listbox):
    def __init__(self, master=None, cnf=None, **kwargs):
        if cnf is None:
            cnf = {}
        super().__init__(master, cnf, **kwargs)

        self.string_map = dict()

    def add(self, obj):
        k = str(obj)
        self.string_map[k] = obj

        self.insert(tk.END, k)

    def get_selected(self):
        k = self.get(self.curselection())
        return self.string_map[k]

class HyperlinkManager:

    def __init__(self, text):

        self.text = text

        self.text.tag_config("hyper", foreground="blue", underline=1)

        self.text.tag_bind("hyper", "<Enter>", self._enter)
        self.text.tag_bind("hyper", "<Leave>", self._leave)
        self.text.tag_bind("hyper", "<Button-1>", self._click)

        self.reset()

    def reset(self):
        self.links = {}

    def add(self, action):
        # add an action to the manager.  returns tags to use in
        # associated text widget
        tag = "hyper-%d" % len(self.links)
        self.links[tag] = action
        return "hyper", tag

    def _enter(self, event):
        self.text.config(cursor="hand2")

    def _leave(self, event):
        self.text.config(cursor="")

    def _click(self, event):
        for tag in self.text.tag_names(tk.CURRENT):
            if tag[:6] == "hyper-":
                self.links[tag]()
                return


class DSSSettingsFrame(tk.Frame):
    def __init__(self, master, settings:Iterable[Tuple[str, object]], setter:Callable[[str,object],None], **kwargs):
        if not 'cnf' in kwargs:
            kwargs['cnf'] = {}
        kwargs['bg'] = 'gray82'
        super().__init__(master, width=120, **kwargs)
        self._refs = []
        self._cnt = 0

        self.settings = settings
        def log(f):
            def inner(*args, **kwargs):
                print(f'Called {f.__name__} with args {args} kwargs {kwargs}')
                f(*args, **kwargs)
            return inner
        self.setter = log(setter)

        for k,v in sorted(settings, key=lambda tup: type(tup[1]).__name__):
            self._add_editor(k, v)

    def _add_editor(self, key, val):
        label = tk.Label(self, text=key)
        label.grid(row=int(self._cnt / 2 + 1), column=self._cnt % 2, sticky='wns')
        self._cnt += 1
        if isinstance(val, bool):
            var = tk.BooleanVar()
            var.set(val)
            self._refs.append(var) # Bug in Tkinter? This reference is somehow needed
            btn = tk.Checkbutton(self, variable=var, bg='gray82')
            btn.grid(row=int(self._cnt / 2 + 1), column=self._cnt % 2, sticky='wns')
            var.trace_add('write', lambda *_: self.setter(key, val.get()))

        elif isinstance(val, int) or isinstance(val, float) or isinstance(val, str):
            t = type(val)
            entry = tk.Entry(self)
            entry.insert(0, val)
            entry.grid(row=int(self._cnt / 2 + 1), column=self._cnt % 2, sticky='wns')
            entry.bind('<FocusOut>', lambda *_: self.setter(key, t(entry.get())))
        self._cnt += 1

    def _get_callback(self, key, parse_func):
        pass

    @classmethod
    def from_settings(cls, master, category):
        return cls(master, settings.get_by_category(category), settings.set_setting)

    @classmethod
    def from_object(cls, master, obj):
        kvps = ((k,v) for k,v in obj.__dict__.items() if not k.startswith('_'))
        setter = lambda k,v: setattr(obj, k, v)
        return cls(master, kvps, setter)




    def add_settings(self, cls):
        def callback_factory(cls, name, bv):
            return lambda *args: cls.set_setting(name, bv.get())

        if not hasattr(cls, 'get_settings'):
            return

        settings = cls.get_settings()
        for name,value in settings.items():
            if name in self.settings:
                continue

            bv = tk.BooleanVar()
            bv.set(value)
            self.settings[name] = bv # Bug in Tkinter? This reference is somehow needed

            # Add a trace to a lambda function that updates the setting
            bv.trace_add("write", callback = callback_factory(cls, name, bv))
            button = tk.Checkbutton(self, text=name, variable=bv,
                                    bg='gray82', highlightthickness=0, justify=tk.LEFT)
            button.grid(row=int(self.counter/2+1), column=self.counter % 2, sticky='wns')
            self.counter += 1

record = collections.defaultdict(list)

def log(func):
    def func_wrapper(*args, **kwargs):
        return_value = func(*args, **kwargs)
        record[func_wrapper].append(return_value)
        return return_value



    return func_wrapper


def R(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, -s],
                     [s, c]])

def min_by(iterable, func):
    min_val = 2**31
    cur_item = None
    for item in iterable:
        k = func(item)
        if k < min_val:
            min_val = k
            cur_item = item
    return cur_item