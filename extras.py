import tkinter as tk
import numpy as np
from numpy.linalg import solve

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
        wscale = float(event.width)/self.width
        hscale = float(event.height)/self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        #self.scale("all",0,0,wscale,hscale)



class DSSCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.bind("<Configure>", self.on_resize)
        self.bind('<B1-Motion>', self.move)
        self.bind('<Double-Button-1>', self.scaleup)
        self.bind('<Double-Button-2>', self.scaledown)
        self.bind('<ButtonRelease-1>', self.on_lbuttonup)
        self.height = 512
        self.width = 768

        self.tx = 0
        self.ty = 0
        self.prev_x = None
        self.prev_y = None

        # Transforms from canvas/screen coordinates to problem space coordinates, or the sender way
        self.transformation_matrix = np.array([[1, 0, -50], [0, -1, 100], [0, 0, 1]], dtype=float)

        self.dss = None
        if 'dss' in kwargs:
            self.dss = kwargs['dss']

        self.objects = []

    def on_resize(self, event):
        self.width = event.width
        self.height = event.height

        self.config(width=self.width, height=self.height)

    def draw_node(self, pt, radius, *args, **kwargs):
        pt_canvas = self.transform(pt)
        super().create_oval(*np.hstack((pt_canvas - radius, pt_canvas + radius)), *args, **kwargs)

    def draw_point(self, pt, radius, *args, **kwargs):
        pt_canvas = self.transform(pt)
        super().create_oval(*np.hstack((pt_canvas - radius, pt_canvas + radius)), *args, **kwargs)

    def draw_line(self, pt1, pt2, *args, **kwargs):
        r1 = self.transform(pt1)
        r2 = self.transform(pt2)
        linewidth = 0.5
        super().create_line(*np.hstack((r1, r2)), width=linewidth, *args, **kwargs)

    def draw_arc(self, arc_start, arc_mid, arc_end, **kwargs):
        arc_start = self.transform(arc_start)
        arc_mid = self.transform(arc_mid)
        arc_end = self.transform(arc_end)
        super().create_line(*arc_start, *arc_mid, *arc_end, **kwargs)

    def draw_polygon(self, pts, *args, **kwargs):
        canvas_pts = []
        for pt in pts:
            canvas_pt = np.linalg.solve(self.transformation_matrix, np.array([*pt, 1]))[:2]
            canvas_pts.extend(canvas_pt)
        self.create_polygon(*canvas_pts, *args, **kwargs)

    def draw_text(self, pt, text, *args, **kwargs):
        canvas_pt = self.transform(pt)
        self.create_text(*canvas_pt, text=text, *args, **kwargs)

    def transform(self, pt):
        return np.linalg.solve(self.transformation_matrix, np.array([*pt, 1]))[0:2]

    def redraw(self):
        self.delete('all')
        for obj in self.objects:
            obj.draw_on_canvas(self)

    def move(self, event):
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = event.x
            self.prev_y = event.y

        self.transformation_matrix[0:,2] = self.transformation_matrix[:,2] + np.array([-self.tx, self.ty, 0])

        self.tx = (event.x - self.prev_x) * self.transformation_matrix[0,0]
        self.ty = (event.y - self.prev_y) * self.transformation_matrix[0,0]
        self.prev_x = event.x
        self.prev_y = event.y
        self.dss.draw_canvas()

    def on_lbuttonup(self, event):
        self.prev_x = None
        self.prev_y = None


    #def get_by_coordinates(self, x, y):
    #    xyz = self.transformation_matrix@np.array([x, y, 1])
    #    for obj in self.objects:
    #        if obj.canvas_highlight(self, xyz[:2]):
    #            return obj

    def add_object(self, obj):
        self.objects.append(obj)
        obj.draw_on_canvas(self)

    def scaleup(self, event):
        self.transformation_matrix[0:2, 0:2] = self.transformation_matrix[0:2, 0:2]*0.8
        self.redraw()

    def scaledown(self, event):
        self.transformation_matrix[0:2, 0:2] = self.transformation_matrix[0:2, 0:2]*1.2
        self.redraw()




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