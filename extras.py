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

    #@property
    def size(self):
        return (self.height**2 + self.width**2) ** 0.5


class DSSCanvas(ResizingCanvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.transformation_matrix = np.array([[1,0,-50],[0,-1,100],[0,0,1]], dtype=float)

    def create_oval(self, pt, radius, *args, **kwargs):
        pt_canvas = solve(self.transformation_matrix, pt)[0:2]
        super().create_oval(*np.hstack((pt_canvas - radius, pt_canvas + radius)), *args, **kwargs)



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