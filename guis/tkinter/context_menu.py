import functools
import inspect
import extras
import tkinter
from collections import defaultdict
from core import problem, elements, settings, solvers, results




class ContextMenu:
    """ The context menu is an object that can be shown in a settings frame
    """
    def __init__(self, items, gui:'DSSGUI'):
        self._items = items
        self._gui = gui
        self._problem:problem.Problem = gui.problem

        typecnt = defaultdict(int)
        for item in items:
            typecnt[type(item)] += 1

        cm_keys = [k for k in dir(self) if k.startswith('_cm_')]
        for k in cm_keys:
            val = getattr(self, k)
            if callable(val):
                counts = defaultdict(int)
                for p in inspect.signature(val).parameters.values():
                    counts[p.annotation] += 1

                if counts == typecnt:
                    setattr(self, k[4:], functools.partial(val, *items))

    def _cm_create_beam(self, node1:elements.Node, node2:elements.Node):
        props = self._show_input_dialog({'A':100, 'E':210000, 'I':1e4/12})
        beam = self._problem.create_beam(node1, node2, **props)
        self._gui.update_canvas()

    def _cm_create_quad(self, node1:elements.Node, node2:elements.Node, node3:elements.Node, node4:elements.Node):
        props = self._show_input_dialog({'E':210000, 'v':0.3, 't':1})

        quad = elements.Quad4(node1, node2, node3, node4, **props)
        self._problem.elements.append(quad)
        self._gui.update_canvas()

    def _show_input_dialog(self, kvps):
        # Shows an input dialog for the user to populate remaining arguments.
        # **kvps are key-value pairs. Values better be editable in a DSSSettingsFrame
        wnd = tkinter.Toplevel()
        
        frame = extras.DSSSettingsFrame.from_dictionary(wnd, kvps)
        frame.pack()
        btn_ok = tkinter.Button(wnd, text="Ok", command=lambda *a: wnd.destroy())
        btn_ok.pack()

        wnd.grab_set() # Makes the window modal
        wnd.wait_window() # Blocks until the window closes

        return kvps



