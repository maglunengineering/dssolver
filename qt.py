import os
import sys
import typing
from typing import Tuple, Collection, Set, Callable
import numpy as np

from PySide6 import QtCore, QtWidgets, QtGui

dirname = os.path.dirname(QtCore.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path



#Pt = typing.Union[np.ndarray, typing.Iterable[int], typing.Iterable[float]]
Pt = typing.Iterable[float]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('DSS')

        #toolbar = QtWidgets.QToolBar('topmenu')
        #toolbar.setIconSize(QtCore.QSize(16, 16))
        #self.addToolBar(toolbar)
        #btn = QtGui.QAction('QAction ctor arg')
        #btn.setStatusTip('setStatusTip arg')
        #btn.triggered.connect(lambda: print('btn.triggered.connect lambda arg call'))
        #toolbar.addAction(btn)
        #toolbar.addSeparator()
        #toolbar.addWidget(QtWidgets.QLabel('toolbar.addWidget QLabel'))


        menu = self.menuBar()
        file_menu = menu.addMenu('File')
        file_menu.addAction("Open", lambda: print('Clicked Open'))

        tabs = QtWidgets.QTabWidget()
        tabs.setTabPosition(QtWidgets.QTabWidget.North)
        tabs.setMovable(False)
#
        tabs.addTab(DSSQCanvas(), 'a')

        self.setCentralWidget(tabs)


class DSSQCanvas(QtWidgets.QLabel):
    def __init__(self, *args):
        super().__init__()
        self.canvas = QtGui.QPixmap(400, 300)
        self.canvas.fill(QtCore.Qt.white)
        self.painter = QtGui.QPainter(self.canvas)
        self.setPixmap(self.canvas)

        #mpe = self.canvas.mousePressEvent;

        self.T:np.ndarray = np.array([
            [1, 0, -50],
            [0, -1, 100],
            [0, 0, 1]], dtype=float)
        self.tx = 0
        self.ty = 0
        self.prev_x = None
        self.prev_y = None

        self.draw_point((0, 0), 5)
        self.setPixmap(self.canvas)
        self.painter.end()

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        print('mousePressEvent')
        print(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        print('mouseMoveEvent')
        pos = ev.localPos()
        x, y = pos.x(), pos.y()
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = x
            self.prev_y = y

        self.T[0:, 2] = self.T[:, 2] + np.array([-self.tx, self.ty, 0])

        self.tx = (x - self.prev_x) * self.T[0, 0]
        self.ty = (y - self.prev_y) * self.T[0, 0]
        self.prev_x = x
        self.prev_y = y

    def draw_line(self, pt1, pt2, *args, **kwargs):
        r1 = self.transform(pt1)
        r2 = self.transform(pt2)
        self.painter.drawLine(QtCore.QLine(QtCore.QPoint(*r1), QtCore.QPoint(*r2)))

    def draw_point(self, pt, radius, *args, **kwargs):
        pt_canvas = self.transform(pt) - radius//2
        self.painter.drawPoint(*pt)
        self.painter.drawEllipse(*pt_canvas, radius, radius)

    def move(self, event):
        if self.prev_x is None or self.prev_y is None:
            self.prev_x = event.x
            self.prev_y = event.y

        self.T[0:,2] = self.T[:,2] + np.array([-self.tx, self.ty, 0])

        self.tx = (event.x - self.prev_x) * self.T[0,0]
        self.ty = (event.y - self.prev_y) * self.T[0,0]
        self.prev_x = event.x
        self.prev_y = event.y

    def transform(self, pt:Pt) -> np.ndarray:
        return np.linalg.solve(self.T, np.array([*pt, 1]))[0:2]



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    w = MainWindow()
    w.show()

    app.exec()