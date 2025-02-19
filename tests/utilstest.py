import unittest
import collections
import time

import numpy as np
import matplotlib
import  matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

import core.problem
from core import problem, solvers

from core.elements import *


class UtilsTest(unittest.TestCase):
    def test_R_should_return_matrix_in_last_two_shapes_0d(self):
        angle = 0.1
        T = R(angle)

        self.assertEqual((2,2), T.shape)

    def test_R_should_return_matrix_in_last_two_shapes_1d(self):
        angle = np.array([0.1])
        T = R(angle)

        self.assertEqual((2,2), T.shape)

    def _test_R_should_return_matrix_in_last_two_shapes_1x1(self):
        angle = np.array([0.1]).reshape((1,1))
        T = R(angle)

        self.assertEqual((1,1,2,2), T.shape)

    def test_R_should_return_matrix_in_last_two_shapes_2x1(self):
        angle = np.array([0.1, 0.15]).reshape((2, 1))
        T = R(angle)

        self.assertEqual((2,2,2), T.shape)
        self.assertTrue(np.allclose(T[0], R(0.1)))
        self.assertTrue(np.allclose(T[1], R(0.15)))