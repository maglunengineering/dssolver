import unittest

from elements import *

class ElementTest(unittest.TestCase):
    def setUp(self):
        self.n1 = Node((0, 0))
        self.n2 = Node((1000, 0))

        self.rod = Rod(self.n1, self.n2)
        self.beam = Beam(self.n1, self.n2)

    def test_should_have_same_axial_stiffness_with_no_displacements(self):

        for indices in ([0,0],[0,3],[3,0],[3,3]):
            i,j = indices
            rod_k00 = self.rod.stiffness_matrix_global()[i,j]
            beam_k00 = self.beam.stiffness_matrix_global()[i,j]

            self.assertEqual(rod_k00, beam_k00)

    def test_should_have_same_axial_stiffness_with_same_axial_displacements(self):
        self.n2.displacements = np.array([-10, 0, 0])

        for indices in ([0,0],[0,3],[3,0],[3,3]):
            i,j = indices
            rod_k00 = self.rod.stiffness_matrix_global()[i,j]
            beam_k00 = self.beam.stiffness_matrix_global()[i,j]

            self.assertEqual(rod_k00, beam_k00)

    def test_should_have_same_forces_with_same_axial_displacements(self):
        self.n2.displacements = np.array([-10, 0, 0])

        self.assertTrue(np.allclose(self.rod.get_forces(), self.beam.get_forces()))