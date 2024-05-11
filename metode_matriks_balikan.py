# Nama  : Femas Arianda Rizki
# NIM   : 21120122130080
# Kelas : Metode Numerik - B

import numpy as np
import unittest

# Function for solving systems of linear equations using the matrix inversion method
def inverse_matrix_method(matrix_A, matrix_B):
    try:
        A_inv = np.linalg.inv(matrix_A)
        X = np.dot(A_inv, matrix_B)
        return X
    except np.linalg.LinAlgError:
        return None

# Define the coefficient matrix (A) and constants matrix (B)
A = np.array([
                [1, -1, 2],
                [3, 0, 1],
                [1, 0, 2]])
B = np.array([
                [5],
                [10],
                [5]])

# Condition whether the inverse matrix can be applied or not
A_det = np.linalg.det(A)
if A_det == 0:
    print("Determinan matriks A adalah 0, tidak dapat dilakukan invers atau balikan.")
else:
    X = inverse_matrix_method(A, B)
    print("Matriks solusi dari persamaan linear:")
    print(X)

    # Print the solution
    print("Jadi solusi dari persamaan linear adalah:")
    print("x = {:.1f}".format(X[0][0]))
    print("y = {:.1f}".format(X[1][0]))
    print("z = {:.1f}".format(X[2][0]))

# Test code
class TestLinearEquations(unittest.TestCase):
    def test_inverse_matrix(self):
        A = np.array([
                        [1, -1, 2],
                        [3, 0, 1],
                        [1, 0, 2]])

        B = np.array([
                        [5],
                        [10],
                        [5]])

        X = inverse_matrix_method(A, B)
        self.assertAlmostEqual(X[0][0], 3)
        self.assertAlmostEqual(X[1][0], 0)
        self.assertAlmostEqual(X[2][0], 1)

    def test_no_inverse_matrix(self):
        A = np.array([
                        [1, 2, 3],
                        [4, 8, 12],
                        [7, 8, 9]])

        B = np.array([
                        [6],
                        [15],
                        [24]])

        self.assertIsNone(inverse_matrix_method(A,B))

if __name__ == '__main__':
    unittest.main()
