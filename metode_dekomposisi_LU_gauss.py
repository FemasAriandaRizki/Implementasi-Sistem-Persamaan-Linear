# Nama  : Femas Arianda Rizki
# NIM   : 21120122130080
# Kelas : Metode Numerik - B

import numpy as np
import unittest

# Function for LU gauss decomposition
def lu_gauss_decomposition(matrix):
    n = len(matrix)
    lower = np.zeros((n, n))
    upper = np.zeros((n, n))

    for i in range(n):
        lower[i][i] = 1

        for j in range(i, n):
            sum = 0
            for k in range(i):
                sum += (lower[i][k] * upper[k][j])

            upper[i][j] = matrix[i][j] - sum

        for j in range(i + 1, n):
            sum = 0
            for k in range(i):
                sum += (lower[j][k] * upper[k][i])

            lower[j][i] = (matrix[j][i] - sum) / upper[i][i]

    return lower, upper

# Function performs forward substitution  substitution on the lower matrix and the constants matrix (B)
def forward_substitution(lower, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i][0]
        for j in range(i):
            y[i] -= lower[i][j] * y[j]

    return y

# Function performs backward substitution on the upper matrix and the result of backward substitution
def back_substitution(upper, y):
    n = len(y)
    x = np.zeros((n, 1))

    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= upper[i][j] * x[j]
        x[i] /= upper[i][i]

    return x

# Function solves a system of linear equations using LU gauss decomposition and forward-backward substitution
def solve_linear_equations(matrix, b):
    lower, upper = lu_gauss_decomposition(matrix)
    y = forward_substitution(lower, b)
    x = back_substitution(upper, y)
    return x

# Define the coefficient matrix (A) and constants matrix (B)
A = np.array([
                [1, -1, 2],
                [3, 0, 1],
                [1, 0, 2]])

B = np.array([
                [5],
                [10],
                [5]])

# Solve the linear equations
X = solve_linear_equations(A, B)
print("Matriks solusi dari persamaan linear:")
print(X)

# Print the solution
print("Jadi solusi dari persamaan linear adalah:")
print("x = {:.1f}".format(X[0][0]))
print("y = {:.1f}".format(X[1][0]))
print("z = {:.1f}".format(X[2][0]))

# Test code
class TestLinearEquations(unittest.TestCase):
    def test_lu_gauss_decomposition_(self):
        A = np.array([
                        [1, -1, 2],
                        [3, 0, 1],
                        [1, 0, 2]])

        B = np.array([
                        [5],
                        [10],
                        [5]])

        X = solve_linear_equations(A, B)
        self.assertAlmostEqual(X[0][0], 3)
        self.assertAlmostEqual(X[1][0], 0)
        self.assertAlmostEqual(X[2][0], 1)

if __name__ == '__main__':
    unittest.main()