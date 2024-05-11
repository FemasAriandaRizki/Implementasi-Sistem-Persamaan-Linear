# Nama  : Femas Arianda Rizki
# NIM   : 21120122130080
# Kelas : Metode Numerik - B

import numpy as np
import unittest

def crout_reduction(matrix):
    n = len(matrix)
    lower = np.zeros((n, n))
    upper = np.zeros((n, n))

    for j in range(n):
        upper[j][j] = 1

        for i in range(j, n):
            sum = 0
            for k in range(j):
                sum += lower[i][k] * upper[k][j]
            lower[i][j] = matrix[i][j] - sum

        for i in range(j+1, n):
            sum = 0
            for k in range(j):
                sum += lower[j][k] * upper[k][i]
            if lower[j][j] == 0:
                return None  # Division by zero, no unique solution
            upper[j][i] = (matrix[j][i] - sum) / lower[j][j]

    return lower, upper

def forward_substitution(lower, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i][0]
        for j in range(i):
            y[i] -= lower[i][j] * y[j]
        y[i] /= lower[i][i]

    return y

def back_substitution(upper, y):
    n = len(y)
    x = np.zeros((n, 1))

    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= upper[i][j] * x[j]
        x[i] /= upper[i][i]

    return x

def solve_linear_equations(matrix, b):
    lower, upper = crout_reduction(matrix)
    if lower is None or upper is None:
        return None  # No unique solution
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
if X is not None:
    print("Solusi dari persamaan linear adalah:")
    print("x =", X[0])
    print("y =", X[1])
    print("z =", X[2])
else:
    print("Tidak ada solusi unik untuk sistem persamaan linear.")

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