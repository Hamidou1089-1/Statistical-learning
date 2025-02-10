import numpy as np
from qiskit.visualization import array_to_latex



ket0 = np.array([[1],[0]])
ket1 = np.array([[0],[1]])

M1 = np.array([[1, 1], [0, 0]])
M2 = np.array([[1, 0], [0, 1]])
M = M1/2 + M2/2;

display(array_to_latex(np.matmul(M1, ket1)))
display(array_to_latex(np.matmul(M1, M2)))
display(array_to_latex(np.matmul(M, M)))


