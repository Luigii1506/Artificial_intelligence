import numpy as np


# A = Inputs, W = Pesos, N = Numero de neuronas de la capa, Q = Instancias
def salidas_neuronas(A, W, N, Q):
    a_aumentada = np.vstack([A, np.ones((1, Q))])
    w_aumentada = np.append(W, np.ones((N, 1)), 1)
    Z = np.dot(w_aumentada, a_aumentada)
    a = np.tanh(Z)
    return a


N0 = 2  # Neuronas en input layer
N1 = 3  # Neuronas en hidden layer
N2 = 2  # Neuronas en el output layer
Q = 4   # Numero de instancias

# Targets [ 4 x 2 ]
targets = np.array([[3.52, 4.02], [5.43, 6.23], [4.95, 5.76], [4.70, 4.28]])

# Inputs [ 4 x 2 ]
inputs = np.array([[4.7, 6.0], [6.1, 3.9], [2.9, 4.2], [7.0, 5.5]])

# Inputs transpuestos [ 2 x 4 ]
inputs_transpuesta = np.transpose(inputs)

# Pesos [ 3 x 2 ]
w1 = np.random.uniform(-1.0, 1.0, (N1, N0))

# Salidas de las neuronas N1 [ 3 x 4 ]
a1 = salidas_neuronas(inputs_transpuesta, w1, N1, Q)

# Weights [ 2 x 3 ]
w2 = np.random.uniform(-1.0, 1.0, (N2, N1))

# Salida de las neuronas N1 [ 2 x 4 ]
a2 = salidas_neuronas(a1, w2, N2, Q)

# Salida de las neuronas capa N1 transpuesta
a2_transpuesta = np.transpose(a2)

# Errores
E = np.subtract(targets, a2_transpuesta)

print(E)
