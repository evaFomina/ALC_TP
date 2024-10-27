"""

Trabajo Práctico 1 - Implementacion de funciones 

Álgebra Lineal Computacional - 2do cuatrimestre 2024

Grupo: TIBURONES

Integrantes:
- Victoria Pérez Olivera
- Ignacio Gallego
- Evangelina Fomina
   
"""




import numpy

def calcularLU(A):
    """
    Calcula la descomposición LU de una matriz cuadrada A.
    
    Parámetros:
    A : numpy.ndarray
        Matriz cuadrada que se desea descomponer.
        
    Retorna:
    L : numpy.ndarray
        Matriz triangular inferior.
    U : numpy.ndarray
        Matriz triangular superior.
    P : numpy.ndarray
        Matriz de permutación utilizada en el proceso de descomposición.
    """
    m = A.shape[0]  # Número de filas de A
    n = A.shape[1]  # Número de columnas de A
    
    U = A.copy()  # Copia de A para realizar la descomposición
    U = U.astype(float)  # Asegura que U sea de tipo float
    
    if m != n:
        print('Matriz no cuadrada')  # Verifica que la matriz sea cuadrada
        return

    P = numpy.eye(n)  # Matriz de permutación inicial (identidad)
    L = numpy.eye(n)  # Matriz triangular inferior inicial (identidad)
    L = L.astype(float)  # Asegura que L sea de tipo float

    for i in range(n):
        Pj = numpy.eye(n)  # Matriz de permutación para la columna actual

        # Si el pivote es cero, se busca una fila para intercambiar
        if U[i, i] == 0:  
            for j in range(i + 1, n):
                if U[j, i] != 0:  # Se encuentra un pivote no cero
                    Pj[i, :] += Pj[j, :]  # Intercambio de filas
                    Pj[j, :] = Pj[i, :] - Pj[j, :]
                    Pj[i, :] -= Pj[j, :]
                    P = Pj @ P  # Actualiza la matriz de permutación
                    break
                elif j == n - 1:
                    print('Todos los coeficientes de esta columna son 0')
                    break    

        U = Pj @ U  # Aplica la permutación a U
        L = Pj @ L @ Pj  # Actualiza L con la permutación

        # Eliminación hacia adelante
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]  # Calcula el factor de eliminación
            L[j, i] = factor  # Almacena el factor en L
            U[j, i:] = U[j, i:] - factor * U[i, i:]  # Actualiza U

    return L, U, P  # Retorna las matrices L, U y P

def inversaLU(L, U, P):
    """
    Calcula la inversa de una matriz utilizando la descomposición LU.
    
    Parámetros:
    L : numpy.ndarray
        Matriz triangular inferior.
    U : numpy.ndarray
        Matriz triangular superior.
    P : numpy.ndarray
        Matriz de permutación utilizada en el proceso de descomposición.
        
    Retorna:
    numpy.ndarray
        Matriz inversa de la matriz original.
    """
    return invertir(U) @ invertir(L) @ P  # Retorna la inversa calculada

def leontiefizar(A):
    """
    Calcula la matriz Leontief a partir de la matriz insumo-producto A.
    
    Parámetros:
    A : numpy.ndarray
        Matriz insumo-producto que se desea transformar.
        
    Retorna:
    numpy.ndarray
        Matriz Leontief resultante.
    """
    n = A.shape[0]  # Número de sectores (filas de A)
    I_A = numpy.eye(n) - A  # Matriz identidad menos A
    Low, Up, P = calcularLU(I_A)  # Descomposición LU de I - A
    return inversaLU(Low, Up, P)  # Retorna la inversa de la matriz resultante

'''
FUNCIONES AUXILIARES
'''

def invertir(M):
    """
    Calcula la inversa de una matriz M utilizando eliminación de Gauss.
    
    Parámetros:
    M : numpy.ndarray
        Matriz que se desea invertir.
        
    Retorna:
    numpy.ndarray
        Matriz inversa de M.
    """
    A = numpy.copy(M)  # Copia de la matriz original
    A = A.astype(float)  # Asegura que A sea de tipo float

    A_aug = numpy.hstack((A, numpy.eye(A.shape[0])))  # Matriz aumentada
    A_inv = sustHaciaAtras(triangularizarU(A_aug))  # Aplicar sustitución hacia atrás en U

    return A_inv  # Retorna la matriz inversa

def sustHaciaAtras(A_aug):
    """
    Realiza la sustitución hacia atrás sobre una matriz aumentada.
    
    Parámetros:
    A_aug : numpy.ndarray
        Matriz aumentada que se desea resolver.
        
    Retorna:
    numpy.ndarray
        Parte de la matriz que contiene la solución.
    """
    n = A_aug.shape[0]  # Número de filas

    for i in range(n - 1, -1, -1):
        A_aug[i] = A_aug[i] / A_aug[i, i]  # Normaliza la fila actual
        
        for j in range(i):
            A_aug[j] -= A_aug[i] * A_aug[j, i]  # Elimina la variable de la fila j

    return A_aug[:, n:]  # Retorna solo la parte de solución

def triangularizarU(M):
    """
    Transforma una matriz M en forma triangular superior.
    
    Parámetros:
    M : numpy.ndarray
        Matriz que se desea triangularizar.
        
    Retorna:
    numpy.ndarray
        Matriz triangular superior resultante.
    """
    A = numpy.copy(M)  # Copia de la matriz original
    A = A.astype(float)  # Asegura que A sea de tipo float

    f, c = A.shape  # Obtiene el número de filas y columnas
    if f == 0 or c == 0:
        return A  # Retorna la matriz vacía si no hay filas o columnas

    i = 0

    # Encuentra la primera fila no cero
    while i < f and A[i, 0] == 0:
        i += 1

    if i == f:
        B = triangularizarU(A[:, 1:])  # Recursión si no se encuentra fila no cero
        return numpy.block([A[:, :1], B])  # Retorna matriz con columna inicial

    if i > 0:
        A[[0, i], :] = A[[i, 0], :]  # Intercambia filas si es necesario

    # Realiza eliminación hacia adelante
    A[1:, :] -= (A[0, :] / A[0, 0]) * A[1:, 0:1] 

    B = triangularizarU(A[1:, 1:])  # Llama recursivamente para triangularizar el resto

    return numpy.block([[A[:1, :]], [A[1:, :1], B]])  # Retorna la matriz triangular superior

#Defino la funcion metodoPotencia
def metodoPotencia(A, v, k):
    #En esta lista guardamos todos los vectores elevados a las potencias
    vectores = []
    for i in range(k):
        #lo multiplicamos por la matriz y lo dividimos por su norma 2
        Av = A @ v
        v = Av / np.linalg.norm(Av, 2)
        vectores.append(v)
    return (v, vectores)






















