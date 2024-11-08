"""

Trabajo Práctico 2   Matrices insumo-producto
Álgebra Lineal Computacional - 2do cuatrimestre 2024

Grupo: TIBURONES

Integrantes:
- Victoria Pérez Olivera
- Ignacio Gallego
- Evangelina Fomina

"""


#CONTENIDO RECICLADO DEL TP1

import numpy as np
from numpy.linalg import matrix_power

def calcularLU(A):
    """
    Calcula la descomposición LU de una matriz cuadrada A.

    Parámetros:
    A : np.ndarray
        Matriz cuadrada que se desea descomponer.

    Retorna:
    L : np.ndarray
        Matriz triangular inferior.
    U : np.ndarray
        Matriz triangular superior.
    P : np.ndarray
        Matriz de permutación utilizada en el proceso de descomposición.
    """
    m = A.shape[0]  # Número de filas de A
    n = A.shape[1]  # Número de columnas de A

    U = A.copy()  # Copia de A para realizar la descomposición
    U = U.astype(float)  # Asegura que U sea de tipo float

    if m != n:
        print('Matriz no cuadrada')  # Verifica que la matriz sea cuadrada
        return

    P = np.eye(n)  # Matriz de permutación inicial (identidad)
    L = np.eye(n)  # Matriz triangular inferior inicial (identidad)
    L = L.astype(float)  # Asegura que L sea de tipo float

    for i in range(n):
        Pj = np.eye(n)  # Matriz de permutación para la columna actual

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
    L : np.ndarray
        Matriz triangular inferior.
    U : np.ndarray
        Matriz triangular superior.
    P : np.ndarray
        Matriz de permutación utilizada en el proceso de descomposición.

    Retorna:
    np.ndarray
        Matriz inversa de la matriz original.
    """
    return invertir(U) @ invertir(L) @ P  # Retorna la inversa calculada

def leontiefizar(A):
    """
    Calcula la matriz Leontief a partir de la matriz insumo-producto A.

    Parámetros:
    A : np.ndarray
        Matriz insumo-producto que se desea transformar.

    Retorna:
    np.ndarray
        Matriz Leontief resultante.
    """
    n = A.shape[0]  # Número de sectores (filas de A)
    I_A = np.eye(n) - A  # Matriz identidad menos A
    Low, Up, P = calcularLU(I_A)  # Descomposición LU de I - A
    return inversaLU(Low, Up, P)  # Retorna la inversa de la matriz resultante

'''
FUNCIONES AUXILIARES
'''

def invertir(M):
    """
    Calcula la inversa de una matriz M utilizando eliminación de Gauss.

    Parámetros:
    M : np.ndarray
        Matriz que se desea invertir.

    Retorna:
    np.ndarray
        Matriz inversa de M.
    """
    A = np.copy(M)  # Copia de la matriz original
    A = A.astype(float)  # Asegura que A sea de tipo float

    A_aug = np.hstack((A, np.eye(A.shape[0])))  # Matriz aumentada
    A_inv = sustHaciaAtras(triangularizarU(A_aug))  # Aplicar sustitución hacia atrás en U

    return A_inv  # Retorna la matriz inversa

def sustHaciaAtras(A_aug):
    """
    Realiza la sustitución hacia atrás sobre una matriz aumentada.

    Parámetros:
    A_aug : np.ndarray
        Matriz aumentada que se desea resolver.

    Retorna:
    np.ndarray
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
    M : np.ndarray
        Matriz que se desea triangularizar.

    Retorna:
    np.ndarray
        Matriz triangular superior resultante.
    """
    A = np.copy(M)  # Copia de la matriz original
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
        return np.block([A[:, :1], B])  # Retorna matriz con columna inicial

    if i > 0:
        A[[0, i], :] = A[[i, 0], :]  # Intercambia filas si es necesario

    # Realiza eliminación hacia adelante
    A[1:, :] -= (A[0, :] / A[0, 0]) * A[1:, 0:1]

    B = triangularizarU(A[1:, 1:])  # Llama recursivamente para triangularizar el resto

    return np.block([[A[:1, :]], [A[1:, :1], B]])  # Retorna la matriz triangular superior


#CONTENIDO DESARROLLADO PARA TP2


def metodoPotencia(A, v, k):
    """
    Usa el método de la potencia para hallar el autovalor principal de A
    
    Parámetros:
    A:  np.ndarray
        Matriz cuadrada   
    v:  np.ndarray
        Vector semilla para el método de la potencia
    k : entero
        Cantidad de iteraciones deseadas 
        
    Retorna: 
    float
        Autovalor de A con mayor valor absoluto
    """  
    
    v = v / np.linalg.norm(v, 2)

    for i in range(k):
        Av = A @ v
        v = Av / np.linalg.norm(Av, 2)
        l = v.T@A@v
    return (l)



def metodoMonteCarlo(A,k):
    """
    Usa el método de Montecarlo para hallar un rango de valores para el autovalor principal de A
    
    Parámetros:
    A:  np.ndarray
        Matriz cuadrada   
    k : entero
        Cantidad de iteraciones deseadas 
        
    Retorna: 
    float
        Promedio de autovalor de A con mayor valor absoluto
    float
        Desvio estandar de autovalor de A con mayor valor absoluto   
        
    """     
    
    
    
    
    avals = np.zeros(k)

    for i in range(k):
        v = np.random.rand(A.shape[0])
        l=metodoPotencia(A, v, k)
        avals[i]=l
    return avals.mean().round(4), avals.std().round(4)



def seriePotencia(A, n):
    """
    Desarrolla la matriz de Leontief de A como una serie de potencias de la matriz A. 
    
    Parámetros:
    A:  np.ndarray
        Matriz cuadrada   
    n : entero
        Potencia hasta la que se desea desarrollar 
        
    Retorna: 
    np.ndarray
        Matriz equivalente al desarrollo hasta n
    list de floats
        Normas de la serie desde 1 hasta n
    """    
       
    normas = []
    suma = np.eye(A.shape[0])
    for i in range(1, n+1):
        suma += matrix_power(A, i)
        normas.append(np.linalg.norm(suma, 2))
    return (suma, normas)



def En (n):
    """
    Devuelve la matriz E(n)
    
    Parámetros:
    n : entero
        Tamaño deseado de la matriz E(n)
        
    Retorna: 
    np.ndarray
        Matriz cuadrada E(n) del tamaño indicado    
    """
    return np.eye(n)-(1/n)*np.outer(np.ones(n),np.ones(n))


def hotelling(A, k, e, max_iter=1000):
    """
    Calcula los primeros k autovectores de la matriz A.
    
    Parámetros:
    A : np.ndarray
        Matriz cuadrada cuyos autovectores se desea encontrar.
    k : int
        Cantidad de autovectores deseados. 
    e : float
        Margen de error deseado.
    max_iter : int
        Máximo número de iteraciones para evitar bucles infinitos.         
    
    Retorna:
    list
        Lista de autovectores.
    list
        lista de autovalores en el mismo orden.    
    """    
    x = np.random.rand(A.shape[0])
    x = x / np.linalg.norm(x, 2)
    
    avecs = []
    avals = []
    
    for i in range(k):
        iter_count = 0
        while True:
            x_prev = x
            x = A @ x
            x = x / np.linalg.norm(x, 2)
            iter_count += 1

            # Criterio de parada: o llega al margen de error deseado o frena a los 10.000 ciclos (esto evita bucles infinitos)
            if np.linalg.norm(x - x_prev, 2) < e or iter_count >100000:
                break

        # Calculamos el autovalor correspondiente al autovector encontrado
        l = (x.T @ A @ x) / (x.T @ x)
        
        # Actualizamos la matriz A para eliminar la contribución del autovalor encontrado
        A = A - l * np.outer(x, x)
             
        avecs.append(x)
        avals.append(l)
        
    return avecs, avals
