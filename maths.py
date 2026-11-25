"""
=============================================================================
 maths.py
 BIBLIOTECA DE MÉTODOS NUMÉRICOS: funciones de numpy (np_) y propias (my_)
 Proyecto Ares-I · Grado en Ingeniería Aeroespacial · Programación
=============================================================================
 Este módulo SOLO contiene matemáticas: no conoce los datos del cohete ni
 dibuja nada. physics.py importa de aquí las funciones de cada apartado.

 Para cada apartado aparece siempre primero la versión con NUMPY (np_) y
 después las funciones PROPIAS (my_), en el mismo orden que en physics.py:

   APARTADO 1 - SISTEMAS LINEALES
       np_sistema_lineal ......... np.linalg.solve
       my_gauss .................. Eliminación de Gauss con pivoteo parcial escalado
       my_lu / my_lu_solve ....... Factorización LU (P·A = L·U) y resolución
       my_jacobi ................. Método iterativo de Jacobi
       my_gauss_seidel ........... Método iterativo de Gauss-Seidel

   APARTADO 2 - REGRESIÓN LINEAL
       np_regresion_lineal ....... np.polyfit (grado 1)
       my_linear_regression ...... Mínimos cuadrados (ecuación normal + my_gauss)

   APARTADO 3 - INTEGRACIÓN NUMÉRICA
       np_trapecio ............... np.trapezoid
       my_trapecio ............... Regla del trapecio compuesta (valor total)
       my_trapecio_acumulado ..... Regla del trapecio acumulada (perfil completo)
       my_simpson ................ Regla de Simpson 1/3 compuesta

   APARTADO 4 - INTERPOLACIÓN
       np_interp_lineal .......... np.interp
       np_polinomio_interpolante . np.polyfit (grado n-1) + np.polyval -> Lagrange
       my_interp_lineal .......... Interpolación lineal a trozos
       my_vandermonde ............ Polinomio interpolador por matriz de Vandermonde
       my_lagrange ............... Polinomio interpolador de Lagrange

   APARTADO 5 - DERIVACIÓN NUMÉRICA (fórmulas de 3 puntos, O(h^2))
       np_derivada ............... np.gradient
       my_derivada_centrada, my_derivada_adelantada, my_derivada_retrasada
       my_derivada_numerica ...... centradas en el interior + extremos

   APARTADO 6 - ECUACIONES NO LINEALES
       np_raiz_cambio_signo ...... cambio de signo vectorizado + interpolación lineal
       my_newton ................. Newton-Raphson
       my_biseccion .............. Bisección

 Criterio de diseño: todas las operaciones sobre arrays están vectorizadas con
 numpy. Los únicos bucles que quedan son los INTRÍNSECOS a cada algoritmo
 (avance de pivote en Gauss/LU, iteraciones de Jacobi/Gauss-Seidel/Newton/
 Bisección, grado del polinomio en Horner), que no se pueden vectorizar.

=============================================================================
"""

import numpy as np

# np.trapz pasó a llamarse np.trapezoid en numpy 2.0 -> compatibilidad con ambas
_np_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# =============================================================================
# APARTADO 1: SISTEMAS DE ECUACIONES LINEALES  A·x = b
# =============================================================================

# ----------------------------------- numpy -----------------------------------
def np_sistema_lineal(A, b):
    """Resuelve A·x = b con el módulo de álgebra lineal de numpy (LU de LAPACK)."""
    return np.linalg.solve(A, b)


# ------------------------------------ my_ ------------------------------------
def my_pivoting(A, b, k):
    """
    Pivoteo parcial ESCALADO sobre la columna k (modifica A y b in situ).

    Elige como pivote la fila l >= k que maximiza |A[l,k]| / s_l, donde s_l es
    el mayor elemento (en valor absoluto) de la fila l. Así se evita dividir
    por ceros o por números muy pequeños (inestabilidad numérica).
    """
    s = np.max(np.abs(A[k:, k:]), axis=1)            # factores de escala
    if np.any(s == 0.0):
        raise ValueError("La matriz es singular (fila nula).")
    ratios = np.abs(A[k:, k]) / s
    l = k + int(np.argmax(ratios))                     # fila del mejor pivote
    if ratios[l - k] == 0.0:
        raise ValueError("La matriz es singular (pivote nulo).")
    if l != k:                                         # intercambio de filas
        A[[k, l], :] = A[[l, k], :]
        b[[k, l]] = b[[l, k]]


def my_back_substitution(U, y):
    """Sustitución regresiva: resuelve U·x = y con U triangular superior."""
    N = len(y)
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def my_forward_substitution(L, b):
    """Sustitución progresiva: resuelve L·y = b con L triangular inferior."""
    N = len(b)
    y = np.zeros(N)
    for i in range(N):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def my_gauss(A_in, b_in):
    """
    Eliminación de Gauss con pivoteo parcial escalado + sustitución regresiva.

    La actualización de todas las filas por debajo del pivote se hace de una
    vez con un producto exterior (vectorizado), sin bucle interno de filas.
    """
    A = np.array(A_in, dtype=float)
    b = np.array(b_in, dtype=float)
    N = len(b)

    for k in range(N - 1):
        my_pivoting(A, b, k)
        factores = A[k + 1:, k] / A[k, k]                     # multiplicadores
        A[k + 1:, k:] -= np.outer(factores, A[k, k:])         # anula columna k
        b[k + 1:] -= factores * b[k]

    if A[-1, -1] == 0.0:
        raise ValueError("La matriz es singular.")
    return my_back_substitution(A, b)


def my_lu(A_in):
    """
    Factorización LU de Doolittle con pivoteo parcial:  P·A = L·U

    Devuelve:
        P : matriz de permutación
        L : triangular inferior con unos en la diagonal (multiplicadores)
        U : triangular superior (matriz eliminada)
    El pivoteo es necesario: en el sistema de masas, sin él, aparece un cero
    en la diagonal en el segundo paso y la LU simple fallaría.
    """
    U = np.array(A_in, dtype=float)
    N = U.shape[0]
    L = np.eye(N)
    P = np.eye(N)

    for k in range(N - 1):
        l = k + int(np.argmax(np.abs(U[k:, k])))              # pivoteo parcial
        if U[l, k] == 0.0:
            raise ValueError("La matriz es singular.")
        if l != k:
            U[[k, l], :] = U[[l, k], :]
            P[[k, l], :] = P[[l, k], :]
            L[[k, l], :k] = L[[l, k], :k]                     # multiplicadores ya calculados
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
    return P, L, U


def my_lu_solve(P, L, U, b):
    """Resuelve A·x = b a partir de P·A = L·U:  L·y = P·b ;  U·x = y."""
    y = my_forward_substitution(L, P @ np.asarray(b, dtype=float))
    return my_back_substitution(U, y)


def my_es_diagonal_dominante(A):
    """True si A es estrictamente diagonal dominante por filas (garantiza la
    convergencia de Jacobi y Gauss-Seidel)."""
    A = np.abs(np.asarray(A, dtype=float))
    diagonal = np.diag(A)
    resto = np.sum(A, axis=1) - diagonal
    return bool(np.all(diagonal > resto))


def my_jacobi(A_in, b_in, x0=None, tol=1e-10, max_iter=500):
    """
    Método iterativo de Jacobi:  x^(k+1) = D^-1 · (b - R·x^(k)),  R = A - D

    Cada iteración está totalmente vectorizada (todas las componentes se
    actualizan a la vez con los valores de la iteración anterior).

    Devuelve: (x, n_iteraciones, historial_del_residuo_relativo)
    """
    A = np.asarray(A_in, dtype=float)
    b = np.asarray(b_in, dtype=float)
    D = np.diag(A)
    if np.any(D == 0.0):
        raise ValueError("Jacobi: hay ceros en la diagonal; reordena el sistema.")
    R = A - np.diag(D)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    norma_b = np.linalg.norm(b)
    historial = []

    for k in range(1, max_iter + 1):
        x = (b - R @ x) / D
        residuo = np.linalg.norm(b - A @ x) / norma_b
        historial.append(residuo)
        if residuo < tol:
            return x, k, np.array(historial)
    raise RuntimeError(f"Jacobi no converge en {max_iter} iteraciones.")


def my_gauss_seidel(A_in, b_in, x0=None, tol=1e-10, max_iter=500):
    """
    Método iterativo de Gauss-Seidel: igual que Jacobi pero cada componente
    nueva x_i se usa inmediatamente para calcular las siguientes (converge más
    rápido). El barrido por filas es secuencial por definición del método.

    Devuelve: (x, n_iteraciones, historial_del_residuo_relativo)
    """
    A = np.asarray(A_in, dtype=float)
    b = np.asarray(b_in, dtype=float)
    N = len(b)
    if np.any(np.diag(A) == 0.0):
        raise ValueError("Gauss-Seidel: hay ceros en la diagonal; reordena el sistema.")
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    norma_b = np.linalg.norm(b)
    historial = []

    for k in range(1, max_iter + 1):
        for i in range(N):
            suma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - suma) / A[i, i]
        residuo = np.linalg.norm(b - A @ x) / norma_b
        historial.append(residuo)
        if residuo < tol:
            return x, k, np.array(historial)
    raise RuntimeError(f"Gauss-Seidel no converge en {max_iter} iteraciones.")


# =============================================================================
# APARTADO 2: REGRESIÓN LINEAL POR MÍNIMOS CUADRADOS
# =============================================================================

# ----------------------------------- numpy -----------------------------------
def np_regresion_lineal(x, y):
    """Ajuste y = m·x + n con np.polyfit de grado 1. Devuelve (m, n)."""
    m, n = np.polyfit(x, y, 1)
    return m, n


# ------------------------------------ my_ ------------------------------------
def my_linear_regression(x, y):
    """
    Ajuste y = m·x + n por mínimos cuadrados mediante la ECUACIÓN NORMAL:
            (X^T · X) · theta = X^T · y ,   X = [x | 1]
    El sistema 2x2 resultante se resuelve con nuestra propia my_gauss.

    Devuelve: (m, n)  ->  pendiente y ordenada en el origen
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack((x, np.ones_like(x)))          # matriz de diseño
    theta = my_gauss(X.T @ X, X.T @ y)
    return theta[0], theta[1]


def my_r2(y, y_ajuste):
    """Coeficiente de determinación R^2 (1 = ajuste perfecto)."""
    y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - y_ajuste) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


# =============================================================================
# APARTADO 3: INTEGRACIÓN NUMÉRICA 1D
# =============================================================================

# ----------------------------------- numpy -----------------------------------
def np_trapecio(y, x):
    """Regla del trapecio compuesta de numpy (np.trapezoid / np.trapz)."""
    return _np_trapezoid(y, x)


# ------------------------------------ my_ ------------------------------------
def my_trapecio(y, x):
    """
    Regla del trapecio compuesta (vectorizada):
        I ≈ Σ (x_{i+1} - x_i) · (y_i + y_{i+1}) / 2
    Válida también para espaciado no uniforme. Error O(h^2).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0)


def my_trapecio_acumulado(y, x):
    """
    Regla del trapecio ACUMULADA: devuelve F(x_k) = ∫_{x_0}^{x_k} y dx para
    todos los nodos a la vez (suma acumulada de las áreas de los trapecios).
    F(x_0) = 0.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    areas = np.diff(x) * (y[:-1] + y[1:]) / 2.0
    return np.concatenate(([0.0], np.cumsum(areas)))


def my_simpson(y, x):
    """
    Regla de Simpson 1/3 compuesta (vectorizada):
        I ≈ h/3 · [y_0 + 4·Σ y_impares + 2·Σ y_pares_interiores + y_n]
    Requiere espaciado uniforme y un número PAR de subintervalos. Error O(h^4).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson 1/3 necesita un número par de subintervalos.")
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        raise ValueError("Simpson 1/3 necesita espaciado uniforme.")
    return h / 3.0 * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]) + y[-1])


# =============================================================================
# APARTADO 4: INTERPOLACIÓN 1D
# =============================================================================

# ----------------------------------- numpy -----------------------------------
def np_interp_lineal(x_eval, x_nodos, y_nodos):
    """Interpolación lineal a trozos de numpy (np.interp)."""
    return np.interp(x_eval, x_nodos, y_nodos)


def np_polinomio_interpolante(x_eval, x_nodos, y_nodos):
    """Polinomio interpolador (Lagrange) con numpy: np.polyfit de grado n-1
    pasa exactamente por los n nodos; se evalúa con np.polyval."""
    coefs = np.polyfit(x_nodos, y_nodos, len(x_nodos) - 1)
    return np.polyval(coefs, x_eval)


# ------------------------------------ my_ ------------------------------------
def my_interp_lineal(x_eval, x_nodos, y_nodos):
    """
    Interpolación lineal a trozos, vectorizada (mismo convenio que np.interp:
    fuera del rango de la tabla se mantiene el valor del extremo).
        y = y_0 + (y_1 - y_0) · (x - x_0) / (x_1 - x_0)
    """
    x_nodos = np.asarray(x_nodos, dtype=float)
    y_nodos = np.asarray(y_nodos, dtype=float)
    x_eval = np.clip(np.asarray(x_eval, dtype=float), x_nodos[0], x_nodos[-1])

    idx = np.clip(np.searchsorted(x_nodos, x_eval), 1, len(x_nodos) - 1)
    x0, x1 = x_nodos[idx - 1], x_nodos[idx]
    y0, y1 = y_nodos[idx - 1], y_nodos[idx]
    return y0 + (y1 - y0) * (x_eval - x0) / (x1 - x0)


def my_vandermonde(x_nodos, y_nodos):
    """
    Polinomio interpolador por el método de VANDERMONDE.
    Plantea V·c = y con V[i, j] = x_i^j y resuelve con my_gauss.

    Devuelve los coeficientes c en orden CRECIENTE: p(x) = c0 + c1·x + ... + cn·x^n
    """
    x_nodos = np.asarray(x_nodos, dtype=float)
    potencias = np.arange(len(x_nodos))
    V = x_nodos[:, np.newaxis] ** potencias            # matriz de Vandermonde
    return my_gauss(V, y_nodos)


def my_polyval(coefs_crecientes, x):
    """Evalúa un polinomio (coeficientes en orden creciente) con el esquema de
    Horner; vectorizado sobre todos los puntos x."""
    x = np.asarray(x, dtype=float)
    resultado = np.zeros_like(x)
    for c in coefs_crecientes[::-1]:
        resultado = resultado * x + c
    return resultado


def my_lagrange(x_eval, x_nodos, y_nodos):
    """
    Polinomio interpolador de LAGRANGE, totalmente vectorizado:
        p(x) = Σ_j y_j · L_j(x),   L_j(x) = Π_{m≠j} (x - x_m) / (x_j - x_m)

    Se construye un array 3D (puntos × j × m) y se multiplica a lo largo de m,
    sustituyendo el término m = j por 1. Sin bucles.
    """
    x_nodos = np.asarray(x_nodos, dtype=float)
    y_nodos = np.asarray(y_nodos, dtype=float)
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    n = len(x_nodos)
    diagonal = np.eye(n, dtype=bool)

    # Denominadores: Π_{m≠j} (x_j - x_m)
    dif_nodos = x_nodos[:, None] - x_nodos[None, :]
    denominador = np.prod(np.where(diagonal, 1.0, dif_nodos), axis=1)

    # Numeradores: Π_{m≠j} (x - x_m)  para todos los x a la vez
    dif_eval = x_eval[:, None, None] - x_nodos[None, None, :]       # (Ne, 1, n)
    dif_eval = np.broadcast_to(dif_eval, (len(x_eval), n, n))
    numerador = np.prod(np.where(diagonal[None, :, :], 1.0, dif_eval), axis=2)

    bases_L = numerador / denominador                                # (Ne, n)
    return bases_L @ y_nodos


# =============================================================================
# APARTADO 5: DERIVACIÓN NUMÉRICA - FÓRMULAS DE 3 PUNTOS (error O(h^2))
# =============================================================================

# ----------------------------------- numpy -----------------------------------
def np_derivada(y, x, edge_order=2):
    """Derivada de un vector con np.gradient: centradas en el interior y, en
    los extremos, fórmulas de 3 puntos (edge_order=2) o de 2 puntos (edge_order=1)."""
    return np.gradient(y, x, edge_order=edge_order)


# ------------------------------------ my_ ------------------------------------
def _paso_uniforme(x):
    """Comprueba que el muestreo es uniforme y devuelve el paso h."""
    x = np.asarray(x, dtype=float)
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        raise ValueError("Las diferencias finitas implementadas requieren paso uniforme.")
    return h


def my_derivada_centrada(y, x):
    """
    Diferencias CENTRADAS de 3 puntos:  f'(x_i) ≈ (f_{i+1} - f_{i-1}) / (2h)
    Solo definida en los nodos interiores (extremos = NaN).
    """
    y = np.asarray(y, dtype=float)
    h = _paso_uniforme(x)
    d = np.full_like(y, np.nan)
    d[1:-1] = (y[2:] - y[:-2]) / (2.0 * h)
    return d


def my_derivada_adelantada(y, x):
    """
    Diferencias ADELANTADAS (progresivas) de 3 puntos:
        f'(x_i) ≈ (-3 f_i + 4 f_{i+1} - f_{i+2}) / (2h)
    Definida en todos los nodos salvo los dos últimos (NaN).
    """
    y = np.asarray(y, dtype=float)
    h = _paso_uniforme(x)
    d = np.full_like(y, np.nan)
    d[:-2] = (-3.0 * y[:-2] + 4.0 * y[1:-1] - y[2:]) / (2.0 * h)
    return d


def my_derivada_retrasada(y, x):
    """
    Diferencias RETRASADAS (regresivas) de 3 puntos:
        f'(x_i) ≈ (3 f_i - 4 f_{i-1} + f_{i-2}) / (2h)
    Definida en todos los nodos salvo los dos primeros (NaN).
    """
    y = np.asarray(y, dtype=float)
    h = _paso_uniforme(x)
    d = np.full_like(y, np.nan)
    d[2:] = (3.0 * y[2:] - 4.0 * y[1:-1] + y[:-2]) / (2.0 * h)
    return d


def my_derivada_numerica(y, x, orden_extremos=2):
    """
    Derivada completa de un vector de telemetría:
        - interior -> centradas de 3 puntos, O(h^2)
        - extremos con orden_extremos = 2 (por defecto): adelantada de 3 puntos en
          el primer nodo y retrasada de 3 puntos en el último -> O(h^2) en TODOS los nodos
        - extremos con orden_extremos = 1 (esquema literal del enunciado):
          (f1 - f0)/h en el primer nodo y (f_n - f_{n-1})/h en el último -> O(h)
    """
    y = np.asarray(y, dtype=float)
    d = my_derivada_centrada(y, x)
    if orden_extremos == 1:
        h = _paso_uniforme(x)
        d[0] = (y[1] - y[0]) / h
        d[-1] = (y[-1] - y[-2]) / h
    else:
        d[0] = my_derivada_adelantada(y, x)[0]
        d[-1] = my_derivada_retrasada(y, x)[-1]
    return d


def my_derivada_funcion(f, x, h=1e-5):
    """Derivada de una función continua f en x (centrada, 3 puntos)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


# =============================================================================
# APARTADO 6: ECUACIONES NO LINEALES  f(x) = 0
# =============================================================================

# ----------------------------------- numpy -----------------------------------
def np_raiz_cambio_signo(f, a, b, n_puntos=12001):
    """
    Raíz de f en [a, b] de forma VECTORIZADA (sin bucles):
      1. se evalúa f en un vector denso de n_puntos,
      2. np.diff(np.sign(...)) localiza el primer cambio de signo,
      3. se interpola linealmente entre los dos puntos que lo rodean.
    """
    x = np.linspace(a, b, n_puntos)
    fx = f(x)
    cambios = np.nonzero(np.diff(np.sign(fx)))[0]
    if cambios.size == 0:
        raise ValueError("No hay cambio de signo de f en el intervalo dado.")
    i = cambios[0]
    return x[i] - fx[i] * (x[i + 1] - x[i]) / (fx[i + 1] - fx[i])


# ------------------------------------ my_ ------------------------------------
def my_newton(f, x0, df=None, tol=1e-10, max_iter=100):
    """
    Método de Newton-Raphson:  x_{k+1} = x_k - f(x_k) / f'(x_k)
    Convergencia cuadrática si x0 está cerca de la raíz.

    df : derivada analítica (opcional). Si no se da, se usa my_derivada_funcion.
    Criterio de parada: |x_{k+1} - x_k| < tol.
    Devuelve: (raiz, n_iteraciones)
    """
    x = float(x0)
    for k in range(1, max_iter + 1):
        pendiente = df(x) if df is not None else my_derivada_funcion(f, x)
        if pendiente == 0.0:
            raise ZeroDivisionError("Newton: derivada nula, elige otro punto inicial.")
        dx = -f(x) / pendiente
        x += dx
        if abs(dx) < tol:
            return x, k
    raise RuntimeError(f"Newton no converge en {max_iter} iteraciones.")


def my_biseccion(f, a, b, tol=1e-10, max_iter=200):
    """
    Método de la Bisección: requiere f(a)·f(b) < 0 (teorema de Bolzano).
    Divide el intervalo por la mitad hasta que su anchura es menor que tol.
    Convergencia lineal pero GARANTIZADA.
    Devuelve: (raiz, n_iteraciones)
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Bisección: f(a) y f(b) deben tener signos opuestos.")
    for k in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        fc = f(c)
        if fc == 0.0 or 0.5 * (b - a) < tol:
            return c, k
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    raise RuntimeError(f"Bisección no converge en {max_iter} iteraciones.")
