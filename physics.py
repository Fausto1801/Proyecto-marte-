"""
=============================================================================
 physics.py
 FÍSICA DEL PROBLEMA: datos del Ares-I y resolución de los 6 apartados
 Proyecto Ares-I · Grado en Ingeniería Aeroespacial · Programación
=============================================================================
 Este módulo plantea cada apartado con los datos del enunciado y lo resuelve
 llamando a las funciones de math.py: primero con numpy (np_) y después con
 las funciones propias (my_). No dibuja nada (eso es gui.py).

   1. Constantes y datos de partida (copiados literalmente del enunciado)
   2. Modelos analíticos (referencia "exacta" para medir el error)
   3. Un def por apartado:
        Apartado 1  Sistema lineal ........ np_sistema_lineal | my_gauss | my_lu | my_jacobi | my_gauss_seidel
        Apartado 2  Regresión lineal ...... np_regresion_lineal | my_linear_regression
        Apartado 3  Integración ........... np_trapecio | my_trapecio | my_simpson | my_trapecio_acumulado
        Apartado 4  Interpolación ......... np_interp_lineal | np_polinomio_interpolante | my_interp_lineal | my_vandermonde | my_lagrange
        Apartado 5  Derivación ............ np_derivada | my_derivada_centrada / adelantada / retrasada
        Apartado 6  Ecuación no lineal .... np_raiz_cambio_signo | my_newton | my_biseccion
        Extra       Presión dinámica máxima (Max Q)
   4. Informe por consola: cada resultado con unidades SI y su interpretación física
=============================================================================
"""

import importlib.util
import os

import numpy as np


# --- Carga de math.py (biblioteca numérica del proyecto) ---------------------
# math.py se llama igual que el módulo estándar "math" de Python, así que un
# "import math" normal devolvería el de Python y no el nuestro. Por eso se
# carga indicando su ruta exacta (la misma carpeta que este fichero).
def _cargar_math_py():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math.py")
    spec = importlib.util.spec_from_file_location("math_proyecto", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


mt = _cargar_math_py()

# Apartado 1 - Sistemas lineales
np_sistema_lineal, my_gauss, my_lu, my_lu_solve = mt.np_sistema_lineal, mt.my_gauss, mt.my_lu, mt.my_lu_solve
my_jacobi, my_gauss_seidel, my_es_diagonal_dominante = mt.my_jacobi, mt.my_gauss_seidel, mt.my_es_diagonal_dominante
# Apartado 2 - Regresión lineal
np_regresion_lineal, my_linear_regression, my_r2 = mt.np_regresion_lineal, mt.my_linear_regression, mt.my_r2
# Apartado 3 - Integración
np_trapecio, my_trapecio = mt.np_trapecio, mt.my_trapecio
my_trapecio_acumulado, my_simpson = mt.my_trapecio_acumulado, mt.my_simpson
# Apartado 4 - Interpolación
np_interp_lineal, np_polinomio_interpolante = mt.np_interp_lineal, mt.np_polinomio_interpolante
my_interp_lineal, my_vandermonde = mt.my_interp_lineal, mt.my_vandermonde
my_polyval, my_lagrange = mt.my_polyval, mt.my_lagrange
# Apartado 5 - Derivación
np_derivada, my_derivada_centrada = mt.np_derivada, mt.my_derivada_centrada
my_derivada_adelantada, my_derivada_retrasada = mt.my_derivada_adelantada, mt.my_derivada_retrasada
my_derivada_numerica = mt.my_derivada_numerica
# Apartado 6 - Ecuaciones no lineales
np_raiz_cambio_signo, my_newton, my_biseccion = mt.np_raiz_cambio_signo, mt.my_newton, mt.my_biseccion


# =============================================================================
# 1. CONSTANTES Y DATOS DE PARTIDA
# =============================================================================
G0 = 9.80665              # Gravedad estándar [m/s^2]
TOLERANCIA = 1e-10        # Tolerancia de los métodos iterativos (raíces y Jacobi/G-S)
V_TARGET_DEFECTO = 340.0  # Velocidad diana por defecto (≈ Mach 1 a nivel del mar) [m/s]
V_TARGET_MIN, V_TARGET_MAX = 300.0, 1000.0   # Rango del Slider [m/s]

# 1. Calibración del tubo Pitot: Voltaje [V] vs Presión [kPa]
V_sensor = np.array([0.5, 1.2, 2.3, 3.1, 4.0, 4.8])
P_sensor = np.array([10.1, 25.5, 48.0, 65.2, 83.9, 101.3])

# 2. Modelo atmosférico terrestre simplificado: Altitud [m] vs Densidad [kg/m^3]
altitudes_tab = np.array([0, 5000, 10000, 15000, 20000, 25000, 30000])
densidades_tab = np.array([1.225, 0.736, 0.413, 0.194, 0.088, 0.040, 0.018])

# 3. Telemetría de vuelo: tiempo [s] (muestreo a 1 Hz durante 2 min)
t_vuelo = np.linspace(0, 120, 121)

# Perfil de velocidad simulado [m/s] (modelo logístico)
v_vuelo = 2500 * (1 / (1 + np.exp(-0.05 * (t_vuelo - 60)))) - 118

# Sistema lineal del balance de masas de propelente:
#   m1 + m2 + m3 = 50000      (masa total)
#   m1 + m2 - m3 = 15000      (m1 + m2 supera a m3 en 15000 kg)
#   m1 - 2 m2    = 0          (m1 es el doble de m2)
A_masas = np.array([[1.0,  1.0,  1.0],
                    [1.0,  1.0, -1.0],
                    [1.0, -2.0,  0.0]])
b_masas = np.array([50000.0, 15000.0, 0.0])


# =============================================================================
# 2. MODELOS FÍSICOS ANALÍTICOS
#    v(t) es una logística, por lo que su derivada y su integral tienen
#    expresión cerrada. Se usan SOLO como referencia "exacta" para medir el
#    error de cada método numérico (criterio de exactitud de la rúbrica).
# =============================================================================
def velocidad_teorica(t):
    """v(t) = 2500 / (1 + e^{-0.05 (t-60)}) - 118   [m/s]"""
    return 2500.0 / (1.0 + np.exp(-0.05 * (t - 60.0))) - 118.0


def aceleracion_teorica(t):
    """a(t) = dv/dt = 2500·0.05·e^{-0.05(t-60)} / (1 + e^{-0.05(t-60)})^2   [m/s^2]"""
    e = np.exp(-0.05 * (t - 60.0))
    return 2500.0 * 0.05 * e / (1.0 + e) ** 2


def altitud_teorica(t):
    """h(t) = ∫_0^t v dτ = (2500/0.05)·[ln(1+e^{0.05(t-60)}) - ln(1+e^{-3})] - 118·t   [m]
    (np.logaddexp(0, x) = ln(1 + e^x) evaluado de forma numéricamente estable)"""
    return (2500.0 / 0.05) * (np.logaddexp(0.0, 0.05 * (t - 60.0))
                              - np.logaddexp(0.0, -3.0)) - 118.0 * t


# =============================================================================
# 3. APARTADOS (un def por apartado: numpy -> funciones my_)
# =============================================================================

# -----------------------------------------------------------------------------
# APARTADO 1: SISTEMA LINEAL - Balance de masas de propelente
# -----------------------------------------------------------------------------
def resolver_apartado_1(A, b):
    """Resuelve A·m = b con numpy y con los 4 métodos propios (Gauss, LU,
    Jacobi y Gauss-Seidel)."""
    # --- numpy ---------------------------------------------------------------
    m_numpy = np_sistema_lineal(A, b)

    # --- my_gauss (directo) --------------------------------------------------
    m_gauss = my_gauss(A, b)

    # --- my_lu (directo): P·A = L·U ------------------------------------------
    P, L, U = my_lu(A)
    m_lu = my_lu_solve(P, L, U, b)

    # --- my_jacobi y my_gauss_seidel (iterativos) ----------------------------
    # La matriz original tiene a33 = 0 y NO es diagonal dominante, así que los
    # métodos iterativos no pueden aplicarse directamente. Se construye un
    # sistema EQUIVALENTE combinando filas (T·A·m = T·b, con det(T) = 1 ≠ 0,
    # luego la solución es la misma):
    #   F1' = F1 + 0.5·F2 + 0.5·F3  -> [2.0,  0.5, 0.5]
    #   F2' = F3                    -> [1.0, -2.0, 0.0]
    #   F3' = F1 - 0.5·F2           -> [0.5,  0.5, 1.5]
    # que SÍ es estrictamente diagonal dominante -> convergencia garantizada.
    T = np.array([[1.0, 0.5, 0.5],
                  [0.0, 0.0, 1.0],
                  [1.0, -0.5, 0.0]])
    A_dd, b_dd = T @ A, T @ b
    m_jacobi, it_jacobi, hist_jacobi = my_jacobi(A_dd, b_dd, tol=TOLERANCIA)
    m_gs, it_gs, hist_gs = my_gauss_seidel(A_dd, b_dd, tol=TOLERANCIA)

    soluciones = {
        "np.linalg.solve": m_numpy,
        "my_gauss": m_gauss,
        "my_lu": m_lu,
        "my_jacobi": m_jacobi,
        "my_gauss_seidel": m_gs,
    }
    residuos = {k: np.linalg.norm(A @ x - b) for k, x in soluciones.items()}

    # Validación física: m = [21666.7, 10833.3, 17500.0] kg. Suman 50000 kg y
    # m1 + m2 = 32500 kg supera a m3 en 15000 kg -> cumple todas las condiciones.
    # Órdenes de magnitud de decenas de toneladas: típicos de una primera etapa.
    return {
        "soluciones": soluciones, "residuos": residuos,
        "P": P, "L": L, "U": U, "lu_correcta": np.allclose(P @ A, L @ U),
        "A_dd": A_dd, "b_dd": b_dd,
        "diag_dominante_original": my_es_diagonal_dominante(A),
        "diag_dominante_equivalente": my_es_diagonal_dominante(A_dd),
        "it_jacobi": it_jacobi, "hist_jacobi": hist_jacobi,
        "it_gs": it_gs, "hist_gs": hist_gs,
        "masas": m_numpy,
    }


# -----------------------------------------------------------------------------
# APARTADO 2: REGRESIÓN LINEAL - Calibración del sensor de presión dinámica
# -----------------------------------------------------------------------------
def resolver_apartado_2(V, P):
    """Ajusta P(V) = θ1·V + θ0 por mínimos cuadrados con numpy y con my_."""
    # --- numpy ---------------------------------------------------------------
    theta1_np, theta0_np = np_regresion_lineal(V, P)

    # --- my_linear_regression (ecuación normal resuelta con my_gauss) --------
    theta1_my, theta0_my = my_linear_regression(V, P)

    P_ajuste = theta1_my * V + theta0_my
    r2 = my_r2(P, P_ajuste)

    # Validación física: sensibilidad ≈ 21.1 kPa/V y offset ≈ -0.28 kPa (casi
    # nulo, como debe ser: a 0 V el transductor no mide presión). Con R² ≈ 0.9999
    # el sensor es lineal en todo el rango 0-5 V. Fondo de escala ≈ 105 kPa.
    return {
        "theta1_np": theta1_np, "theta0_np": theta0_np,
        "theta1_my": theta1_my, "theta0_my": theta0_my,
        "r2": r2, "residuos": P - P_ajuste,
        "dif_max": max(abs(theta1_np - theta1_my), abs(theta0_np - theta0_my)),
    }


# -----------------------------------------------------------------------------
# APARTADO 3: INTEGRACIÓN NUMÉRICA - Reconstrucción de la altitud h(t)
# -----------------------------------------------------------------------------
def resolver_apartado_3(t, v):
    """h(t_k) = ∫_0^{t_k} v dτ. Valor final con numpy, my_trapecio y my_simpson,
    y perfil completo h(t) con el trapecio acumulado."""
    # --- numpy ---------------------------------------------------------------
    h_final_numpy = np_trapecio(v, t)

    # --- my_trapecio y my_simpson (altitud final a t = 120 s) ----------------
    h_final_trapecio = my_trapecio(v, t)
    h_final_simpson = my_simpson(v, t)        # 120 subintervalos (par) -> válido

    # --- my_trapecio_acumulado (perfil h(t) completo, se usa en el Apartado 4)
    h_perfil = my_trapecio_acumulado(v, t)

    h_exacta = altitud_teorica(t)
    # Validación física: h(120 s) ≈ 135.8 km. Un lanzador pesado supera la
    # línea de Kármán (100 km) hacia los 2 min de vuelo -> orden de magnitud
    # coherente. Nota: el error al final es ~0 porque v'(0) = v'(120) (la
    # logística es simétrica respecto a t = 60 s) y los términos de error de
    # Euler-Maclaurin se cancelan; en instantes intermedios el error sí aparece.
    return {
        "h_perfil": h_perfil, "h_exacta": h_exacta,
        "h_final": {"np.trapezoid": h_final_numpy,
                    "my_trapecio": h_final_trapecio,
                    "my_simpson": h_final_simpson},
        "h_final_exacta": h_exacta[-1],
        "error_perfil_max": np.max(np.abs(h_perfil - h_exacta)),
    }


# -----------------------------------------------------------------------------
# APARTADO 4: INTERPOLACIÓN 1D - Densidad atmosférica ρ(t)
# -----------------------------------------------------------------------------
def resolver_apartado_4(h_vuelo, h_tab, rho_tab, t):
    """ρ(t) a partir de h(t) del Apartado 3 y de la tabla atmosférica.
    Se trabaja en km para que la matriz de Vandermonde esté mejor condicionada."""
    h_km = h_vuelo / 1000.0
    nodos_km = h_tab / 1000.0
    dentro_tabla = h_km <= nodos_km[-1]

    # --- numpy: lineal (np.interp) y polinomio de Lagrange (np.polyfit grado n-1)
    rho_np_lineal = np_interp_lineal(h_km, nodos_km, rho_tab)
    rho_np_lagrange = np.where(dentro_tabla,
                               np_polinomio_interpolante(h_km, nodos_km, rho_tab), np.nan)

    # --- my_interp_lineal (método pedido por el enunciado) -------------------
    rho_my_lineal = my_interp_lineal(h_km, nodos_km, rho_tab)

    # --- my_vandermonde y my_lagrange (polinomio interpolador de grado 6) ----
    coefs_vander = my_vandermonde(nodos_km, rho_tab)
    rho_vander = np.where(dentro_tabla, my_polyval(coefs_vander, h_km), np.nan)
    rho_lagrange = np.where(dentro_tabla, my_lagrange(h_km, nodos_km, rho_tab), np.nan)

    # Curvas densas sobre la tabla (para comparar los interpolantes en altitud)
    h_denso = np.linspace(nodos_km[0], nodos_km[-1], 301)
    curvas_altitud = {
        "lineal": my_interp_lineal(h_denso, nodos_km, rho_tab),
        "polinomio": my_lagrange(h_denso, nodos_km, rho_tab),
    }

    # Instante en que se abandona la tabla (h = 30 km). h(t) es monótona
    # creciente (v > 0), así que se puede invertir con interpolación lineal.
    t_fuera_tabla = np_interp_lineal(nodos_km[-1], h_km, t)

    # Validación física: ρ cae de 1.225 kg/m³ (nivel del mar) a 0.018 kg/m³ a
    # 30 km (≈ 1.5 %), coherente con una atmósfera exponencial (H ≈ 6-7 km).
    # El polinomio de grado 6 baja a ≈ 0.0143 kg/m³ hacia 29 km, POR DEBAJO del
    # último nodo: es una oscilación de tipo Runge -> se elige la interpolación
    # LINEAL (la que pide el enunciado) como resultado principal.
    # Por encima de 30 km la tabla no tiene datos: los métodos lineales
    # mantienen el último valor y los polinomios no se evalúan (extrapolar un
    # polinomio de grado 6 daría valores sin sentido físico).
    i60 = int(np.argmin(np.abs(t - 60.0)))
    return {
        "h_km": h_km, "dentro_tabla": dentro_tabla, "t_fuera_tabla": t_fuera_tabla,
        "rho": rho_my_lineal,
        "rho_np_lineal": rho_np_lineal, "rho_np_lagrange": rho_np_lagrange,
        "rho_my_lineal": rho_my_lineal, "rho_vander": rho_vander, "rho_lagrange": rho_lagrange,
        "h_denso": h_denso, "curvas_altitud": curvas_altitud,
        "dif_lineal": np.max(np.abs(rho_np_lineal - rho_my_lineal)),
        "dif_polinomios": max(np.nanmax(np.abs(rho_vander - rho_lagrange)),
                              np.nanmax(np.abs(rho_np_lagrange - rho_lagrange))),
        "min_polinomio": np.min(curvas_altitud["polinomio"]),
        "i60": i60,
        "rho_t60": {"np.interp": rho_np_lineal[i60], "my_interp_lineal": rho_my_lineal[i60],
                    "np.polyfit (Lagrange)": rho_np_lagrange[i60],
                    "my_vandermonde": rho_vander[i60], "my_lagrange": rho_lagrange[i60]},
    }


# -----------------------------------------------------------------------------
# APARTADO 5: DERIVACIÓN NUMÉRICA - Aceleración estructural a(t)
# -----------------------------------------------------------------------------
def resolver_apartado_5(t, v):
    """a(t) = dv/dt con np.gradient y con las fórmulas propias de 3 puntos."""
    # --- numpy (edge_order=2 -> extremos de 3 puntos, mismo orden que my_) ---
    a_numpy = np_derivada(v, t, edge_order=2)

    # --- my_ : centradas, adelantadas y retrasadas de 3 puntos ---------------
    a_centrada = my_derivada_centrada(v, t)       # nodos interiores
    a_adelantada = my_derivada_adelantada(v, t)   # todos menos los 2 últimos
    a_retrasada = my_derivada_retrasada(v, t)     # todos menos los 2 primeros
    a_my = my_derivada_numerica(v, t)             # combinación: perfil completo

    # --- Esquema literal del enunciado: centradas en el interior y
    #     adelantada/atrasada de PRIMER orden (2 puntos) en los extremos.
    #     np.gradient con edge_order=1 hace exactamente esto.
    a_numpy_1 = np_derivada(v, t, edge_order=1)
    a_my_1 = my_derivada_numerica(v, t, orden_extremos=1)

    a_exacta = aceleracion_teorica(t)
    errores = {
        "np.gradient": np.max(np.abs(a_numpy - a_exacta)),
        "np.gradient (1er ord)": np.max(np.abs(a_numpy_1 - a_exacta)),
        "my_numerica (1er ord)": np.max(np.abs(a_my_1 - a_exacta)),
        "my_centrada": np.nanmax(np.abs(a_centrada - a_exacta)),
        "my_adelantada": np.nanmax(np.abs(a_adelantada - a_exacta)),
        "my_retrasada": np.nanmax(np.abs(a_retrasada - a_exacta)),
        "my_derivada_numerica": np.max(np.abs(a_my - a_exacta)),
    }
    i_max = int(np.argmax(a_my))

    # Validación física: a_max ≈ 31.2 m/s² ≈ 3.2 g en t = 60 s (punto de
    # inflexión de la logística). Un lanzador tripulado/carga útil sensible se
    # limita a ~4-5 g -> la carga de pago está dentro de la envolvente segura.
    return {
        "a": a_my, "a_numpy": a_numpy, "a_centrada": a_centrada,
        "a_adelantada": a_adelantada, "a_retrasada": a_retrasada, "a_exacta": a_exacta,
        "errores": errores, "a_max": a_my[i_max], "t_a_max": t[i_max],
        "a_inicial": a_my[0], "a_final": a_my[-1],
        "dif_my_numpy": np.max(np.abs(a_my - a_numpy)),
        "dif_1er_orden": np.max(np.abs(a_my_1 - a_numpy_1)),
    }


# -----------------------------------------------------------------------------
# APARTADO 6: ECUACIÓN NO LINEAL - Instante crítico t_crit con v(t) = v_target
# -----------------------------------------------------------------------------
def resolver_apartado_6(v_target):
    """Resuelve f(t) = v(t) - v_target = 0 con numpy, my_newton y my_biseccion.

    Justificación de los puntos iniciales:
      - Bisección en [0, 120] s: f(0) = 0.57 - v_target < 0 y
        f(120) = 2263 - v_target > 0 para todo v_target del Slider (300-1000 m/s)
        -> por Bolzano hay raíz, y es única porque v(t) es estrictamente creciente.
      - Newton con t0 = 60 s: es el punto de inflexión de la logística, donde
        f'(t) es máxima (31.25 m/s²) -> el primer paso nunca diverge y, al quedar
        la raíz en la zona convexa (t < 60), las iteraciones se acercan de forma
        monótona.
    """
    f = lambda t: velocidad_teorica(t) - v_target
    df = aceleracion_teorica

    # --- numpy: cambio de signo sobre un vector denso + interpolación lineal --
    t_numpy = np_raiz_cambio_signo(f, 0.0, 120.0)

    # --- my_newton -----------------------------------------------------------
    t_newton, it_newton = my_newton(f, x0=60.0, df=df, tol=TOLERANCIA)

    # --- my_biseccion --------------------------------------------------------
    t_biseccion, it_biseccion = my_biseccion(f, 0.0, 120.0, tol=TOLERANCIA)

    # Solución analítica (la logística se puede invertir) -> referencia exacta
    t_exacto = 60.0 - np.log(2500.0 / (v_target + 118.0) - 1.0) / 0.05

    # Validación física: para 340 m/s (Mach 1) t_crit ≈ 30.1 s. Los lanzadores
    # reales rompen la barrera del sonido entre 30 y 60 s -> resultado coherente.
    return {
        "v_target": v_target, "t_exacto": t_exacto,
        "raices": {"numpy (cambio signo)": t_numpy,
                   "my_newton": t_newton, "my_biseccion": t_biseccion},
        "iteraciones": {"numpy (cambio signo)": None,
                        "my_newton": it_newton, "my_biseccion": it_biseccion},
        "t_crit": t_newton,
    }


# -----------------------------------------------------------------------------
# EXTRA: PRESIÓN DINÁMICA MÁXIMA (Max Q)   q(t) = ½ ρ(t) v(t)²
# -----------------------------------------------------------------------------
def resolver_extra_max_q(h_km, v, t, h_tab, rho_tab):
    """Combina los Apartados 3 y 4 para localizar el instante de máxima carga
    aerodinámica. Para que q(t) tenga sentido también por encima de 30 km, se
    interpola ln(ρ) linealmente (atmósfera exponencial) y se extrapola con la
    altura de escala H del último tramo de la tabla."""
    nodos_km = h_tab / 1000.0
    ln_rho = np.log(rho_tab)
    H = (nodos_km[-1] - nodos_km[-2]) / (ln_rho[-2] - ln_rho[-1])   # altura de escala [km]
    ln_rho_t = np.where(h_km <= nodos_km[-1],
                        my_interp_lineal(h_km, nodos_km, ln_rho),
                        ln_rho[-1] - (h_km - nodos_km[-1]) / H)
    rho_t = np.exp(ln_rho_t)
    q = 0.5 * rho_t * v ** 2 / 1000.0                                # [kPa]
    i = int(np.argmax(q))
    # Validación física: Max Q ≈ 79 kPa hacia t ≈ 43 s y h ≈ 10 km. Los
    # lanzadores reales alcanzan Max Q entre 10 y 15 km de altitud -> coherente.
    return {"q": q, "q_max": q[i], "t_max_q": t[i], "h_max_q": h_km[i], "H": H}


def calcular_todo():
    """Ejecuta los apartados en orden. El 4 usa h(t) del 3 (encadenamiento)."""
    r1 = resolver_apartado_1(A_masas, b_masas)
    r2 = resolver_apartado_2(V_sensor, P_sensor)
    r3 = resolver_apartado_3(t_vuelo, v_vuelo)
    r4 = resolver_apartado_4(r3["h_perfil"], altitudes_tab, densidades_tab, t_vuelo)
    r5 = resolver_apartado_5(t_vuelo, v_vuelo)
    r6 = resolver_apartado_6(V_TARGET_DEFECTO)
    rx = resolver_extra_max_q(r4["h_km"], v_vuelo, t_vuelo, altitudes_tab, densidades_tab)
    return {1: r1, 2: r2, 3: r3, 4: r4, 5: r5, 6: r6, 7: rx}


# =============================================================================
# 4. INFORME POR CONSOLA (resultados con unidades SI + interpretación física)
# =============================================================================
def _cabecera(texto):
    print("\n" + "=" * 78 + f"\n {texto}\n" + "=" * 78)


def informe_apartado_1(r):
    _cabecera("APARTADO 1 - SISTEMA LINEAL: Balance de masas de propelente")
    for metodo, m in r["soluciones"].items():
        print(f"  {metodo:<17} m1 = {m[0]:10.2f} kg | m2 = {m[1]:10.2f} kg | "
              f"m3 = {m[2]:10.2f} kg | ‖A·m-b‖ = {r['residuos'][metodo]:.1e}")
    print(f"  LU con pivoteo: P·A = L·U verificado -> {r['lu_correcta']}")
    print(f"  Matriz original diagonal dominante: {r['diag_dominante_original']} "
          f"-> se usa un sistema equivalente (dominante: {r['diag_dominante_equivalente']})")
    print(f"  Iteraciones hasta tol = {TOLERANCIA:.0e}:  Jacobi = {r['it_jacobi']} | "
          f"Gauss-Seidel = {r['it_gs']}")
    m = r["masas"]
    print(f"  -> Interpretación: el tanque 1 alberga {m[0]:.0f} kg, el doble que el 2 "
          f"({m[1]:.0f} kg), y m1+m2 = {m[0]+m[1]:.0f} kg supera a m3 = {m[2]:.0f} kg en "
          f"15000 kg: el centro de masas queda adelantado, lo que da estabilidad en ascenso.")


def informe_apartado_2(r):
    _cabecera("APARTADO 2 - REGRESIÓN LINEAL: Calibración del sensor de presión")
    print(f"  np.polyfit             P(V) = {r['theta1_np']:.4f}·V + ({r['theta0_np']:.4f}) kPa")
    print(f"  my_linear_regression   P(V) = {r['theta1_my']:.4f}·V + ({r['theta0_my']:.4f}) kPa")
    print(f"  Diferencia máxima entre métodos: {r['dif_max']:.1e} | R² = {r['r2']:.6f}")
    print(f"  -> Interpretación: cada voltio equivale a {r['theta1_my']:.2f} kPa de presión "
          f"dinámica; el offset ({r['theta0_my']:.2f} kPa) es casi nulo y R² ≈ 1 confirma "
          f"que el transductor es lineal en todo su rango operativo (0-5 V).")


def informe_apartado_3(r):
    _cabecera("APARTADO 3 - INTEGRACIÓN NUMÉRICA: Reconstrucción de la altitud")
    for metodo, h in r["h_final"].items():
        print(f"  {metodo:<14} h(120 s) = {h:12.4f} m   (error = {abs(h - r['h_final_exacta']):.1e} m)")
    print(f"  Valor exacto (analítico) h(120 s) = {r['h_final_exacta']:.4f} m")
    print(f"  Error máximo del perfil acumulado h(t): {r['error_perfil_max']:.3f} m")
    print(f"  -> Interpretación: a los 2 minutos el vehículo está a "
          f"{r['h_final_exacta']/1000:.1f} km, por encima de la línea de Kármán (100 km): "
          f"la primera etapa ha sacado a la sonda de la atmósfera densa.")


def informe_apartado_4(r):
    _cabecera("APARTADO 4 - INTERPOLACIÓN 1D: Densidad atmosférica ρ(t)")
    print(f"  Densidad a t = 60 s (h = {r['h_km'][r['i60']]:.2f} km):")
    for metodo, rho in r["rho_t60"].items():
        print(f"    {metodo:<22} ρ = {rho:.5f} kg/m³")
    print(f"  Diferencia np.interp vs my_interp_lineal: {r['dif_lineal']:.1e} kg/m³")
    print(f"  Diferencia entre polinomios (Vandermonde/Lagrange/polyfit): {r['dif_polinomios']:.1e} kg/m³")
    print(f"  Mínimo del polinomio de grado 6 en [0, 30] km: {r['min_polinomio']:.4f} kg/m³ "
          f"(< 0.018 del último nodo -> oscilación de Runge)")
    print(f"  -> Interpretación: la densidad cae un {100*(1-r['rho'][r['i60']]/1.225):.0f} % "
          f"en el primer minuto; a partir de t = {r['t_fuera_tabla']:.1f} s el vehículo supera "
          f"los 30 km y abandona el modelo tabulado (atmósfera prácticamente despreciable).")


def informe_apartado_5(r):
    _cabecera("APARTADO 5 - DERIVACIÓN NUMÉRICA: Aceleración estructural")
    for metodo, err in r["errores"].items():
        print(f"  {metodo:<24} error máximo frente a la exacta = {err:.2e} m/s²")
    print(f"  Esquema del enunciado (extremos de 1er orden): my_ vs np.gradient -> dif = {r['dif_1er_orden']:.1e} m/s²")
    print(f"  a(0) = {r['a_inicial']:.3f} m/s² | a(120) = {r['a_final']:.3f} m/s²")
    print(f"  Aceleración máxima: {r['a_max']:.2f} m/s² = {r['a_max']/G0:.2f} g en t = {r['t_a_max']:.0f} s")
    print(f"  -> Interpretación: el pico de {r['a_max']/G0:.2f} g está por debajo del límite "
          f"habitual de 4-5 g para cargas útiles sensibles: la sonda no sufre daño estructural.")


def informe_apartado_6(r):
    _cabecera("APARTADO 6 - ECUACIÓN NO LINEAL: Instante crítico (Mach target)")
    print(f"  v_target = {r['v_target']:.1f} m/s  |  t exacto (analítico) = {r['t_exacto']:.10f} s")
    for metodo, t in r["raices"].items():
        it = r["iteraciones"][metodo]
        texto_it = f"{it:3d} iteraciones" if it is not None else "vectorizado    "
        print(f"  {metodo:<21} t_crit = {t:.10f} s | {texto_it} | error = {abs(t - r['t_exacto']):.1e} s")
    print(f"  -> Interpretación: el vehículo rompe la barrera del sonido a los "
          f"{r['t_crit']:.2f} s del despegue; Newton converge cuadráticamente "
          f"({r['iteraciones']['my_newton']} iteraciones) frente a la convergencia lineal "
          f"de la bisección ({r['iteraciones']['my_biseccion']} iteraciones).")


def informe_extra(r):
    _cabecera("EXTRA - PRESIÓN DINÁMICA MÁXIMA (Max Q)")
    print(f"  Max Q = {r['q_max']:.2f} kPa en t = {r['t_max_q']:.0f} s (h = {r['h_max_q']:.2f} km)")
    print(f"  -> Interpretación: es el instante de máximo esfuerzo aerodinámico sobre el "
          f"fuselaje; los lanzadores reducen empuje alrededor de este punto.")


def imprimir_informe(res):
    informe_apartado_1(res[1])
    informe_apartado_2(res[2])
    informe_apartado_3(res[3])
    informe_apartado_4(res[4])
    informe_apartado_5(res[5])
    informe_apartado_6(res[6])
    informe_extra(res[7])
    print()
