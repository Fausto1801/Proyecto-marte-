"""
=============================================================================
 main.py
 ANÁLISIS TELEMÉTRICO Y AERODINÁMICO DEL LANZAMIENTO DE UNA SONDA A MARTE
 Vehículo de lanzamiento Ares-I  -  Dashboard interactivo de telemetría
 Grado en Ingeniería Aeroespacial - Programación
=============================================================================
 Estructura del fichero (mismo orden en todo el código):

   0. Importaciones y constantes
   1. Datos de partida (enunciado)
   2. Modelos físicos analíticos (solo para validar la exactitud)
   3. Cálculos numéricos - un def por apartado, siempre en el mismo orden:
        numpy  ->  funciones my_ de my_numerical_lib.py
        Apartado 1  Sistema lineal ........ np.linalg.solve | my_gauss | my_lu | my_jacobi | my_gauss_seidel
        Apartado 2  Regresión lineal ...... np.polyfit | my_linear_regression
        Apartado 3  Integración ........... np.trapezoid | my_trapecio | my_simpson
        Apartado 4  Interpolación ......... np.interp + np.polyfit (Lagrange) | my_interp_lineal | my_vandermonde | my_lagrange
        Apartado 5  Derivación ............ np.gradient | my_ centradas, adelantadas, retrasadas (3 puntos)
        Apartado 6  Ecuación no lineal .... numpy (cambio de signo vectorizado) | my_newton | my_biseccion
        Extra       Presión dinámica máxima (Max Q)
   4. Informe por consola con interpretación física
   5. Interfaz gráfica (matplotlib.widgets): RadioButtons + Slider + texto
   6. Programa principal

 Ejecución:   python main.py     (my_numerical_lib.py en la misma carpeta)
 Requisitos:  numpy >= 1.20, matplotlib >= 3.5
=============================================================================
"""

# =============================================================================
# 0. IMPORTACIONES Y CONSTANTES
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

from my_numerical_lib import (
    # Apartado 1 - Sistemas lineales
    my_gauss, my_lu, my_lu_solve, my_jacobi, my_gauss_seidel, my_es_diagonal_dominante,
    # Apartado 2 - Regresión lineal
    my_linear_regression, my_r2,
    # Apartado 3 - Integración
    my_trapecio, my_trapecio_acumulado, my_simpson,
    # Apartado 4 - Interpolación
    my_interp_lineal, my_vandermonde, my_polyval, my_lagrange,
    # Apartado 5 - Derivación
    my_derivada_centrada, my_derivada_adelantada, my_derivada_retrasada, my_derivada_numerica,
    # Apartado 6 - Ecuaciones no lineales
    my_newton, my_biseccion,
)

# np.trapz pasó a llamarse np.trapezoid en numpy 2.0 -> compatibilidad con ambas versiones
np_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

G0 = 9.80665              # Gravedad estándar [m/s^2]
TOLERANCIA = 1e-10        # Tolerancia de los métodos iterativos (raíces y Jacobi/G-S)
V_TARGET_DEFECTO = 340.0  # Velocidad diana por defecto (≈ Mach 1 a nivel del mar) [m/s]
V_TARGET_MIN, V_TARGET_MAX = 300.0, 1000.0   # Rango del Slider [m/s]


# =============================================================================
# 1. DATOS DE PARTIDA (copiados literalmente del enunciado)
# =============================================================================
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
# 3. CÁLCULOS NUMÉRICOS (un def por apartado)
# =============================================================================

# -----------------------------------------------------------------------------
# APARTADO 1: SISTEMA LINEAL - Balance de masas de propelente
# -----------------------------------------------------------------------------
def resolver_apartado_1(A, b):
    """Resuelve A·m = b con numpy y con los 4 métodos propios (Gauss, LU,
    Jacobi y Gauss-Seidel)."""
    # --- numpy ---------------------------------------------------------------
    m_numpy = np.linalg.solve(A, b)

    # --- my_gauss (directo) --------------------------------------------------
    m_gauss = my_gauss(A, b)

    # --- my_lu (directo): P·A = L·U --------------------------------------------
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
    theta1_np, theta0_np = np.polyfit(V, P, 1)

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
    """h(t_k) = ∫_0^{t_k} v dτ. Perfil completo con trapecio acumulado y valor
    final con np.trapezoid, my_trapecio y my_simpson."""
    # --- numpy ---------------------------------------------------------------
    h_final_numpy = np_trapezoid(v, t)

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
    rho_np_lineal = np.interp(h_km, nodos_km, rho_tab)
    coefs_np = np.polyfit(nodos_km, rho_tab, len(nodos_km) - 1)
    rho_np_lagrange = np.where(dentro_tabla, np.polyval(coefs_np, h_km), np.nan)

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
    t_fuera_tabla = np.interp(nodos_km[-1], h_km, t)

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
    a_numpy = np.gradient(v, t, edge_order=2)

    # --- my_ : centradas, adelantadas y retrasadas de 3 puntos ---------------
    a_centrada = my_derivada_centrada(v, t)       # nodos interiores
    a_adelantada = my_derivada_adelantada(v, t)   # todos menos los 2 últimos
    a_retrasada = my_derivada_retrasada(v, t)     # todos menos los 2 primeros
    a_my = my_derivada_numerica(v, t)             # combinación: perfil completo

    # --- Esquema literal del enunciado: centradas en el interior y
    #     adelantada/atrasada de PRIMER orden (2 puntos) en los extremos.
    #     np.gradient con edge_order=1 (su valor por defecto) hace exactamente esto.
    a_numpy_1 = np.gradient(v, t, edge_order=1)
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
    t_denso = np.linspace(0.0, 120.0, 12001)
    f_denso = f(t_denso)
    i = np.nonzero(np.diff(np.sign(f_denso)))[0][0]
    t_numpy = t_denso[i] - f_denso[i] * (t_denso[i + 1] - t_denso[i]) / (f_denso[i + 1] - f_denso[i])

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


# =============================================================================
# 5. INTERFAZ GRÁFICA (matplotlib + matplotlib.widgets)
# =============================================================================
# Colores fijos por método (mismo color = mismo método en todas las vistas)
COLOR = {"numpy": "#222222", "my1": "#1f77b4", "my2": "#d62728", "my3": "#2ca02c",
         "my4": "#ff7f0e", "exacta": "#9e9e9e", "datos": "#d62728"}

OPCIONES = (
    "1. Sistema lineal",
    "2. Regresión lineal",
    "3. Integración",
    "4. Interpolación",
    "5. Derivación",
    "6. Ecuación no lineal",
    "Extra: Max Q",
)

# Zona de gráficos (en coordenadas de figura): [izq, abajo, ancho, alto]
ZONA_GRAFICOS = [0.29, 0.33, 0.68, 0.58]
SEPARACION = 0.07


def _estilo(ax, titulo, xlabel, ylabel, leyenda=True):
    """Formato científico común: título, etiquetas con unidades, rejilla y leyenda."""
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    if leyenda:
        ax.legend(fontsize=8, loc="best")


def _colocar_ejes(ax1, ax2, dos_paneles):
    """Una vista ocupa toda la zona de gráficos o la divide en dos paneles."""
    x, y, w, h = ZONA_GRAFICOS
    ax1.clear()
    ax2.clear()
    if dos_paneles:
        mitad = (w - SEPARACION) / 2
        ax1.set_position([x, y, mitad, h])
        ax2.set_position([x + mitad + SEPARACION, y, mitad, h])
        ax2.set_visible(True)
    else:
        ax1.set_position([x, y, w, h])
        ax2.set_visible(False)


# ---------------------------- vistas por apartado ----------------------------
def vista_apartado_1(ax1, ax2, r):
    _colocar_ejes(ax1, ax2, dos_paneles=True)
    metodos = list(r["soluciones"].keys())
    colores = [COLOR["numpy"], COLOR["my1"], COLOR["my2"], COLOR["my3"], COLOR["my4"]]
    x = np.arange(3)
    ancho = 0.16
    for k, (metodo, color) in enumerate(zip(metodos, colores)):
        ax1.bar(x + (k - 2) * ancho, r["soluciones"][metodo] / 1000.0, ancho,
                color=color, label=metodo, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x, ["Tanque 1 ($m_1$)", "Tanque 2 ($m_2$)", "Tanque 3 ($m_3$)"])
    ax1.set_ylim(0, 27)
    _estilo(ax1, "Masa de propelente por tanque", "", "Masa [t]")
    ax1.legend(fontsize=7.5, loc="upper right", ncol=2)

    ax2.semilogy(np.arange(1, r["it_jacobi"] + 1), r["hist_jacobi"], "o-", ms=3,
                 color=COLOR["my3"], label=f"my_jacobi ({r['it_jacobi']} it.)")
    ax2.semilogy(np.arange(1, r["it_gs"] + 1), r["hist_gs"], "s-", ms=3,
                 color=COLOR["my4"], label=f"my_gauss_seidel ({r['it_gs']} it.)")
    ax2.axhline(TOLERANCIA, color=COLOR["exacta"], ls=":", label=f"tolerancia {TOLERANCIA:.0e}")
    _estilo(ax2, "Convergencia de los métodos iterativos", "Iteración",
            r"Residuo relativo $\|b-Am\|/\|b\|$")

    m = r["masas"]
    filas = "\n".join(f"  {k:<16} m1={v[0]:9.2f}  m2={v[1]:9.2f}  m3={v[2]:9.2f} kg   ‖Am-b‖={r['residuos'][k]:.0e}"
                      for k, v in r["soluciones"].items())
    return ("APARTADO 1 · Sistema lineal A·m = b (balance de masas)\n" + filas +
            f"\nInterpretación: m1 = 2·m2 y m1+m2 = {m[0]+m[1]:.0f} kg supera a m3 en 15000 kg -> "
            "centro de masas adelantado y vuelo estable.\nJacobi/G-S sobre un sistema equivalente "
            "diagonal dominante (la matriz original tiene a33 = 0).")


def vista_apartado_2(ax1, ax2, r):
    _colocar_ejes(ax1, ax2, dos_paneles=False)
    V_fino = np.linspace(0, 5.2, 100)
    ax1.plot(V_sensor, P_sensor, "o", ms=8, color=COLOR["datos"], label="Datos de laboratorio", zorder=3)
    ax1.plot(V_fino, r["theta1_np"] * V_fino + r["theta0_np"], "-", lw=3, alpha=0.35,
             color=COLOR["numpy"], label="np.polyfit")
    ax1.plot(V_fino, r["theta1_my"] * V_fino + r["theta0_my"], "--", lw=1.8,
             color=COLOR["my1"], label="my_linear_regression")
    _estilo(ax1, "Calibración del sensor de presión dinámica  P(V) = θ₁·V + θ₀",
            "Voltaje del transductor V [V]", "Presión P [kPa]")
    return ("APARTADO 2 · Regresión lineal por mínimos cuadrados\n"
            f"  np.polyfit             θ1 = {r['theta1_np']:.4f} kPa/V   θ0 = {r['theta0_np']:.4f} kPa\n"
            f"  my_linear_regression   θ1 = {r['theta1_my']:.4f} kPa/V   θ0 = {r['theta0_my']:.4f} kPa\n"
            f"  R² = {r['r2']:.6f}   |   diferencia entre métodos = {r['dif_max']:.1e}\n"
            f"Interpretación: sensibilidad de {r['theta1_my']:.2f} kPa por voltio y offset casi nulo: "
            "el transductor es lineal en todo su rango.")


def vista_apartado_3(ax1, ax2, r):
    _colocar_ejes(ax1, ax2, dos_paneles=True)
    ax1.plot(t_vuelo, v_vuelo, color=COLOR["my1"], lw=2, label="v(t) telemetría")
    ax1.fill_between(t_vuelo, 0, v_vuelo, color=COLOR["my1"], alpha=0.15, label=r"$\int v\,dt$ = altitud")
    _estilo(ax1, "Integrando: velocidad de vuelo", "Tiempo t [s]", "Velocidad v [m/s]")

    ax2.plot(t_vuelo, r["h_exacta"] / 1000, lw=4, alpha=0.35, color=COLOR["exacta"], label="h(t) exacta")
    ax2.plot(t_vuelo, r["h_perfil"] / 1000, "--", lw=1.8, color=COLOR["my1"], label="my_trapecio_acumulado")
    marcadores = {"np.trapezoid": ("o", COLOR["numpy"], 11),
                  "my_trapecio": ("s", COLOR["my2"], 8),
                  "my_simpson": ("^", COLOR["my3"], 6)}
    for metodo, h in r["h_final"].items():
        mk, c, s = marcadores[metodo]
        ax2.plot(120, h / 1000, mk, color=c, ms=s, mfc="none" if mk == "o" else c, label=f"{metodo} (t=120 s)")
    ax2.axhline(100, color=COLOR["datos"], ls=":", lw=1, label="Línea de Kármán (100 km)")
    _estilo(ax2, "Altitud reconstruida h(t)", "Tiempo t [s]", "Altitud h [km]")

    filas = "\n".join(f"  {k:<13} h(120 s) = {v:12.4f} m   error = {abs(v - r['h_final_exacta']):.1e} m"
                      for k, v in r["h_final"].items())
    return ("APARTADO 3 · Integración numérica  h(t) = ∫ v(t) dt\n" + filas +
            f"\n  Exacta = {r['h_final_exacta']:.4f} m  |  error máx. del perfil h(t) = {r['error_perfil_max']:.2f} m\n"
            f"Interpretación: a los 120 s el vehículo está a {r['h_final_exacta']/1000:.1f} km, "
            "por encima de la línea de Kármán.")


def vista_apartado_4(ax1, ax2, r):
    _colocar_ejes(ax1, ax2, dos_paneles=True)
    nodos_km = altitudes_tab / 1000
    ax1.plot(r["h_denso"], r["curvas_altitud"]["polinomio"], "-", color=COLOR["my2"], lw=1.8,
             label="Polinomio grado 6 (Lagrange/Vandermonde)")
    ax1.plot(r["h_denso"], r["curvas_altitud"]["lineal"], "-", color=COLOR["my1"], lw=1.8,
             label="Lineal a trozos")
    ax1.plot(nodos_km, densidades_tab, "ko", ms=6, label="Tabla atmosférica", zorder=3)
    _estilo(ax1, "Interpolantes sobre la tabla", "Altitud h [km]", r"Densidad ρ [kg/m$^3$]")

    ax2.plot(t_vuelo, r["rho_np_lineal"], lw=4, alpha=0.3, color=COLOR["numpy"], label="np.interp")
    ax2.plot(t_vuelo, r["rho_my_lineal"], "-", lw=1.6, color=COLOR["my1"], label="my_interp_lineal")
    ax2.plot(t_vuelo, r["rho_np_lagrange"], "o", ms=6, mfc="none", color=COLOR["numpy"], label="np.polyfit (Lagrange)")
    ax2.plot(t_vuelo, r["rho_vander"], "s", ms=3.5, color=COLOR["my2"], label="my_vandermonde")
    ax2.plot(t_vuelo, r["rho_lagrange"], "x", ms=4, color=COLOR["my3"], label="my_lagrange")
    ax2.axvspan(r["t_fuera_tabla"], t_vuelo[-1], color=COLOR["exacta"], alpha=0.15,
                label=f"h > 30 km (fuera de tabla)")
    _estilo(ax2, "Densidad a lo largo del vuelo ρ(t)", "Tiempo t [s]", r"Densidad ρ [kg/m$^3$]")

    i = r["i60"]
    filas = "   ".join(f"{k}: {v:.5f}" for k, v in r["rho_t60"].items())
    return (f"APARTADO 4 · Interpolación 1D   ρ en t = 60 s (h = {r['h_km'][i]:.2f} km)  [kg/m³]\n"
            f"  {filas}\n"
            f"  np.interp vs my_interp_lineal: dif = {r['dif_lineal']:.0e}  |  Vandermonde vs Lagrange vs polyfit: dif = {r['dif_polinomios']:.0e}\n"
            f"Interpretación: el polinomio de grado 6 cae a {r['min_polinomio']:.4f} kg/m³ cerca de 29 km (oscilación de Runge),\n"
            f"por eso se usa la interpolación lineal. Desde t = {r['t_fuera_tabla']:.1f} s el vehículo supera los 30 km.")


def vista_apartado_5(ax1, ax2, r):
    _colocar_ejes(ax1, ax2, dos_paneles=True)
    ax1.plot(t_vuelo, r["a_exacta"], lw=4, alpha=0.35, color=COLOR["exacta"], label="a(t) exacta")
    ax1.plot(t_vuelo, r["a_numpy"], "-", lw=2.5, alpha=0.6, color=COLOR["numpy"], label="np.gradient")
    ax1.plot(t_vuelo, r["a"], "--", lw=1.5, color=COLOR["my1"], label="my_derivada_numerica")
    ax1.plot(r["t_a_max"], r["a_max"], "o", color=COLOR["datos"], ms=7,
             label=f"máx: {r['a_max']:.2f} m/s² = {r['a_max']/G0:.2f} g")
    _estilo(ax1, "Aceleración axial a(t) = dv/dt", "Tiempo t [s]", r"Aceleración a [m/s$^2$]")

    for clave, nombre, color in (("a_centrada", "my_centrada", COLOR["my1"]),
                                 ("a_adelantada", "my_adelantada", COLOR["my2"]),
                                 ("a_retrasada", "my_retrasada", COLOR["my3"]),
                                 ("a_numpy", "np.gradient", COLOR["numpy"])):
        estilo = ":" if clave == "a_numpy" else "-"
        ax2.semilogy(t_vuelo, np.abs(r[clave] - r["a_exacta"]) + 1e-16, estilo, lw=1.4,
                     color=color, label=nombre)
    _estilo(ax2, "Error absoluto de cada fórmula (3 puntos)", "Tiempo t [s]", r"|error| [m/s$^2$]")

    items = [f"{k}: {v:.1e}" for k, v in r["errores"].items()]
    fila_1, fila_2 = "   ".join(items[:3]), "   ".join(items[3:])
    return ("APARTADO 5 · Derivación numérica (fórmulas de 3 puntos, O(h²))   error máximo [m/s²]\n"
            f"  {fila_1}\n  {fila_2}\n"
            f"  a(0) = {r['a_inicial']:.3f} m/s²   a(120) = {r['a_final']:.3f} m/s²   (extremos: adelantada / retrasada de 3 puntos)\n"
            f"Interpretación: pico de {r['a_max']:.2f} m/s² = {r['a_max']/G0:.2f} g en t = {r['t_a_max']:.0f} s, "
            "por debajo del límite de 4-5 g de la carga útil.")


def vista_apartado_6(ax1, ax2, r):
    _colocar_ejes(ax1, ax2, dos_paneles=False)
    t_fino = np.linspace(0, 120, 600)
    ax1.plot(t_fino, velocidad_teorica(t_fino), color=COLOR["my1"], lw=2, label="v(t) modelo teórico")
    ax1.axhline(r["v_target"], color=COLOR["exacta"], ls="--", label=f"$v_{{target}}$ = {r['v_target']:.0f} m/s")
    estilos = {"numpy (cambio signo)": ("o", COLOR["numpy"], 14),
               "my_newton": ("s", COLOR["my2"], 9),
               "my_biseccion": ("^", COLOR["my3"], 6)}
    for metodo, t in r["raices"].items():
        mk, c, s = estilos[metodo]
        ax1.plot(t, r["v_target"], mk, color=c, ms=s, mfc="none" if mk == "o" else c,
                 label=f"{metodo}: t = {t:.4f} s", zorder=4)
    ax1.axvline(r["t_crit"], color=COLOR["datos"], ls=":", lw=1)
    _estilo(ax1, "Instante crítico: raíz de f(t) = v(t) - v_target", "Tiempo t [s]", "Velocidad v [m/s]")
    ax1.legend(fontsize=8, loc="upper left")

    filas = "\n".join(
        f"  {k:<21} t_crit = {t:.10f} s   "
        f"{(str(r['iteraciones'][k]) + ' it.') if r['iteraciones'][k] else 'vectorizado':>11}   "
        f"error = {abs(t - r['t_exacto']):.1e} s"
        for k, t in r["raices"].items())
    return (f"APARTADO 6 · Ecuación no lineal  (exacta: t = {r['t_exacto']:.10f} s, tol = {TOLERANCIA:.0e})\n" + filas +
            f"\nInterpretación: alcanza {r['v_target']:.0f} m/s a los {r['t_crit']:.2f} s. "
            "Newton (t0 = 60 s, inflexión) | Bisección en [0, 120] s (cambio de signo garantizado).")


def vista_extra(ax1, ax2, r):
    _colocar_ejes(ax1, ax2, dos_paneles=False)
    ax1.plot(t_vuelo, r["q"], color="darkviolet", lw=2, label="q(t) = ½·ρ·v²")
    ax1.plot(r["t_max_q"], r["q_max"], "ko", ms=8, label=f"Max Q = {r['q_max']:.1f} kPa")
    ax1.axvline(r["t_max_q"], color=COLOR["exacta"], ls=":")
    _estilo(ax1, "Extra: presión dinámica y punto de Max Q", "Tiempo t [s]", "Presión dinámica q [kPa]")
    return ("EXTRA · Presión dinámica máxima (combina los Apartados 3 y 4)\n"
            f"  Max Q = {r['q_max']:.2f} kPa en t = {r['t_max_q']:.0f} s a h = {r['h_max_q']:.2f} km\n"
            f"  ρ interpolada en escala logarítmica (atmósfera exponencial, H = {r['H']:.2f} km por encima de 30 km)\n"
            "Interpretación: instante de máximo esfuerzo aerodinámico sobre el fuselaje.")


VISTAS = (vista_apartado_1, vista_apartado_2, vista_apartado_3, vista_apartado_4,
          vista_apartado_5, vista_apartado_6, vista_extra)


def construir_dashboard(res):
    """Crea la ventana única con: menú RadioButtons, zona de gráficos, panel de
    resultados numéricos (texto) y Slider de v_target (solo Apartado 6)."""
    fig = plt.figure(figsize=(15, 8.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Dashboard de telemetría - Ares-I")
    fig.suptitle("Ares-I · Análisis telemétrico y aerodinámico del lanzamiento de una sonda a Marte",
                 fontsize=13, fontweight="bold")

    # Ejes de la zona de gráficos (se reutilizan: ax.clear() en cada vista)
    ax1 = fig.add_axes(ZONA_GRAFICOS)
    ax2 = fig.add_axes(ZONA_GRAFICOS)

    # Panel de resultados numéricos
    ax_texto = fig.add_axes([0.29, 0.02, 0.68, 0.20])
    ax_texto.axis("off")
    texto = ax_texto.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.6,
                          bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#adb5bd"))

    # Menú de navegación
    ax_radio = fig.add_axes([0.015, 0.40, 0.21, 0.48], facecolor="#f8f9fa")
    ax_radio.set_title("Apartados", fontsize=11, fontweight="bold")
    radio = RadioButtons(ax_radio, OPCIONES, activecolor=COLOR["my1"])

    # Slider de la velocidad diana (Apartado 6)
    ax_slider = fig.add_axes([0.06, 0.30, 0.15, 0.03], facecolor="#e9ecef")
    slider = Slider(ax_slider, "$v_{target}$\n[m/s]", V_TARGET_MIN, V_TARGET_MAX,
                    valinit=V_TARGET_DEFECTO, valstep=5.0, color=COLOR["my1"])

    def dibujar(etiqueta):
        indice = OPCIONES.index(etiqueta)
        ax_slider.set_visible(indice == 5)
        if indice == 5:   # el Apartado 6 se recalcula con el valor actual del Slider
            res[6] = resolver_apartado_6(slider.val)
        texto.set_text(VISTAS[indice](ax1, ax2, res[indice + 1]))
        fig.canvas.draw_idle()

    def al_mover_slider(_valor):
        if radio.value_selected == OPCIONES[5]:
            dibujar(OPCIONES[5])

    radio.on_clicked(dibujar)
    slider.on_changed(al_mover_slider)
    dibujar(OPCIONES[0])
    # Se devuelven los widgets para que no los elimine el recolector de basura
    return fig, radio, slider


# =============================================================================
# 6. PROGRAMA PRINCIPAL
# =============================================================================
def main():
    resultados = calcular_todo()
    imprimir_informe(resultados)
    fig, radio, slider = construir_dashboard(resultados)
    plt.show()


if __name__ == "__main__":
    main()
