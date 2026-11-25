"""
=============================================================================
 src/gui.py
 GRAPHICAL USER INTERFACE: gráficas y colores de cada apartado
 Proyecto Ares-I · Grado en Ingeniería Aeroespacial · Programación
=============================================================================
 Este módulo solo DIBUJA. Recibe los resultados ya calculados en physics.py
 y los representa con matplotlib. La ventana y los widgets (RadioButtons,
 Slider) se montan en main.py.

   1. Colores y nombres del menú
   2. Estilo común de las gráficas (título, unidades, rejilla, leyenda)
   3. Una función vista_apartado_X por apartado: dibuja en los ejes y devuelve
      el texto con los resultados numéricos y su interpretación física
=============================================================================
"""

import numpy as np

from .physics import (G0, TOLERANCIA, t_vuelo, v_vuelo, V_sensor, P_sensor,
                      altitudes_tab, densidades_tab, velocidad_teorica)


# =============================================================================
# 1. COLORES Y MENÚ
# =============================================================================
# Colores fijos por método (mismo color = mismo método en todas las vistas)
COLOR = {"numpy": "#222222", "my1": "#1f77b4", "my2": "#d62728", "my3": "#2ca02c",
         "my4": "#ff7f0e", "exacta": "#9e9e9e", "datos": "#d62728", "extra": "darkviolet",
         "panel": "#f8f9fa", "borde": "#adb5bd", "slider": "#e9ecef"}

OPCIONES = (
    "1. Sistema lineal",
    "2. Regresión lineal",
    "3. Integración",
    "4. Interpolación",
    "5. Derivación",
    "6. Ecuación no lineal",
    "Extra: Max Q",
)
INDICE_APARTADO_6 = 5   # posición del Apartado 6 en el menú (el que lleva Slider)

# Zona de gráficos (en coordenadas de figura): [izq, abajo, ancho, alto]
ZONA_GRAFICOS = [0.29, 0.33, 0.68, 0.58]
SEPARACION = 0.07


# =============================================================================
# 2. ESTILO COMÚN
# =============================================================================
def _estilo(ax, titulo, xlabel, ylabel, leyenda=True):
    """Formato científico común: título, etiquetas con unidades, rejilla y leyenda."""
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    if leyenda:
        ax.legend(fontsize=8, loc="best")


def _colocar_ejes(ax1, ax2, dos_paneles):
    """Limpia los ejes (ax.clear()) y hace que la vista ocupe toda la zona de
    gráficos o la divida en dos paneles."""
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


# =============================================================================
# 3. VISTAS POR APARTADO
# =============================================================================
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
                label="h > 30 km (fuera de tabla)")
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
    ax1.plot(t_vuelo, r["q"], color=COLOR["extra"], lw=2, label="q(t) = ½·ρ·v²")
    ax1.plot(r["t_max_q"], r["q_max"], "ko", ms=8, label=f"Max Q = {r['q_max']:.1f} kPa")
    ax1.axvline(r["t_max_q"], color=COLOR["exacta"], ls=":")
    _estilo(ax1, "Extra: presión dinámica y punto de Max Q", "Tiempo t [s]", "Presión dinámica q [kPa]")
    return ("EXTRA · Presión dinámica máxima (combina los Apartados 3 y 4)\n"
            f"  Max Q = {r['q_max']:.2f} kPa en t = {r['t_max_q']:.0f} s a h = {r['h_max_q']:.2f} km\n"
            f"  ρ interpolada en escala logarítmica (atmósfera exponencial, H = {r['H']:.2f} km por encima de 30 km)\n"
            "Interpretación: instante de máximo esfuerzo aerodinámico sobre el fuselaje.")


# Mismo orden que OPCIONES: VISTAS[i] dibuja la opción i del menú
VISTAS = (vista_apartado_1, vista_apartado_2, vista_apartado_3, vista_apartado_4,
          vista_apartado_5, vista_apartado_6, vista_extra)
