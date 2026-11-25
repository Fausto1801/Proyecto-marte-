# Ares-I · Análisis telemétrico y aerodinámico del lanzamiento de una sonda a Marte

Proyecto de **Programación – Grado en Ingeniería Aeroespacial** (presentación oral, enunciado 6).

**Integrantes:** Pablo Ruiz, Jon Recio, Beltrán Rodríguez, Fausto Rapisardi

Dashboard interactivo en `matplotlib` que procesa la telemetría del vehículo Ares-I con métodos numéricos.
Cada apartado se resuelve **primero con numpy** y **después con las funciones propias `my_`**, comparando
el resultado y el error frente a la solución exacta.

## Ejecución

```bash
pip install numpy matplotlib
python main.py
```

Se imprime un informe por consola (cada resultado con sus unidades y su interpretación física) y se abre una
única ventana interactiva: menú `RadioButtons` (apartados 1–6 + extra), panel de resultados numéricos y
`Slider` de velocidad diana (300–1000 m/s) en el Apartado 6.

## Organización: qué hace cada fichero

| Fichero | Qué contiene |
|---|---|
| `main.py` | **Construye la interfaz gráfica.** Es el fichero que se ejecuta: crea la ventana, el menú `RadioButtons`, el `Slider` y el panel de resultados, y los conecta con los cálculos |
| `physics.py` | **Los apartados 1–6.** Datos del enunciado y un `def resolver_apartado_X` por apartado, que llama a las funciones de `maths.py`. También el informe por consola con la interpretación física |
| `maths.py` | **Las funciones numéricas.** En cada apartado, primero la versión con numpy (`np_...`) y después las funciones propias (`my_...`) |
| `gui.py` | **Graphical user interface: gráficas y colores.** Una función `vista_apartado_X` por apartado que dibuja sus gráficas |

Flujo: `maths.py` → `physics.py` → `gui.py` → `main.py`.

## Dónde está cada apartado

| Apartado | `physics.py` (resolución) | `maths.py` – numpy | `maths.py` – funciones propias | `gui.py` (gráficas) |
|---|---|---|---|---|
| 1. Sistema lineal | `resolver_apartado_1` | `np_sistema_lineal` | `my_gauss`, `my_lu`, `my_jacobi`, `my_gauss_seidel` | `vista_apartado_1` |
| 2. Regresión lineal | `resolver_apartado_2` | `np_regresion_lineal` | `my_linear_regression` | `vista_apartado_2` |
| 3. Integración | `resolver_apartado_3` | `np_trapecio` | `my_trapecio`, `my_simpson`, `my_trapecio_acumulado` | `vista_apartado_3` |
| 4. Interpolación | `resolver_apartado_4` | `np_interp_lineal`, `np_polinomio_interpolante` | `my_interp_lineal`, `my_vandermonde`, `my_lagrange` | `vista_apartado_4` |
| 5. Derivación | `resolver_apartado_5` | `np_derivada` | `my_derivada_centrada`, `my_derivada_adelantada`, `my_derivada_retrasada`, `my_derivada_numerica` | `vista_apartado_5` |
| 6. Ecuación no lineal | `resolver_apartado_6` | `np_raiz_cambio_signo` | `my_newton`, `my_biseccion` | `vista_apartado_6` |
| Extra: Max Q | `resolver_extra_max_q` | — | — | `vista_extra` |

## Resultados

| Apartado | Método numpy | Funciones propias | Resultado |
|---|---|---|---|
| 1. Sistema lineal (masas de propelente) | `np.linalg.solve` | Gauss, LU, Jacobi, Gauss-Seidel | m = 21 666.7 / 10 833.3 / 17 500 kg |
| 2. Regresión lineal (sensor Pitot) | `np.polyfit` | Ecuación normal | P = 21.11·V − 0.28 kPa (R² = 0.9999) |
| 3. Integración (altitud) | `np.trapezoid` | Trapecio, Simpson, trapecio acumulado | h(120 s) = 135.84 km |
| 4. Interpolación (densidad) | `np.interp`, `np.polyfit` (Lagrange) | Lineal, Vandermonde, Lagrange | ρ(t) a lo largo del vuelo |
| 5. Derivación (aceleración) | `np.gradient` | Diferencias finitas de 3 puntos: centradas, adelantadas y retrasadas | a_max = 31.24 m/s² = 3.19 g |
| 6. Ecuación no lineal (Mach 1) | Cambio de signo vectorizado | Newton-Raphson, bisección | t_crit = 30.10 s |
| Extra | — | — | Max Q = 78.6 kPa en t = 43 s |
