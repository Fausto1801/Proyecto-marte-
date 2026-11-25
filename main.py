"""
=============================================================================
 main.py
 ANÁLISIS TELEMÉTRICO Y AERODINÁMICO DEL LANZAMIENTO DE UNA SONDA A MARTE
 Vehículo de lanzamiento Ares-I · Dashboard interactivo de telemetría
 Grado en Ingeniería Aeroespacial · Programación
=============================================================================
 Punto de entrada del proyecto: CONSTRUYE LA INTERFAZ GRÁFICA.

 Organización del repositorio (cada fichero tiene una única función):

   main.py ....... construye la interfaz: ventana, menú RadioButtons,
                   Slider y panel de resultados (este fichero)
   physics.py .... datos del enunciado y resolución de los apartados 1-6
                   (importa los métodos de maths.py)
   maths.py ...... funciones numéricas: numpy (np_) y propias (my_)
   gui.py ........ gráficas y colores de cada apartado

   Flujo:  maths.py  ->  physics.py  ->  gui.py  ->  main.py

 Ejecución:   python main.py
 Requisitos:  numpy >= 1.20, matplotlib >= 3.5
=============================================================================
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

from physics import (calcular_todo, imprimir_informe, resolver_apartado_6,
                     V_TARGET_DEFECTO, V_TARGET_MIN, V_TARGET_MAX)
from gui import COLOR, OPCIONES, INDICE_APARTADO_6, ZONA_GRAFICOS, VISTAS


def construir_interfaz(res):
    """Crea la ventana única con: menú RadioButtons, zona de gráficos, panel de
    resultados numéricos (ax.text) y Slider de v_target (solo Apartado 6)."""
    fig = plt.figure(figsize=(15, 8.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Dashboard de telemetría - Ares-I")
    fig.suptitle("Ares-I · Análisis telemétrico y aerodinámico del lanzamiento de una sonda a Marte",
                 fontsize=13, fontweight="bold")

    # --- Zona de gráficos: dos ejes que cada vista reutiliza (ax.clear()) ----
    ax1 = fig.add_axes(ZONA_GRAFICOS)
    ax2 = fig.add_axes(ZONA_GRAFICOS)

    # --- Panel de resultados numéricos ---------------------------------------
    ax_texto = fig.add_axes([0.29, 0.02, 0.68, 0.20])
    ax_texto.axis("off")
    texto = ax_texto.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.6,
                          bbox=dict(boxstyle="round,pad=0.6", facecolor=COLOR["panel"],
                                    edgecolor=COLOR["borde"]))

    # --- Menú de navegación (RadioButtons) -----------------------------------
    ax_radio = fig.add_axes([0.015, 0.40, 0.21, 0.48], facecolor=COLOR["panel"])
    ax_radio.set_title("Apartados", fontsize=11, fontweight="bold")
    radio = RadioButtons(ax_radio, OPCIONES, activecolor=COLOR["my1"])

    # --- Slider de la velocidad diana (solo visible en el Apartado 6) --------
    ax_slider = fig.add_axes([0.06, 0.30, 0.15, 0.03], facecolor=COLOR["slider"])
    slider = Slider(ax_slider, "$v_{target}$\n[m/s]", V_TARGET_MIN, V_TARGET_MAX,
                    valinit=V_TARGET_DEFECTO, valstep=5.0, color=COLOR["my1"])

    def dibujar(etiqueta):
        """Callback del menú: limpia y redibuja la vista elegida."""
        indice = OPCIONES.index(etiqueta)
        ax_slider.set_visible(indice == INDICE_APARTADO_6)
        if indice == INDICE_APARTADO_6:   # se recalcula t_crit con el valor actual del Slider
            res[6] = resolver_apartado_6(slider.val)
        texto.set_text(VISTAS[indice](ax1, ax2, res[indice + 1]))
        fig.canvas.draw_idle()

    def al_mover_slider(_valor):
        """Callback del Slider: recalcula la raíz en tiempo real."""
        if radio.value_selected == OPCIONES[INDICE_APARTADO_6]:
            dibujar(OPCIONES[INDICE_APARTADO_6])

    radio.on_clicked(dibujar)
    slider.on_changed(al_mover_slider)
    dibujar(OPCIONES[0])
    # Se devuelven los widgets para que no los elimine el recolector de basura
    return fig, radio, slider


def main():
    resultados = calcular_todo()          # physics.py (con los métodos de maths.py)
    imprimir_informe(resultados)          # resultados + interpretación física
    fig, radio, slider = construir_interfaz(resultados)
    plt.show()


if __name__ == "__main__":
    main()
