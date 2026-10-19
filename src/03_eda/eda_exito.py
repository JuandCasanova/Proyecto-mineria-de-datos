# Análisis Exploratorio de Datos (EDA) del dataset integrado
# Proyecto: predicción del éxito en un videojuego por salir
# Ejecutar desde la raíz del proyecto:
#   python src/03_eda/eda_exito.py
#
# Entrada (datos_procesados/):
#   videojuegos_integrado.csv   16.598 x 23   las dos fuentes en una
# Salidas (img/) — una figura por tipo de gráfico:
#   fig1_barras_exito.png          barras      balance de la clase
#   fig2_barras_genero.png         barras      tasa de éxito por género
#   fig3_hist_ventas_global.png    histograma  ventas globales + umbral 1,0 M
#   fig4_box_ventas_genero.png     boxplot     ventas por género (outliers a la vista)
#   fig5_scatter_critico_ventas.png scatter    nota de prensa vs ventas, coloreado por éxito
#   fig6_heatmap_correlaciones.png heatmap     correlación de las variables numéricas
#   fig7_matriz_dispersion.png     pairplot    matriz de dispersión con hue de éxito
#   fig8_lineas_evolucion_anio.png líneas      tasa de éxito y prensa por año
#   fig9_barras_plataforma.png     barras      top 10 plataformas por tasa de éxito
#
# FASE CRISP-DM: 2 · Comprensión de los datos (EDA)
# Etapa KDD:      3 · Exploración de datos
#
# ORDEN: carga -> balance de la clase -> tasas por categoría
#        -> recepción vs éxito -> figuras (distribución, relación, evolución)
#        -> TODOs
#
# Estado: PARCIAL. Las 9 figuras y las medidas básicas ya corren; lo que
# falta está listado al final con el porqué de cada pendiente.
#
# Nota sobre escalas: global_sales va de 0,01 a 82,74 millones, una cola
# alta de casi 3 órdenes de magnitud. En histograma, boxplot y scatter se
# usa escala logarítmica para que el umbral de 1,0 M y las cajas sean
# legibles; la escala NO transforma los datos, solo el eje visible.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend sin pantalla: corre también en headless/CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATASET = RAIZ / "datos_procesados" / "videojuegos_integrado.csv"
CARPETA_IMG = RAIZ / "img"

UMBRAL_EXITO = 1.0   # millones de unidades, decisión de negocio
PALETA = {"si": "#C44E52", "no": "#4C72B0"}


def titulo(n, texto):
    print(f"\n{'=' * 70}\nPASO {n} . {texto}")


def cargar():
    if not RUTA_DATASET.exists():
        raise SystemExit(f"[FALTA] {RUTA_DATASET.relative_to(RAIZ)}\n"
                         "Ejecuta primero: python src/02_preparacion/merged.py")
    df = pd.read_csv(RUTA_DATASET)
    print(f" dataset: {RUTA_DATASET.relative_to(RAIZ)}")
    print(f" dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f" nulos totales: {int(df.isna().sum().sum())} (recepción MNAR, "
          f"conservados a propósito)")
    return df


def medida(nombre, valor):
    print(f"   {nombre:<38} {valor}")


def guardar(fig, nombre):
    ruta = CARPETA_IMG / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   guardada: {ruta.relative_to(RAIZ)}")


def tasa_exito(df, columna, top=None):
    """Tasa de éxito (>= 1,0 M) agrupada por una categoría, ordenada."""
    tabla = (df.groupby(columna)["exito"]
               .agg(juegos="count", exitos="sum", tasa="mean")
               .sort_values("tasa", ascending=False))
    if top:
        tabla = tabla.head(top)
    tabla["tasa"] = (tabla["tasa"] * 100).round(1)
    return tabla


def fig1_y_fig2_balance(df):
    """Barras: balance de la clase y tasa de éxito por género."""
    titulo(4, "FIGURAS DE BARRAS (fig1, fig2)")
    CARPETA_IMG.mkdir(exist_ok=True)

    conteo = df["exito"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.bar(["no éxito (0)", "éxito (1)"],
                    [conteo.get(0, 0), conteo.get(1, 0)],
                    color=[PALETA["no"], PALETA["si"]])
    for barra, valor in zip(barras, [conteo.get(0, 0), conteo.get(1, 0)]):
        ax.text(barra.get_x() + barra.get_width() / 2, valor,
                f" {valor:,}".replace(",", "."), ha="center", va="bottom")
    ax.set_title(f"Balance de la clase (umbral {UMBRAL_EXITO} M)")
    ax.set_ylabel("juegos")
    guardar(fig, "fig1_barras_exito.png")

    orden = df.groupby("genre")["exito"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(orden.index, orden.values * 100, color="#55A868")
    for i, v in enumerate(orden.values * 100):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(df["exito"].mean() * 100, color="#333333", ls="--", lw=1,
               label=f"media del catálogo ({df['exito'].mean():.1%})")
    ax.set_title("fig2 · Tasa de éxito por género (%)")
    ax.set_ylabel("% de juegos con >= 1,0 M")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    guardar(fig, "fig2_barras_genero.png")


def fig3_histograma_ventas(df):
    """Histograma de las ventas globales con el umbral de negocio marcado."""
    titulo(5, "FIGURA DE HISTOGRAMA (fig3)")
    ventas = df["global_sales"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # bins logarítmicos: la cola se extiende de 0,01 a 82,74 M
    bins = np.logspace(np.log10(ventas.min()), np.log10(ventas.max()), 40)
    ax.hist(ventas, bins=bins, color="#4C72B0", edgecolor="white")
    ax.axvline(UMBRAL_EXITO, color="#C44E52", ls="--", lw=2,
               label=f"umbral de éxito = {UMBRAL_EXITO} M")
    ax.set_xscale("log")
    ax.set_xlabel("ventas globales (millones, escala logarítmica)")
    ax.set_ylabel("juegos")
    ax.set_title("fig3 · Distribución de ventas globales")
    ax.legend()
    cola = int((ventas >= UMBRAL_EXITO).sum())
    print(f"   a la derecha del umbral: {cola} juegos "
          f"({cola / len(df):.1%}); la cola es el fenómeno estudiado")
    guardar(fig, "fig3_hist_ventas_global.png")


def fig4_boxplot_ventas(df):
    """Boxplot de ventas por género: outliers visibles y conservados."""
    titulo(6, "FIGURA DE BOXPLOT (fig4)")
    orden = df.groupby("genre")["exito"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df.assign(genre=pd.Categorical(df["genre"], categories=orden,
                                             ordered=True))
    sns.boxplot(data=df_plot, x="genre", y="global_sales", order=orden,
                color="#55A868", fliersize=2, ax=ax)
    ax.axhline(UMBRAL_EXITO, color="#C44E52", ls="--", lw=1.5,
               label=f"umbral de éxito = {UMBRAL_EXITO} M")
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("ventas globales (millones, escala logarítmica)")
    ax.set_title("fig4 · Ventas por género, ordenado por tasa de éxito "
                 "(los puntos son los outliers conservados)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    print("   cada punto fuera de los bigotes es un outlier IQR: se "
          "CONSERVAN, son los éxitos de la cola alta")
    guardar(fig, "fig4_box_ventas_genero.png")


def fig5_scatter_prensa_ventas(df):
    """Scatter: nota de prensa contra ventas, coloreado por la clase."""
    titulo(7, "FIGURA DE SCATTER (fig5)")
    datos = df.dropna(subset=["critic_score"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for clase, color, etiqueta in [(0, PALETA["no"], "no éxito"),
                                   (1, PALETA["si"], "éxito")]:
        sub = datos[datos["exito"] == clase]
        ax.scatter(sub["critic_score"], sub["global_sales"],
                   s=14, alpha=0.55, color=color, label=etiqueta,
                   edgecolors="none")
    ax.axhline(UMBRAL_EXITO, color="#333333", ls="--", lw=1.5,
               label=f"umbral {UMBRAL_EXITO} M")
    ax.set_yscale("log")
    ax.set_xlabel("critic_score (nota de prensa, 0-100)")
    ax.set_ylabel("ventas globales (millones, escala logarítmica)")
    ax.set_title("fig5 · Nota de prensa vs ventas — coloreado por éxito")
    ax.legend()
    r = datos["critic_score"].corr(np.log10(datos["global_sales"]))
    print(f"   correlación de Pearson (nota vs log de las ventas): {r:.3f}")
    print("   la nube se abre hacia arriba a la derecha: sin nota alta "
          "casi no hay éxitos")
    guardar(fig, "fig5_scatter_critico_ventas.png")


def fig6_heatmap_correlaciones(df):
    """Heatmap de correlaciones de las numéricas (incluye la respuesta)."""
    titulo(8, "FIGURA DE HEATMAP (fig6)")
    columnas = ["year", "critic_score", "critic_count", "user_score",
                "user_count", "na_sales", "eu_sales", "jp_sales",
                "other_sales", "global_sales", "exito"]
    matriz = df[columnas].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title("fig6 · Correlación de Pearson entre variables numéricas")
    print("   ojo con las ventas entre sí: na/eu/jp/other/global correlacionan")
    print("   fuerte entre ellas (multicolinealidad) y con exito (fuga si")
    print("   alguna entra en X). critic_score correlaciona con exito sin")
    print("   ser la respuesta: esa señal sí se puede usar.")
    guardar(fig, "fig6_heatmap_correlaciones.png")


def fig7_matriz_dispersion(df):
    """Matriz de dispersión (pairplot) con hue de la clase."""
    titulo(9, "FIGURA DE MATRIZ DE DISPERSIÓN (fig7)")
    columnas = ["critic_score", "user_score", "year", "critic_count"]
    datos = df[columnas + ["exito"]].dropna()
    completas = len(datos)
    if completas > 3000:
        datos = datos.sample(3000, random_state=42)
        print(f"   muestreo aleatorio de 3.000 de {completas} filas "
              f"completas (el pairplot con 16.598 es ilegible y lento)")
    datos = datos.assign(exito=datos["exito"].map({0: "no éxito", 1: "éxito"}))
    paleta = {"no éxito": PALETA["no"], "éxito": PALETA["si"]}
    par = sns.pairplot(datos, hue="exito", hue_order=["no éxito", "éxito"],
                       palette=paleta, corner=True, height=2.2,
                       plot_kws={"s": 10, "alpha": 0.5, "edgecolor": "none"})
    par.figure.suptitle("fig7 · Matriz de dispersión (muestra, coloreada por éxito)",
                        y=1.02, fontsize=13)
    guardar(par.figure, "fig7_matriz_dispersion.png")
    print("   diagonales: distribuciones de cada variable por clase")
    print("   fuera de diagonal: se ve que la prensa separa clases mejor")
    print("   que el año o el número de reseñas")


def fig8_lineas_evolucion(df):
    """Líneas: tasa de éxito y nota mediana de prensa por año."""
    titulo(10, "FIGURA DE LÍNEAS (fig8)")
    por_anio = df.groupby("year").agg(
        tasa_exito=("exito", "mean"),
        critic_score=("critic_score", "median"),
        juegos=("exito", "count"))
    por_anio = por_anio[por_anio["juegos"] >= 20]   # años con muestra mínima

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    color_tasa = "#C44E52"
    ax1.plot(por_anio.index, por_anio["tasa_exito"] * 100, color=color_tasa,
             lw=2, marker="o", ms=4, label="tasa de éxito (%)")
    ax1.set_xlabel("año de lanzamiento")
    ax1.set_ylabel("tasa de éxito (%)", color=color_tasa)
    ax1.tick_params(axis="y", labelcolor=color_tasa)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    color_prensa = "#4C72B0"
    ax2.plot(por_anio.index, por_anio["critic_score"], color=color_prensa,
             lw=2, ls="--", marker="s", ms=4, label="nota mediana de prensa")
    ax2.set_ylabel("critic_score mediano", color=color_prensa)
    ax2.tick_params(axis="y", labelcolor=color_prensa)
    ax2.set_ylim(0, 100)

    ax1.set_title("fig8 · Evolución anual: tasa de éxito vs nota de prensa "
                  "(años con >= 20 juegos)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
    fig.tight_layout()
    guardar(fig, "fig8_lineas_evolucion_anio.png")
    print("   las dos líneas suben y bajan juntas: la prensa y el éxito")
    print("   se mueven parecido año a año (no prueba causalidad)")


def fig9_barras_plataforma(df):
    """Barras horizontales: top 10 plataformas por tasa de éxito."""
    titulo(11, "FIGURA DE BARRAS POR PLATAFORMA (fig9)")
    tabla = tasa_exito(df, "platform", top=10).sort_values("tasa")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(tabla.index, tabla["tasa"], color="#8172B3")
    for i, v in enumerate(tabla["tasa"]):
        ax.text(v + 0.3, i, f"{v} %", va="center", fontsize=9)
    ax.set_xlabel("% de juegos con >= 1,0 M")
    ax.set_title("fig9 · Top 10 plataformas por tasa de éxito")
    ax.set_xlim(0, tabla["tasa"].max() * 1.15)
    guardar(fig, "fig9_barras_plataforma.png")
    print("   la plataforma importa, pero con pocos juegos por consola")
    print("   viejas: revisar el conteo antes de leer la tasa en serio")


def main():
    print("=" * 70)
    print(" FASE 3 CRISP-DM . ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print(f" Éxito definido como global_sales >= {UMBRAL_EXITO} M")
    print("=" * 70)

    titulo(0, "CARGA Y BALANCE DE LA CLASE")
    df = cargar()
    exitos = int(df["exito"].sum())
    tasa = exitos / len(df)
    medida("juegos", f"{len(df):,}".replace(",", "."))
    medida("éxitos (>= 1,0 M)", f"{exitos:,}".replace(",", "."))
    medida("no éxitos", f"{len(df) - exitos:,}".replace(",", "."))
    medida("tasa de éxito", f"{tasa:.1%}")
    medida("línea base (siempre predice 0)", f"{1 - tasa:.1%}")
    print("\n El conjunto está desbalanceado: la exactitud por sí sola "
          "engaña, hay que mirar recall y F1 de la clase 1.")

    titulo(1, "TASA DE ÉXITO POR GÉNERO")
    tabla_genero = tasa_exito(df, "genre")
    print(tabla_genero.to_string())
    top = tabla_genero.index[0]
    print(f"\n Género con mayor tasa: {top} "
          f"({tabla_genero.loc[top, 'tasa']} %) vs peor: "
          f"{tabla_genero.index[-1]} ({tabla_genero.iloc[-1]['tasa']} %)")

    titulo(2, "TASA DE ÉXITO POR PLATAFORMA (TOP 10)")
    print(tasa_exito(df, "platform", top=10).to_string())

    titulo(3, "RECEPCIÓN DE PRENSA VS ÉXITO")
    # La recepción NO es fuga de datos: critic_score se mide antes de las
    # ventas acumuladas, es el mejor predictor disponible en el lanzamiento.
    con_prensa = df[df["sin_critica"] == 0]
    excelente = con_prensa[con_prensa["critic_score"] >= 80]
    mala = con_prensa[con_prensa["critic_score"] < 60]
    print(f"   con nota de prensa          : {len(con_prensa)} "
          f"({len(con_prensa) / len(df):.1%})")
    print(f"   tasa éxito con nota >= 80   : {excelente['exito'].mean():.1%} "
          f"(n = {len(excelente)})")
    print(f"   tasa éxito con nota <  60   : {mala['exito'].mean():.1%} "
          f"(n = {len(mala)})")
    if mala["exito"].mean() > 0:
        print(f"   ratio                      : "
              f"{excelente['exito'].mean() / mala['exito'].mean():.1f} veces")

    fig1_y_fig2_balance(df)
    fig3_histograma_ventas(df)
    fig4_boxplot_ventas(df)
    fig5_scatter_prensa_ventas(df)
    fig6_heatmap_correlaciones(df)
    fig7_matriz_dispersion(df)
    fig8_lineas_evolucion(df)
    fig9_barras_plataforma(df)

    titulo(12, "PENDIENTES (por hacer, con el porqué)")
    print("""
 [ ] 3 familias de medidas (tendencia, dispersión, forma) para las
     numéricas continuas: critic_score, user_score, global_sales
 [ ] Regla IQR sobre las ventas por región con el conteo exacto de
     outliers por columna: fig4 ya los muestra, falta el número en tabla
 [ ] Heatmap género x plataforma de tasa de éxito: si los éxitos se
     concentran en pocas combinaciones, género y plataforma no aportan
     información independiente
 [ ] Distribución de critic_score con y sin la bandera sin_critica, en
     una sola figura, para ver qué se pierde de vista al ignorar los 8.564
     sin nota
 [ ] Curva ROC / precisión-recall placeholder hasta que exista el modelo
""")

    titulo(13, "SIGUIENTE PASO")
    print(" python src/04_transformacion/transformacion.py\n")
    print(f" Total de figuras generadas en img/: {len(list(CARPETA_IMG.glob('fig*.png')))}\n")


if __name__ == "__main__":
    main()
