# Cómo interpretar los gráficos del EDA

Nueve figuras generadas por `src/03_eda/eda_exito.py` (o con
`python main.py --fase eda`) y guardadas en `img/`. Esta guía explica qué
muestra cada una, cómo leerla y —tan importante— qué **no** se puede
concluir de ella.

---

## Convenciones compartidas

| Convención | Significado |
|------------|-------------|
| Color **rojo** = éxito, **azul** = no éxito | fig1, fig5 y fig7 coloreán por la clase `exito` (1 = vendió ≥ 1,0 M) |
| Línea roja punteada vertical/horizontal | umbral de negocio: **1,0 millones** de unidades. No es un percentil ni una decisión estadística |
| **Eje en escala logarítmica** | cada salto de bigote es ×10 (0,01 → 0,1 → 1 → 10). La distancia entre 1 y 2 es la misma que entre 10 y 20: se mide proporción, no diferencia |
| Dashed negro/coloreado de referencia | línea de media del catálogo o valor de comparación |

Todas las figuras se regeneran byte a byte con
`python src/03_eda/eda_exito.py`; ningún número de esta guía está escrito
a mano, sale del dataset `datos_procesados/videojuegos_integrado.csv`.

---

## fig1 · Barras del balance de la clase

**Qué es:** dos barras con el conteo de juegos: 14.517 no éxitos (azul) y
2.081 éxitos (rojo).

**Cómo leerlo:** la altura es el número de juegos. El 87,5 % del conjunto
es clase 0.

**Qué dice:** un modelo que responda siempre «no tiene éxito» acierta el
87,5 % sin aprender nada. Esa es la línea base que hay que batir.

**Qué NO dice:** no dice que el problema sea irresoluble; dice que la
**exactitud** (accuracy) por sí sola no sirve como métrica. El modelo se
evalúa con recall y F1 de la clase 1.

---

## fig2 · Barras de la tasa de éxito por género

**Qué es:** un barra por género con el **porcentaje** de juegos de ese
género que superaron el millón, ordenado de mayor a menor. La línea
discontinua negra es la media del catálogo (12,5 %).

**Cómo leerlo:** toda barra que pasa la línea está por encima de la media
del mercado. Platform 22,0 % y Shooter 19,2 % arriba; Adventure 3,3 % y
Strategy 4,7 % abajo: casi **siete veces** de diferencia entre el mejor y
el peor.

**Qué dice:** el género separa —unos géneros producen éxitos con mucha
más frecuencia que otros— y el orden es contraintuitivo (no gana el
género «premium»).

**Qué NO dice:** no es causal (el género no *provoca* el éxito) y no
muestra cuántos juegos hay en cada barra: un género con pocos títulos
puede saltar por azar. El conteo está en la tabla que imprime el script.

---

## fig3 · Histograma de ventas globales (log)

**Qué es:** distribución de `global_sales` con 40 contenedores
logarítmicos, cortada en el umbral de 1,0 M con línea roja punteada.

**Cómo leerlo:** el eje x va de 0,01 a 82,74 millones en log —cada marca
de diez ocupa el mismo ancho—. La mediana del catálogo es **0,17 M** y el
percentil 90 apenas llega a **1,21 M**: la mayoría de juegos queda a la
izquierda de la línea roja (87,5 %), y la cola derecha, fina pero larga,
son los 2.081 éxitos.

**Qué dice:** el fenómeno estudiado vive en la cola derecha; el umbral de
1,0 M corta justo donde la distribución se vuelve rara. Por eso la cola
no se recorta: winsorizar borraría la diferencia entre 1,1 M y 82 M.

**Qué NO dice:** el área a la izquierda no es «fracaso»: vender 0,8 M es
un caso de negocio distinto de vender 0,02 M, pero ambos son clase 0.

---

## fig4 · Boxplot de ventas por género (log)

**Qué es:** una caja por género (ordenados como en fig2) con las ventas
globales en escala logarítmica.

**Cómo leerlo:**

- la **línea dentro de la caja** es la mediana de ventas de ese género;
- la **caja** abarca el 50 % central (de Q1 a Q3);
- los **bigotes** llegan a 1,5 × IQR;
- los **puntos** fuera de los bigotes son los outliers IQR —1.893
  conservados—;
- la línea roja punteada es el umbral de 1,0 M: géneros cuya caja se
  acerca o cruza esa línea tienen éxitos «normales» dentro de su
  distribución; en los de abajo, superarla es un punto aislado.

**Qué dice:** la cola alta es estructural y está concentrada en pocos
géneros; los puntos sueltos son en su mayoría los éxitos del catálogo.

**Qué NO dice:** un punto fuera de bigote **no es un dato error**: aquí
outlier = éxito masivo (GTA, Mario Kart). Recortarlos por «limpieza»
destruiría la respuesta del problema. Ojo con la escala log: la caja
«pequeña» en el gráfico sigue representando millones de unidades.

---

## fig5 · Scatter de nota de prensa vs ventas

**Qué es:** cada juego es un punto: eje x = `critic_score` (0–100), eje
y = ventas globales (log), color rojo si es éxito. La línea negra es el
umbral de 1,0 M.

**Cómo leerlo:** la zona superior derecha (nota alta, ventas sobre la
línea) se llena de rojos; abajo a la izquierda casi no hay rojos. La
correlación de Pearson entre la nota y el log de las ventas es **+0,382**
—relación positiva moderada— y el resto del EDA lo confirma: con
nota ≥ 80 la tasa de éxito es 39,3 %; con nota < 60, 4,1 % (**9,6
veces** más).

**Qué dice:** la prensa es la señal más fuerte del dataset disponible en
el momento del lanzamiento: sin nota alta, casi no hay éxitos.

**Qué NO dice:** no es causal —un juego esperado recibe nota alta *y*
marketing masivo; la nota puede ser consecuencia del mismo hype—. Y solo
el 48,4 % de los juegos tiene nota: la nube muestra una **muestra
sesgada** hacia lo reseñado, no el catálogo completo.

---

## fig6 · Heatmap de correlaciones

**Qué es:** matriz 11 × 11 con el coeficiente de Pearson (−1 a +1) de
las numéricas. En `RdBu_r`, **rojo = correlación positiva, azul =
negativa, blanco = 0**; el número de cada celda es el valor exacto.

**Cómo leerlo (las celdas que importan):**

| Celda | Valor | Lectura |
|-------|-------|---------|
| `global_sales` × `na_sales` | +0,94 | multicolinealidad: `na_sales` es *parte* de `global_sales`; en un modelo solo puede entrar una de las dos, o ninguna |
| `global_sales` × `exito` | +0,54 | relación definicional: `exito` se derivó de `global_sales`; **es fuga** si alguna entra en `X` |
| `critic_score` × `exito` | +0,33 | la señal usable: la prensa correlaciona con el éxito sin ser la respuesta |
| `critic_score` × `user_score` | +0,58 | prensa y usuarios coinciden, pero no son lo mismo |
| `critic_count` × `critic_score` | +0,43 | juegos más reseñados tienden a mejor nota |
| `user_count` × `user_score` | +0,03 | **la cantidad de reseñas de usuarios no dice nada de su valoración** |
| `year` × `exito` | −0,11 | los años recientes venden menos por título (catálogo más saturado) |

**Qué NO dice:** Pearson solo mide relación **lineal**; un 0 no descarta
relación no lineal. Ninguna celda implica causalidad. Y la caja
`exito` × `global_sales` no es un descubrimiento, es la definición de la
variable objetivo.

---

## fig7 · Matriz de dispersión con hue (pairplot)

**Qué es:** esquina inferior de la matriz de pares entre
`critic_score`, `user_score`, `year` y `critic_count` sobre una muestra
aleatoria de 3.000 de las 6.920 filas con las 4 variables completas,
coloreada por clase.

**Cómo leerlo:**

- la **diagonal** muestra la distribución de cada variable separada por
  clase: si las dos curvas se separan, esa variable distingue éxitos de
  no éxitos;
- las **celdas fuera de diagonal** son scatter de cada par: se busca
  estructura (franja, nube abierta) y no solo manchas ovaladas;
- `critic_score` es la diagonal donde las clases más se separan;
  `user_count` vs `user_score` no muestra ninguna relación —coherente con
  el +0,03 del fig6—.

**Qué NO dice:** es una **muestra** (para que la figura sea legible), no
el dataset completo; y las relaciones cuantitativas están en el fig6,
no se miden «a ojo» aquí.

---

## fig8 · Líneas de evolución por año (eje doble)

**Qué es:** eje izquierdo (rojo) = tasa de éxito (%) de los juegos
lanzados en cada año; eje derecho (azul discontinuo) = `critic_score`
mediano de ese año. Solo se grafican los **29 años con ≥ 20 juegos**
(1981–2016).

**Cómo leerlo:** mira si las dos líneas suben y bajan juntas: lo hacen
—la correlación entre ambas series anuales es **+0,585**—. La línea azul
empieza mucho después que la roja: en los 80 casi no existe nota de
prensa registrada, por eso hay años sin valor azul. La tasa anual recorre
de 4,4 % (2016) hasta 81 % (un año temprano con catálogo pequeño).

**Qué dice:** la prensa y el éxito se mueven parecidos año a año; además
la tasa de éxito cae en las últimas filas, coherente con un catálogo más
saturado.

**Qué NO dice:** **+0,585 no es causalidad** —ambas pueden responder a
una tercera cosa (presupuesto de marketing de la industria). Y los años
tempranos tienen pocos juegos: una tasa del 81 % sobre un catálogo
pequeño pesa lo mismo en la línea que el 4,4 % de 2016 con 344 juegos.

---

## fig9 · Barras horizontales: top 10 plataformas

**Qué es:** las 10 plataformas con mayor tasa de éxito, en barras
horizontales (%).

**Cómo leerlo:** compara barra **y** conteo. NES 75,5 % y Game Boy 56,1 %
lideran, pero ambas tienen solo 98 juegos con catálogo de época;
PS4 21,7 % (336 juegos) y X360 18,7 % (**1.265 juegos**) son las
decisiones más robustas por volumen. Al final de la tabla general
aparecen NG (12 juegos), WS (6) y TG16 (2) con 0 % de éxito.

**Qué dice:** la plataforma importa, y las consolas con catálogo amplio
y moderno concentran la mayoría de los éxitos absolutos aunque su
*tasa* no lidere.

**Qué NO dice:** **tasa sin conteo engaña** —cualquier consola con
pocos juegos puede quedar arriba o abajo por azar—. Y plataforma
correlaciona con era: NES es «época de NES», no solo «consola».

---

## Reglas de oro al leer estas figuras

1. **Escalas logarítmicas** (fig3, fig4, fig5): la distancia es
   proporcional; un centímetro a la derecha es «multiplicar por», no
   «sumar».
2. **Correlación ≠ causalidad** (fig5, fig6, fig8): nada de lo que
   muestren prueba que la nota *cause* ventas.
3. **Outlier ≠ error** (fig4): en este proyecto la cola alta son los
   éxitos; se conservan.
4. **Tasa sin n = trampa** (fig2, fig9): siempre mira cuántos juegos
   hay detrás del porcentaje.
5. **Balance de clases** (fig1): la exactitud sola nunca decide; mira
   recall y F1 de la clase 1.
6. **Muestra parcial** (fig5, fig7): solo el 48,4 % del catálogo tiene
   nota de prensa; el resto de figuras lo indica en sus notas.

---

## Índice rápido

| Archivo | Tipo de gráfico | Pregunta que responde |
|---------|-----------------|------------------------|
| `fig1_barras_exito.png` | barras | ¿Qué tan desbalanceado está el conjunto? |
| `fig2_barras_genero.png` | barras | ¿Qué géneros superan la media de éxito? |
| `fig3_hist_ventas_global.png` | histograma | ¿Cómo se distribuyen las ventas y dónde queda el umbral? |
| `fig4_box_ventas_genero.png` | boxplot | ¿Dónde están los outliers de ventas y quién los tiene? |
| `fig5_scatter_critico_ventas.png` | scatter | ¿La nota de prensa se asocia con vender más? |
| `fig6_heatmap_correlaciones.png` | heatmap | ¿Qué variables se mueven juntas y cuáles son fuga? |
| `fig7_matriz_dispersion.png` | matriz de dispersión | ¿Cómo se ven las 4 numéricas clave por clase? |
| `fig8_lineas_evolucion_anio.png` | líneas (eje doble) | ¿Cómo evolucionan éxito y prensa año a año? |
| `fig9_barras_plataforma.png` | barras | ¿Qué plataformas tienen más tasa de éxito? |
