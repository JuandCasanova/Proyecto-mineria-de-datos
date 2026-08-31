# Análisis del Éxito en la Industria de los Videojuegos

##  ¿De qué trata el proyecto?

El objetivo principal de este proyecto es analizar los factores que determinan si un videojuego es exitoso en el mercado global. Para definir el concepto de "éxito", combinamos dos criterios clave:
* **Rendimiento económico:** Altas ventas globales.
* **Recepción del público y la crítica:** Calificaciones destacadas por parte de usuarios y prensa especializada.

A partir de un dataset de más de **16.000 videojuegos** (*Video Game Sales with Ratings*), exploraremos patrones y tendencias clave para responder a preguntas como:
* ¿Qué géneros y plataformas tienen mayor probabilidad de éxito?
* ¿Influye más la calificación de la crítica o la de los usuarios en las ventas?
* ¿Es posible predecir si un nuevo juego triunfará antes de su lanzamiento al mercado?

---

##  Metodología (CRISP-DM)

El proyecto se desarrolla siguiendo la metodología estándar de la industria **CRISP-DM**, estructurada en 6 fases:

1. **Entendimiento del negocio:** Definición de los criterios de éxito e identificación de objetivos analíticos.
2. **Entendimiento de los datos:** Exploración de la estructura del dataset, distribuciones y comportamiento de variables.
3. **Preparación de los datos:** Tratamiento de datos faltantes, corrección de inconsistencias de formato y codificación de variables categóricas (géneros, plataformas).
4. **Modelado:** Aplicación de algoritmos de **Clasificación** (para predecir el éxito de un título) y **Clustering** (para agrupar videojuegos por perfiles similares).
5. **Evaluación:** Validación de la precisión y efectividad de los modelos machine learning.
6. **Despliegue:** Presentación e interpretación de hallazgos mediante un panel de control interactivo (Power BI / Streamlit).

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3 (Ejecutado en Google Colab / Jupyter Notebooks)
* **Manipulación de datos:** `pandas`, `numpy`
* **Visualización de datos:** `matplotlib`, `seaborn`
* **Machine Learning / Minería de datos:** `scikit-learn` (Modelos de clasificación, clustering y métricas de evaluación)
