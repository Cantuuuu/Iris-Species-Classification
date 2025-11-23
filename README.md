# 🌸 Clasificación de Especies de Iris con KNN

## Descripción del Proyecto

Este proyecto implementa una aplicación web interactiva que utiliza **Machine Learning** para clasificar especies de flores Iris basándose en sus características morfológicas. El sistema permite a los usuarios ingresar las medidas de una flor y obtener una predicción inmediata sobre su especie, junto con visualizaciones interactivas que facilitan la comprensión del resultado.

El modelo clasifica entre tres especies diferentes:
- **Iris Setosa** 
- **Iris Versicolor** 
- **Iris Virginica** 

## Objetivo

Desarrollar un sistema completo de clasificación que incluya:
1. Análisis exploratorio de datos (EDA)
2. Preprocesamiento y limpieza de datos
3. Entrenamiento y optimización de un modelo KNN
4. Creación de una interfaz web interactiva
5. Despliegue para acceso público

## Despliegue

🔗 **[Probar la aplicación en Streamlit Cloud](https://iris-species-classification-proyecto-final.streamlit.app/)**

## Dataset

**Fuente**: [UCI Machine Learning Repository - Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)

### Características del Dataset:
- **Total de muestras**: 150 (50 por cada especie)
- **Características numéricas**: 4
  - Longitud del Sépalo (cm)
  - Ancho del Sépalo (cm)
  - Longitud del Pétalo (cm)
  - Ancho del Pétalo (cm)
- **Variable objetivo**: Especie (categórica)
- **Balance**: Dataset perfectamente balanceado

## Metodología

### 1. Análisis Exploratorio de Datos (EDA)

Se realizó un análisis exhaustivo para comprender las características del dataset:

#### Visualizaciones Implementadas:
- **Correlaciones**: Heatmap de correlaciones entre variables
- **Boxplots**: Detección de outliers por especie
- **Scatter Plots**: Análisis de separabilidad entre especies
- **Pairplots**: Relaciones multivariadas

#### Hallazgos Clave:
- No se encontraron valores nulos
- No se detectaron outliers significativos
- **Iris Setosa** es linealmente separable del resto
- **Iris Versicolor** e **Iris Virginica** presentan cierto solapamiento
- Las características del pétalo son más discriminativas que las del sépalo

### 2. Preprocesamiento de Datos

#### Codificación de Variables
```python
LabelEncoder:
- Iris-setosa → 0
- Iris-versicolor → 1
- Iris-virginica → 2
```

#### Normalización
Se aplicó **StandardScaler** para estandarizar las características:
- Media (μ) = 0
- Desviación estándar (σ) = 1

Esto es crucial para KNN ya que el algoritmo se basa en distancias.

#### División de Datos
- **Train**: 80% (120 muestras)
- **Test**: 20% (30 muestras)
- **Estratificación**: Manteniendo la proporción de clases

### 3. Selección y Entrenamiento del Modelo

#### Algoritmo Elegido: K-Nearest Neighbors (KNN)

**Razones de la elección**:
- Simple pero efectivo para problemas multiclase
- No asume distribución de datos
- Ideal para datasets pequeños y bien definidos

#### Optimización de Hiperparámetros

Se utilizó **GridSearchCV** con validación cruzada de 5 folds:

| Hiperparámetro | Valores Probados | Valor Óptimo |
|----------------|------------------|--------------|
| `n_neighbors` | range(1, 20) | **17** |
| `weights` | uniform, distance | **distance** |
| `metric` | euclidean, manhattan, minkowski | **euclidean** |

**Configuración final del modelo**:
```python
KNeighborsClassifier(
    n_neighbors=17,
    weights='distance',
    metric='euclidean'
)
```

### 4. Evaluación del Modelo

#### Métricas Obtenidas

| Métrica | Valor | 
|---------|---------------------------|
| **Accuracy** | 96.67% |
| **Precision** | 96.97% |
| **Recall** | 96.67% |
| **F1-Score** | 96.66% |


### 5. Serialización del Modelo

Los componentes del modelo se guardaron usando **pickle**:

```python
knnModel.pkl        # Modelo KNN entrenado
scaler.pkl          # StandardScaler con parámetros ajustados
labelEncoder.pkl    # Codificador de especies
metricas.pkl        # Diccionario con métricas de evaluación
```

## Interfaz de Usuario - Streamlit

### Funcionalidades Implementadas

#### 1. **Barra Lateral de Entrada**
- **Entrada numerica** para las 4 características
- Valores mínimos y máximos basados en el dataset real
- Actualización en tiempo real de la predicción

#### 2. **Panel de Predicción**
- **Resultado principal**: Especie predicha
- **Probabilidades**: Porcentaje de confianza para cada especie

### **Visualizaciones Interactivas**

#### 1. **Gráfico 3D Interactivo (Scatter 3D)**

**Librería:** `Plotly` (`go.Scatter3d`)

**Descripción:**
- Visualiza el dataset completo en **3 dimensiones seleccionables** por el usuario
- Muestra las **150 muestras** del dataset, coloreadas por especie
- Resalta **tu predicción** con un marcador rojo en forma de diamante
- Permite **rotación e interacción** en tiempo real (zoom, pan)

**Características seleccionables:**
- Por defecto: `Longitud del Pétalo`, `Ancho del Pétalo`, `Longitud del Sépalo`
- El usuario puede elegir cualquier combinación de 3 características mediante un selector en el sidebar

#### 2.  **Gráfico de Barras Agrupadas (Bar Chart)**

**Librería:** `Plotly` (`go.Bar`)

**Descripción:**
- Compara los **valores promedio** de las 4 características entre las 3 especies
- Cada característica tiene una barra de diferente color
- Muestra valores exactos encima de cada barra (tooltip automático)

---

#### 3. **Gráfico de Pastel (Donut Chart)**

**Librería:** `Plotly` (`go.Pie`)

**Descripción:**
- Muestra la **distribución de especies** en el dataset
- Formato de dona con hueco en el centro (`hole=0.4`)

---

#### 4.  **Gráfico de Líneas Comparativas (Line + Markers)**

**Librería:** `Plotly` (`go.Scatter`)

**Descripción:**
- Compara **tus valores ingresados** vs **promedio de la especie predicha**
- Dos líneas superpuestas:
  1. **Línea roja sólida**: Tus medidas ingresadas
  2. **Línea punteada**: Promedio de la especie detectada

### Integrantes del Equipo

| Nombre | Matrícula | GitHub |
|--------|-----------|--------|
| **Arturo Cantú Olivarez** | 1919010 | [@Cantuuuu](https://github.com/Cantuuuu) |
| **Diego Sebastián Cruz Cervantes** | 1910032 |  [@Cantuuuu](https://github.com/Cantuuuu)  |


</div>
