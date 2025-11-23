import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuración de la página
st.set_page_config(page_title="Clasificador de Iris", layout="wide", page_icon="🌸")


# Cargar modelo y componentes
@st.cache_resource
def cargaModelo():
    knn = pickle.load(open('knnModel.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoder = pickle.load(open('labelEncoder.pkl', 'rb'))
    return knn, scaler, encoder


@st.cache_data
def cargaDataset():
    return pd.read_csv('irisProcesado.csv')

# Cargar métricas reales del test set
@st.cache_data
def cargarMetricas():
    try:
        with open('metricas.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("⚠️ Archivo de métricas no encontrado. Ejecuta el notebook primero.")
        return None

metricas = cargarMetricas()
knnModel, scaler, labelEncoder = cargaModelo()
df = cargaDataset()
caracteristicas = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Título
st.title("🌸 Clasificador de Especies de Iris con KNN 🌸")
st.markdown("---")

# Sidebar para predicción
st.sidebar.header("Panel de Predicción")
st.sidebar.write("Ingresa las medidas de la flor:")

# Inputs numéricos para las características
sepalLength = st.sidebar.number_input(
    'Longitud del Sépalo (cm)',
    min_value=4.0,
    max_value=8.0,
    value=5.8,
    step=0.1,
    format="%.1f")

sepalWidth = st.sidebar.number_input(
    'Ancho del Sépalo (cm)',
    min_value=2.0,
    max_value=4.5,
    value=3.0,
    step=0.1,
    format="%.1f")

petalLength = st.sidebar.number_input(
    'Longitud del Pétalo (cm)',
    min_value=1.0,
    max_value=7.0,
    value=4.0,
    step=0.1,
    format="%.1f")

petalWidth = st.sidebar.number_input(
    'Ancho del Pétalo (cm)',
    min_value=0.1,
    max_value=2.5,
    value=1.3,
    step=0.1,
    format="%.1f")

# Realizar predicción
userInput = [[sepalLength, sepalWidth, petalLength, petalWidth]]
userInputScaled = scaler.transform(userInput)
prediction = knnModel.predict(userInputScaled)[0] # Obtener la clase predicha, indice 0 porque
predictionProba = knnModel.predict_proba(userInputScaled)[0] #Sklearn ofrece esta probabilidad, pero no es una estadistica como tal, si no que es una estimacion basada en los vecinos.
predictedSpecies = labelEncoder.inverse_transform([prediction])[0]

# Mostrar predicción en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Resultado:")
st.sidebar.success(f"**Especie: {predictedSpecies}**")

# Mostrar probabilidades
st.sidebar.write("**Probabilidades:**")
for idx, species in enumerate(labelEncoder.classes_):
    st.sidebar.write(f"{species}: {predictionProba[idx]:.2%}")


st.header("Métricas del Modelo")
if metricas:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Accuracy", f"{metricas['accuracy']:.2%}")
    with kpi2:
        st.metric("Precision", f"{metricas['precision']:.2%}")
    with kpi3:
        st.metric("Recall", f"{metricas['recall']:.2%}")
    with kpi4:
        st.metric("F1-Score", f"{metricas['f1_score']:.2%}")

st.markdown("---")

# Información del modelo
st.subheader("Configuración del Modelo")
cfg1, cfg2, cfg3, cfg4 = st.columns(4)
with cfg1:
    st.write(f"**Algoritmo:** K-Nearest Neighbors")
with cfg2:
    st.write(f"**K (vecinos):** {knnModel.n_neighbors}")
with cfg3:
    st.write(f"**Métrica:** {knnModel.metric}")
with cfg4:
    st.write(f"**Pesos:** {knnModel.weights}")

st.markdown("---")

# Visualización 3D
st.header("Visualización 3D - Predicción de especies")

# Selector de características para visualización 3D, en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Caracteríscas del grafico 3D")

caracteGrafico3D = st.sidebar.multiselect( "Selecciona 3 características para visualizar:",
                                           caracteristicas,
                                           default=['PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm'])
#Los valores por defecto son los que mejor separan las clases, se ve en la matriz de correlacion.

# Validar selección
if len(caracteGrafico3D) != 3:
    st.warning("Selecciona exactamente 3 características para la visualización 3D")
    st.stop()

# Mapeo de nombres para los ejes
nombresEjes = {
    'SepalLengthCm': 'Longitud Sépalo (cm)',
    'SepalWidthCm': 'Ancho Sépalo (cm)',
    'PetalLengthCm': 'Longitud Pétalo (cm)',
    'PetalWidthCm': 'Ancho Pétalo (cm)'
}

# Crear scatter 3D con las características seleccionadas
fig = go.Figure()

# Agregar puntos del dataset por especie
for species in df['Species'].unique():
    subset = df[df['Species'] == species]
    fig.add_trace(go.Scatter3d(
        x=subset[caracteGrafico3D[0]],
        y=subset[caracteGrafico3D[1]],
        z=subset[caracteGrafico3D[2]],
        mode='markers',
        name=species,
        marker=dict(size=5, opacity=0.6)))

# Obtener valores del usuario según características seleccionadas
valoresUsuario = {
    'SepalLengthCm': sepalLength,
    'SepalWidthCm': sepalWidth,
    'PetalLengthCm': petalLength,
    'PetalWidthCm': petalWidth
}

# Agregar punto de predicción
fig.add_trace(go.Scatter3d(
    x=[valoresUsuario[caracteGrafico3D[0]]],
    y=[valoresUsuario[caracteGrafico3D[1]]],
    z=[valoresUsuario[caracteGrafico3D[2]]],
    mode='markers',
    name='Tu Predicción',
    marker=dict(size=15, color='red', symbol='diamond', line=dict(color='black', width=2))))

fig.update_layout(
    scene=dict(
        xaxis_title=nombresEjes[caracteGrafico3D[0]],
        yaxis_title=nombresEjes[caracteGrafico3D[1]],
        zaxis_title=nombresEjes[caracteGrafico3D[2]]),
    height=600,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# Información sobre las características visualizadas
st.info(f""" **Características visualizadas:** {', '.join([nombresEjes[c] for c in caracteGrafico3D])}

El modelo KNN usa las **4 características** para clasificar, pero solo se pueden visualizar 3 dimensiones a la vez.
""")

st.markdown("---")
st.header("Análisis del dataset")

# 1 Comparación de características promedio
st.subheader("Características promedio en cada especie")
promedios = df.groupby('Species')[caracteristicas].mean()

barrasComparativas = go.Figure()
for caracteristica in caracteristicas:
    barrasComparativas.add_trace(go.Bar(
        name=nombresEjes[caracteristica],
        x=promedios.index,
        y=promedios[caracteristica],
        text=promedios[caracteristica].round(2),
        textposition='auto'))

barrasComparativas.update_layout(
    barmode='group',
    yaxis_title='Centímetros (cm)',
    height=400,
    showlegend=True)
st.plotly_chart(barrasComparativas, use_container_width=True)

st.markdown("---")

# 2 Distribución de especies
st.subheader("Distribución del Dataset")
col1, col2 = st.columns(2)
with col1:
    # Grafico de pastel
    conteoEspecies = df['Species'].value_counts()
    graficoPastelEspecies = go.Figure(data=[go.Pie(
        labels=conteoEspecies.index,
        values=conteoEspecies.values,
        hole=0.4)])
    graficoPastelEspecies.update_layout(height=350, title_text='Cantidad por especie')
    st.plotly_chart(graficoPastelEspecies, use_container_width=True)

with col2:
    # Estadísticas básicas
    st.write("**Total de muestras:**", len(df))
    st.write("**Distribución:**")
    for especie, cantidad in conteoEspecies.items():
        st.write(f"- {especie}: {cantidad} ({cantidad/len(df)*100:.1f}%)")

st.markdown("---")

# 3 Comparación de los valores ingresados vs promedios
st.subheader("Flor analizada vs promedios")

# Crear DataFrame para comparar
comparacion = pd.DataFrame({ # Se crea una tabla con 3 columnas
    'Característica': [nombresEjes[c] for c in caracteristicas],
    'Tu valor': [sepalLength, sepalWidth, petalLength, petalWidth],
    'Promedio general': [df[c].mean() for c in caracteristicas],
    f'Promedio {predictedSpecies}': [promedios.loc[predictedSpecies, c] for c in caracteristicas]})
fig_comp = go.Figure()

fig_comp.add_trace(go.Scatter(
    x=comparacion['Característica'], #Cada caracteristica en el eje x
    y=comparacion['Tu valor'], #Los valores ingresados por el usuario en el eje y
    mode='markers+lines',
    name='Flor analizada',
    marker=dict(size=12, color='red'),
    line=dict(width=2) ))

fig_comp.add_trace(go.Scatter(
    x=comparacion['Característica'],
    y=comparacion[f'Promedio {predictedSpecies}'], #Los valores promedio de la especie predicha
    mode='markers+lines',
    name=f'Promedio {predictedSpecies}', #La especie predicha
    marker=dict(size=10),
    line=dict(dash='dash')))

fig_comp.update_layout(
    yaxis_title='Centímetros (cm)',
    height=400,
    showlegend=True)

st.plotly_chart(fig_comp, use_container_width=True)


