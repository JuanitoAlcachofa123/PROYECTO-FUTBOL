import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Agregar estilo CSS personalizado con animaciones
st.markdown(
    """
    <style>
    .main {
        background-color: #2e3b4e;
        color: #ffffff;
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stButton button {
        background-color: #1f77b4;
        color: #ffffff;
        font-size: 18px;
        border-radius: 5px;
        margin: 10px;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0b5fa8;
        transform: scale(1.05);
    }
    .stFileUploader label {
        font-size: 18px;
        color: #ffffff;
        animation: slideIn 1s ease-out;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        font-size: 18px;
        color: #ffffff;
        background-color: #1c2b3a;
        border-radius: 5px;
        animation: slideIn 1s ease-out;
    }
    .stTextInput div, .stNumberInput div, .stSelectbox div {
        font-size: 18px;
        color: #ffffff;
        animation: slideIn 1s ease-out;
    }
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    .stMarkdown {
        font-size: 18px;
        animation: fadeIn 1.5s ease-in;
    }
    .stDataFrame, .stTable {
        font-size: 18px;
        animation: fadeIn 1.5s ease-in;
    }
    .stApp {
        background-image: url('https://www.example.com/path/to/your/background.jpg');
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título de la aplicación
st.title("⚽ Análisis y Predicción de Jugadores de Fútbol de la Liga Boliviana Temporada 24 ⚽")

# Crear carpeta para gráficos si no existe
if not os.path.exists("Graficos"):
    os.makedirs("Graficos")

# Subir el archivo CSV
uploaded_file = st.file_uploader("📁 Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Datos del archivo subido:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"❌ Error al leer el archivo CSV: {e}")
        st.stop()

    # Selección de columna para análisis numérico
    columna_analisis = st.selectbox("📈 Selecciona la columna para análisis numérico", df.columns.tolist())

    # Diseño de columnas para filtrado
    columna_filtro = st.selectbox("🔍 Selecciona la columna para filtrar (opcional)", ["Ninguno"] + df.columns.tolist())
    valor_seleccionado_filtro = None
    if columna_filtro != "Ninguno":
        valores_filtro = df[columna_filtro].unique().tolist()
        valor_seleccionado_filtro = st.selectbox(f"Selecciona el valor para {columna_filtro}", valores_filtro)

    # Aplicar filtro
    df_filtrado = df.copy()
    if valor_seleccionado_filtro:
        df_filtrado = df_filtrado[df_filtrado[columna_filtro] == valor_seleccionado_filtro]
    
    # Construir el título del gráfico
    filtro_titulo = ""
    if valor_seleccionado_filtro:
        filtro_titulo = f" de {columna_filtro}: {valor_seleccionado_filtro}"
    
    # Verificar si df_filtrado está vacío
    if df_filtrado.empty:
        st.warning("⚠️ No se encontraron datos que coincidan con el filtro aplicado.")
    else:
        # Función para guardar gráfico y generar botón de descarga
        def guardar_y_descargar(fig, titulo):
            # Guardar figura
            filename = f"Graficos/{titulo}.png"
            fig.savefig(filename)
            # Mostrar botón de descarga
            with open(filename, "rb") as f:
                st.download_button("Descargar imagen", f, file_name=f"{titulo}.png", mime="image/png")

        # Botón para generar gráfico de barras
        if st.button("📊 Generar Gráfico de Barras"):
            st.write(f"📊 Gráfico de barras para {columna_analisis}{filtro_titulo}")
            fig, ax = plt.subplots()
            df_filtrado[columna_analisis].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Gráfico de barras para {columna_analisis}{filtro_titulo}", fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=7)
            st.pyplot(fig)
            guardar_y_descargar(fig, f"grafico_barras_{columna_analisis}{filtro_titulo}")
            st.text("El gráfico de barras muestra la distribución de los valores en la columna seleccionada. Cada barra representa un valor único en la columna y la altura de la barra indica la frecuencia de ese valor en el conjunto de datos.")

        # Botón para generar gráfico de pastel
        if st.button("🧁 Generar Gráfico de Pastel"):
            st.write(f"🧁 Gráfico de pastel para {columna_analisis}{filtro_titulo}")
            fig, ax = plt.subplots()
            df_filtrado[columna_analisis].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title(f"Gráfico de pastel para {columna_analisis}{filtro_titulo}", fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=7)
            st.pyplot(fig)
            guardar_y_descargar(fig, f"grafico_pastel_{columna_analisis}{filtro_titulo}")
            st.text("El gráfico de pastel muestra la proporción de cada valor en la columna seleccionada en relación al total de valores. Cada sección del pastel representa un valor único y el tamaño de la sección indica su proporción en el conjunto de datos.")

        # Botón para generar gráfico de histograma solo si la columna es numérica
        if pd.api.types.is_numeric_dtype(df_filtrado[columna_analisis]):
            if st.button("📉 Generar Gráfico de Histograma"):
                st.write(f"📉 Gráfico de histograma para {columna_analisis}{filtro_titulo}")
                fig, ax = plt.subplots()
                df_filtrado[columna_analisis].plot(kind='hist', ax=ax, bins=30)
                ax.set_title(f"Gráfico de histograma para {columna_analisis}{filtro_titulo}", fontsize=14)
                ax.tick_params(axis='both', which='major', labelsize=7)
                st.pyplot(fig)
                guardar_y_descargar(fig, f"grafico_histograma_{columna_analisis}{filtro_titulo}")
                st.text("El gráfico de histograma muestra la distribución de frecuencias de los valores en la columna seleccionada. Los datos se agrupan en intervalos llamados 'bins', y la altura de cada barra muestra la cantidad de datos que caen dentro de cada intervalo.")

        # Botón para calcular promedios solo si la columna es numérica
        if pd.api.types.is_numeric_dtype(df_filtrado[columna_analisis]):
            if st.button("📐 Generar Promedios"):
                promedio = df_filtrado[columna_analisis].mean()
                st.write(f"El promedio de {columna_analisis} es {promedio}")
                st.text("El promedio es una medida de tendencia central que indica el valor medio de los datos en la columna seleccionada.")

        # Generar gráficos adicionales según el tipo de columna
        st.write("📊 Opciones avanzadas de gráficos")
        
        # Gráfico de dispersión para columnas numéricas
        columnas_numericas = df_filtrado.select_dtypes(include=[np.number]).columns.tolist()
        if columnas_numericas:
            columna_x = st.selectbox("Selecciona la columna para el eje X", columnas_numericas)
            columna_y = st.selectbox("Selecciona la columna para el eje Y", columnas_numericas)
            if st.button("📉 Generar Gráfico de Dispersión"):
                st.write(f"📉 Gráfico de dispersión para {columna_x} vs {columna_y}{filtro_titulo}")
                fig, ax = plt.subplots()
                ax.scatter(df_filtrado[columna_x], df_filtrado[columna_y])
                ax.set_title(f"Gráfico de dispersión para {columna_x} vs {columna_y}{filtro_titulo}", fontsize=14)
                ax.set_xlabel(columna_x)
                ax.set_ylabel(columna_y)
                st.pyplot(fig)
                guardar_y_descargar(fig, f"grafico_dispersión_{columna_x}_vs_{columna_y}{filtro_titulo}")
                st.text("El gráfico de dispersión muestra la relación entre dos variables numéricas. Cada punto en el gráfico representa una observación en el conjunto de datos, y la posición del punto viene determinada por los valores de las dos variables seleccionadas.")

            # Gráfico de regresión lineal
            if st.button("📉 Generar Gráfico de Regresión Lineal"):
                st.write(f"📉 Gráfico de regresión lineal para {columna_x} vs {columna_y}{filtro_titulo}")
                X = df_filtrado[[columna_x]].values
                y = df_filtrado[columna_y].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, label="Datos reales")
                ax.plot(X_test, y_pred, color='red', label="Predicción")
                ax.set_title(f"Regresión lineal para {columna_x} vs {columna_y}{filtro_titulo}", fontsize=14)
                ax.set_xlabel(columna_x)
                ax.set_ylabel(columna_y)
                ax.legend()
                st.pyplot(fig)
                guardar_y_descargar(fig, f"grafico_regresion_lineal_{columna_x}_vs_{columna_y}{filtro_titulo}")

                st.write("📊 Métricas del modelo")
                st.write(f"R²: {r2_score(y_test, y_pred)}")
                st.write(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred)}")
                st.text("La regresión lineal es un método estadístico que se utiliza para modelar la relación entre una variable dependiente y una o más variables independientes. El coeficiente de determinación (R²) indica qué tan bien se ajustan los datos al modelo, y el error cuadrático medio (MSE) mide el promedio de los errores al cuadrado entre los valores reales y los valores predichos.")
