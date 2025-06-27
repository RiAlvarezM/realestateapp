import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib
import json
import sys

# 1. Función personalizada para convertir columnas a minúsculas
def to_lowercase(dataframe):
    return dataframe.apply(lambda x: x.str.lower() if x.dtype == "object" else x)


with open('model_config.json', 'r', encoding='utf-8') as archivo:
    # 3. Usar json.load() para convertir el contenido del archivo en un diccionario de Python
    datos = json.load(archivo)

    # Acceder a las listas y al valor por su clave
    locations = datos['locations']
    buildings = datos['buildings']
    rmse_train = datos['average_rmse']

st.set_page_config(
    page_title="Predicción de Precios de Propiedades",
    page_icon="🏠",
    layout="wide", # "wide" para más espacio, "centered" para un look más compacto
    initial_sidebar_state="expanded"
)

model_pipeline = joblib.load('real_estate_model_pipeline.pkl')
st.sidebar.success("Modelo cargado exitosamente.")

st.write("---") # Separador
st.subheader("Detalles de la Propiedad")

transaction_type_input = 'Venta'

st.write("") # Espacio
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Características Principales")
    # Característica numérica: 'bedroom'
    bedroom_input = st.number_input(
        "Habitaciones (bedroom)",
        min_value=0, max_value=10, value=2, step=1,
        help="Número de habitaciones."
    )

    # Característica numérica: 'size'
    size_input = st.number_input(
        "Superficie en m² (size)",
        min_value=0, value=120, step=10,
        help="Superficie total en metros cuadrados."
    )

    # Característica numérica: 'parking_spaces'
    parking_spaces_input = st.number_input(
        "Estacionamientos (parking_spaces)",
        min_value=0, max_value=10, value=1, step=1,
        help="Cantidad de espacios de estacionamiento."
    )

    photos_input = st.radio("¿Tiene fotos? (photos)", ('sí', 'no'), horizontal=True)

with col2:
    st.markdown("#### Ubicación y Detalles")
    # Característica categórica: 'location'
    available_locations = locations#['San Francisco', 'Costa del Este', 'Punta Pacífica', 'Bella Vista', 'Obarrio']
    location_input = st.selectbox(
        "Zona (location)",
        available_locations,
        index=0,
        help="Ubicación geográfica de la propiedad."
    )

    # Característica categórica: 'building'
    available_buildings = buildings#['San Francisco', 'Costa del Este', 'Punta Pacífica', 'Bella Vista', 'Obarrio']
    building_input = st.selectbox(
        "Edificios",
        available_buildings,
        index=0,
        help="Edificios."
    )


    # Características categóricas adicionales
    st.markdown("#### Otros Atributos")
    
    den_input = st.radio("¿Tiene den/estudio? (den)", ('sí', 'no'), horizontal=True)
    commercial_input = st.radio("¿Es de uso comercial? (commercial)", ('sí', 'no'), horizontal=True)

st.write("---")

button_label = "Calcular Precio de Venta" if transaction_type_input == 'Venta' else "Calcular Renta Estimada"
if st.button(button_label):
    # Creamos un diccionario para asegurar que los nombres de las columnas son correctos.
    input_data = {
        'photos': [photos_input],
        'location': [location_input],
        'building': [building_input],
        'den': [den_input],
        'commercial': [commercial_input],
        'bedroom': [bedroom_input],
        'size': [size_input],
        'parking_spaces': [parking_spaces_input]
    }
    input_df = pd.DataFrame(input_data)

    # Es buena práctica aplicar la misma transformación de minúsculas que en el entrenamiento.
    input_df = to_lowercase(input_df)

    # Realizar la predicción
    predicted_price = model_pipeline.predict(input_df)[0]

    # Mostrar el resultado de forma destacada.
    st.success(f"El precio estimado es: **${predicted_price:,.2f}**")

    confidence_level = 0.95
    alpha = 1 - confidence_level

    t_value = 1.96 

    margin_of_error = t_value * rmse_train
    lower_bound = predicted_price - margin_of_error
    upper_bound = predicted_price + margin_of_error

    st.info(f"Con un 95% de confianza, el precio se encuentra entre {lower_bound:,.2f} y {upper_bound:,.2f}.")
