"""
Laboratorio 01: Exploración y Diagnóstico de Calidad de Datos
==============================================================

Objetivo: Aprender a identificar problemas comunes de calidad en datos
          de la industria petrolera.

Tareas:
1. Cargar y explorar el dataset de producción diaria
2. Identificar y cuantificar valores faltantes
3. Detectar valores anómalos o imposibles
4. Generar un reporte básico de calidad
5. Visualizar los patrones de problemas encontrados

Tiempo estimado: 30 minutos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("LABORATORIO 01: EXPLORACIÓN Y DIAGNÓSTICO DE CALIDAD")
print("=" * 55)

# TAREA 1: Cargar el dataset
# ==========================
print("\nTAREA 1: Cargar datos de producción")
print("-" * 40)

# TODO: Cargar el archivo 'produccion_diaria.csv' desde la carpeta datos
df = None  # Reemplazar con pd.read_csv(...)

# TODO: Mostrar información básica del dataset
# - Número de filas y columnas
# - Primeras 5 filas
# - Tipos de datos de cada columna


# TAREA 2: Análisis de valores faltantes
# =======================================
print("\nTAREA 2: Identificar valores faltantes")
print("-" * 40)

# TODO: Calcular el número de valores faltantes por columna
valores_faltantes = None  # Usar df.isnull().sum()

# TODO: Calcular el porcentaje de valores faltantes por columna
porcentaje_faltantes = None  # (valores_faltantes / len(df)) * 100

# TODO: Mostrar solo las columnas que tienen valores faltantes
print("\nColumnas con valores faltantes:")
# Iterar sobre las columnas y mostrar solo las que tienen NaN


# TAREA 3: Detectar valores anómalos
# ===================================
print("\nTAREA 3: Detectar valores anómalos o imposibles")
print("-" * 40)

# TODO: Identificar valores negativos en columnas de producción
# Las columnas de producción no pueden tener valores negativos
columnas_produccion = ['produccion_oil_bbl', 'produccion_gas_mcf', 'produccion_agua_bbl']

print("\nValores negativos en producción:")
for col in columnas_produccion:
    # TODO: Contar valores negativos en cada columna
    pass

# TODO: Identificar valores fuera de rango operacional normal
# Presión normal: 1000-2000 psi
# Temperatura normal: 150-210 °F
print("\nValores fuera de rango operacional:")
# Verificar presion_boca_psi y temperatura_f


# TAREA 4: Análisis estadístico básico
# =====================================
print("\nTAREA 4: Estadísticas descriptivas")
print("-" * 40)

# TODO: Calcular estadísticas básicas para columnas numéricas
# Usar df.describe()

# TODO: Identificar la variabilidad de cada columna
# Calcular el coeficiente de variación (CV = std/mean * 100)
print("\nCoeficiente de variación por columna:")
for col in columnas_produccion:
    # TODO: Calcular CV para cada columna de producción
    pass


# TAREA 5: Visualización de problemas de calidad
# ===============================================
print("\nTAREA 5: Visualizar problemas de calidad")
print("-" * 40)

# TODO: Crear visualizaciones para entender los problemas
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Heatmap de valores faltantes
ax1 = axes[0, 0]
# TODO: Crear un heatmap mostrando dónde están los valores faltantes
# Pista: Usar sns.heatmap con df.isnull()

# Subplot 2: Barplot de porcentaje de valores faltantes
ax2 = axes[0, 1]
# TODO: Crear un gráfico de barras con el porcentaje de NaN por columna

# Subplot 3: Boxplot de producción de oil
ax3 = axes[1, 0]
# TODO: Crear un boxplot para identificar outliers en produccion_oil_bbl

# Subplot 4: Histograma de presión
ax4 = axes[1, 1]
# TODO: Crear un histograma de presion_boca_psi

plt.tight_layout()
plt.savefig('exploracion_calidad.png')
print("Visualización guardada como 'exploracion_calidad.png'")


# TAREA 6: Generar reporte de calidad
# ====================================
print("\nTAREA 6: Reporte de calidad de datos")
print("-" * 40)

def generar_reporte_calidad(dataframe):
    """
    Genera un reporte básico de calidad de datos
    
    Parámetros:
    -----------
    dataframe : pd.DataFrame
        DataFrame a analizar
    
    Retorna:
    --------
    dict : Diccionario con métricas de calidad
    """
    reporte = {
        'total_registros': len(dataframe),
        'total_columnas': len(dataframe.columns),
        'valores_faltantes_total': 0,  # TODO: Calcular total de NaN
        'completitud_porcentaje': 0,   # TODO: Calcular % de datos completos
        'columnas_problematicas': [],  # TODO: Listar columnas con > 5% NaN
        'tiene_duplicados': False,     # TODO: Verificar si hay duplicados
        'tiene_valores_negativos': False  # TODO: Verificar valores negativos
    }
    
    # TODO: Completar el cálculo de métricas
    
    return reporte

# TODO: Generar y mostrar el reporte
reporte = generar_reporte_calidad(df)
print("\nREPORTE DE CALIDAD:")
print("=" * 40)
# Mostrar cada métrica del reporte


# PREGUNTA DE REFLEXIÓN
# =====================
print("\n" + "=" * 55)
print("PREGUNTA DE REFLEXIÓN:")
print("=" * 55)
print("""
Basándote en tu análisis, responde:

1. ¿Cuáles son los 3 principales problemas de calidad en este dataset?
2. ¿Qué columnas requieren atención prioritaria?
3. ¿Qué estrategia de limpieza recomendarías para cada problema?

Escribe tus respuestas como comentarios aquí:
""")

# Tu respuesta:
# 1. Principales problemas:
#    - 
#    - 
#    - 
#
# 2. Columnas prioritarias:
#    - 
#    - 
#
# 3. Estrategias recomendadas:
#    - 
#    - 
#    - 

print("\n¡Laboratorio completado!")
print("Continúa con lab_02_imputacion_basica.py")