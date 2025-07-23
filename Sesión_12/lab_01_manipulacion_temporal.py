"""
LABORATORIO 1: MANIPULACIÓN TEMPORAL BÁSICA
Sesión 12: Series Temporales en Pandas

Objetivo: Dominar la manipulación de datos temporales con DatetimeIndex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuración para mejor visualización
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("=" * 80)
print("LABORATORIO 1: MANIPULACIÓN TEMPORAL BÁSICA")
print("=" * 80)

# =============================================================================
# EJERCICIO 1: CARGAR Y PREPARAR DATOS TEMPORALES
# =============================================================================

print("\n1. CARGANDO DATOS DE PRODUCCIÓN HISTÓRICA...")
print("-" * 50)

# Cargar datos de producción

print(f"Forma del dataset: {df_produccion.shape}")
print(f"Columnas disponibles: {list(df_produccion.columns)}")
print(f"Tipos de datos:\n{df_produccion.dtypes}")

# Mostrar las primeras filas
print("\nPrimeras 5 filas:")

# =============================================================================
# EJERCICIO 2: CONVERTIR A DATETIMEINDEX
# =============================================================================

print("\n\n2. CONVERTIENDO A DATETIMEINDEX...")
print("-" * 50)

# Convertir la columna fecha a datetime

# Establecer fecha como índice

print("Índice temporal creado:")
print(f"Tipo de índice: {type(df_produccion.index)}")
print(f"Rango temporal: {df_produccion.index.min()} a {df_produccion.index.max()}")
print(f"Frecuencia de datos: {df_produccion.index.freq}")

# Mostrar información del DataFrame
print("\nInformación del DataFrame:")

# =============================================================================
# EJERCICIO 3: SELECCIÓN TEMPORAL BÁSICA
# =============================================================================

print("\n\n3. SELECCIÓN TEMPORAL BÁSICA...")
print("-" * 50)

# Seleccionar datos de una fecha específica
print("Datos del 15 de enero de 2023:")

print(fecha_especifica)

# Seleccionar rango de fechas
print("\nDatos del 10 al 15 de enero:")
print(f"Registros en el rango: {len(rango_fechas)}")
print(rango_fechas.head())

# Seleccionar por mes
print("\nDatos de enero de 2023:")
print(f"Registros en enero: {len(datos_enero)}")
print(datos_enero.head())

# =============================================================================
# EJERCICIO 4: OPERACIONES TEMPORALES AVANZADAS
# =============================================================================

print("\n\n4. OPERACIONES TEMPORALES AVANZADAS...")
print("-" * 50)

# Obtener componentes temporales

print("Componentes temporales agregados:")
print(df_produccion[['año', 'mes', 'dia', 'hora', 'dia_semana', 'produccion_bpd']].head(10))

# Calcular estadísticas por día de la semana
print("\nProducción promedio por día de la semana:")
print(produccion_por_dia)

# Calcular estadísticas por hora del día
print("\nProducción promedio por hora del día:")
print(produccion_por_hora)

# =============================================================================
# EJERCICIO 5: FILTRADO TEMPORAL CONDICIONAL
# =============================================================================

print("\n\n5. FILTRADO TEMPORAL CONDICIONAL...")
print("-" * 50)







# Filtrar datos de la primera semana
print(f"Datos de la primera semana: {len(primera_semana)} registros")

# Filtrar datos de las horas de mayor producción (6:00 y 12:00)
print(f"Datos de horas pico (6:00 y 12:00): {len(horas_pico)} registros")

# Filtrar datos de los fines de semana (sábado=5, domingo=6)
print(f"Datos de fines de semana: {len(fines_semana)} registros")

# =============================================================================
# EJERCICIO 6: ANÁLISIS DE TENDENCIAS TEMPORALES
# =============================================================================

print("\n\n6. ANÁLISIS DE TENDENCIAS TEMPORALES...")
print("-" * 50)

# Calcular producción total por día
print("Producción total por día:")
print(produccion_diaria.head(10))

# Calcular producción promedio por día
print("\nProducción promedio por día:")
print(produccion_promedio_diaria.head(10))

# Calcular estadísticas por semana
print("\nEstadísticas semanales de producción:")
print(produccion_semanal)

# =============================================================================
# EJERCICIO 7: VISUALIZACIÓN TEMPORAL BÁSICA
# =============================================================================

print("\n\n7. VISUALIZACIÓN TEMPORAL BÁSICA...")
print("-" * 50)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis Temporal de Producción Petrolera', fontsize=16)

# Gráfico 1: Producción a lo largo del tiempo
axes[0, 0].plot(df_produccion.index, df_produccion['produccion_bpd'], 'b-', linewidth=1)
axes[0, 0].set_title('Producción vs Tiempo')
axes[0, 0].set_xlabel('Fecha')
axes[0, 0].set_ylabel('Producción (BPD)')
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Producción por hora del día
produccion_por_hora_plot = df_produccion.groupby('hora')['produccion_bpd'].mean()
axes[0, 1].bar(produccion_por_hora_plot.index, produccion_por_hora_plot.values, color='green', alpha=0.7)
axes[0, 1].set_title('Producción Promedio por Hora del Día')
axes[0, 1].set_xlabel('Hora del Día')
axes[0, 1].set_ylabel('Producción Promedio (BPD)')
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Producción por día de la semana
produccion_por_dia_plot = df_produccion.groupby('dia_semana')['produccion_bpd'].mean()
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
axes[1, 0].bar(range(7), produccion_por_dia_plot.values, color='orange', alpha=0.7)
axes[1, 0].set_title('Producción Promedio por Día de la Semana')
axes[1, 0].set_xlabel('Día de la Semana')
axes[1, 0].set_ylabel('Producción Promedio (BPD)')
axes[1, 0].set_xticks(range(7))
axes[1, 0].set_xticklabels(dias_semana, rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Producción diaria agregada
axes[1, 1].plot(produccion_diaria.index, produccion_diaria.values, 'r-', linewidth=2)
axes[1, 1].set_title('Producción Total Diaria')
axes[1, 1].set_xlabel('Fecha')
axes[1, 1].set_ylabel('Producción Total (BPD)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# EJERCICIO 8: ANÁLISIS DE PATRONES TEMPORALES
# =============================================================================

print("\n\n8. ANÁLISIS DE PATRONES TEMPORALES...")
print("-" * 50)

# Calcular diferencias temporales
df_produccion['cambio_produccion'] = df_produccion['produccion_bpd'].diff()
df_produccion['cambio_porcentual'] = df_produccion['produccion_bpd'].pct_change() * 100

print("Estadísticas de cambios en producción:")
print(f"Cambio promedio: {df_produccion['cambio_produccion'].mean():.2f} BPD")
print(f"Cambio máximo: {df_produccion['cambio_produccion'].max():.2f} BPD")
print(f"Cambio mínimo: {df_produccion['cambio_produccion'].min():.2f} BPD")
print(f"Cambio porcentual promedio: {df_produccion['cambio_porcentual'].mean():.2f}%")

# Identificar los mayores cambios
print("\nTop 5 mayores incrementos en producción:")
mayores_incrementos = df_produccion.nlargest(5, 'cambio_produccion')[['produccion_bpd', 'cambio_produccion', 'cambio_porcentual']]
print(mayores_incrementos)

print("\nTop 5 mayores decrementos en producción:")
mayores_decrementos = df_produccion.nsmallest(5, 'cambio_produccion')[['produccion_bpd', 'cambio_produccion', 'cambio_porcentual']]
print(mayores_decrementos)

# =============================================================================
# EJERCICIO 9: RESUMEN Y CONCLUSIONES
# =============================================================================

print("\n\n9. RESUMEN Y CONCLUSIONES...")
print("-" * 50)

print("RESUMEN DEL ANÁLISIS TEMPORAL:")
print(f"• Período analizado: {df_produccion.index.min()} a {df_produccion.index.max()}")
print(f"• Total de registros: {len(df_produccion)}")
print(f"• Producción promedio: {df_produccion['produccion_bpd'].mean():.2f} BPD")
print(f"• Producción máxima: {df_produccion['produccion_bpd'].max():.2f} BPD")
print(f"• Producción mínima: {df_produccion['produccion_bpd'].min():.2f} BPD")

# Análisis de tendencia
produccion_inicial = df_produccion.iloc[0]['produccion_bpd']
produccion_final = df_produccion.iloc[-1]['produccion_bpd']
tendencia = ((produccion_final - produccion_inicial) / produccion_inicial) * 100

print(f"• Tendencia general: {tendencia:.2f}%")
if tendencia > 0:
    print("  → Tendencia POSITIVA (aumento en producción)")
else:
    print("  → Tendencia NEGATIVA (disminución en producción)")

print("\nPATRONES IDENTIFICADOS:")
print("• Variación diurna: Producción varía según la hora del día")
print("• Variación semanal: Diferentes patrones entre días laborales y fines de semana")
print("• Tendencia temporal: Cambios graduales en la producción a lo largo del tiempo")

print("\n" + "=" * 80)
print("LABORATORIO 1 COMPLETADO")
print("=" * 80) 