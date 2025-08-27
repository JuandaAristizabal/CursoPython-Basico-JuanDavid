"""
Script para generar datos sintéticos para los laboratorios de ML
================================================================
"""

import csv
import random
from datetime import datetime, timedelta

# Intentar importar pandas y numpy, si no están disponibles usar alternativas
try:
    import pandas as pd
    import numpy as np
    USE_PANDAS = True
    np.random.seed(42)
except ImportError:
    USE_PANDAS = False
    random.seed(42)
    print("⚠️  Pandas/NumPy no instalados. Generando datos con Python estándar...")
    print("   Para una mejor experiencia, instala: pip install pandas numpy")

# DATASET 1: Producción Histórica
# ================================
print("Generando datos de producción histórica...")

n_dias = 500
fechas = pd.date_range(end=datetime.now(), periods=n_dias, freq='D')

# Variables operacionales con correlaciones realistas
presion_base = 1500
temperatura_base = 180
dias_operacion = np.arange(1, n_dias + 1)

# Agregar variación y tendencias
presion_boca_psi = presion_base + np.random.normal(0, 100, n_dias) - dias_operacion * 0.1
temperatura_f = temperatura_base + np.random.normal(0, 15, n_dias) + np.sin(dias_operacion/30) * 10
choke_size = np.random.choice([24, 28, 32, 36, 40], n_dias, p=[0.1, 0.2, 0.4, 0.2, 0.1])

# Producción correlacionada con las variables
produccion_oil_bbl = (
    800 + 
    presion_boca_psi * 0.3 + 
    temperatura_f * 0.5 + 
    choke_size * 5 - 
    dias_operacion * 0.2 +
    np.random.normal(0, 50, n_dias)
)

# Asegurar valores positivos
produccion_oil_bbl = np.maximum(produccion_oil_bbl, 100)
presion_boca_psi = np.maximum(presion_boca_psi, 800)
temperatura_f = np.maximum(temperatura_f, 150)

df_produccion = pd.DataFrame({
    'fecha': fechas,
    'well_id': np.random.choice(['POZO-001', 'POZO-002', 'POZO-003'], n_dias),
    'presion_boca_psi': presion_boca_psi.round(1),
    'temperatura_f': temperatura_f.round(1),
    'dias_operacion': dias_operacion,
    'choke_size': choke_size,
    'produccion_oil_bbl': produccion_oil_bbl.round(1),
    'produccion_gas_mcf': (produccion_oil_bbl * 0.5 + np.random.normal(0, 20, n_dias)).round(1),
    'produccion_agua_bbl': (produccion_oil_bbl * 0.3 + np.random.normal(0, 10, n_dias)).round(1)
})

# Agregar algunos valores faltantes aleatorios
mask = np.random.random(df_produccion.shape) < 0.02
df_produccion = df_produccion.mask(mask)

df_produccion.to_csv('produccion_historica.csv', index=False)
print(f"✓ produccion_historica.csv generado ({len(df_produccion)} registros)")


# DATASET 2: Eventos Operacionales
# =================================
print("\nGenerando datos de eventos operacionales...")

n_eventos = 1000
fechas_eventos = pd.date_range(end=datetime.now(), periods=n_eventos, freq='6H')

tipos_evento = ['normal', 'mantenimiento_preventivo', 'falla_bomba', 'obstruccion', 
                'ajuste_parametros', 'fuga_menor', 'alarma_falsa']
probabilidades = [0.6, 0.15, 0.05, 0.05, 0.08, 0.04, 0.03]

tipos = np.random.choice(tipos_evento, n_eventos, p=probabilidades)

# Generar características correlacionadas con el tipo de evento
df_eventos = pd.DataFrame({
    'fecha': fechas_eventos,
    'well_id': np.random.choice(['POZO-001', 'POZO-002', 'POZO-003'], n_eventos),
    'tipo_evento': tipos,
    'duracion_horas': np.where(
        tipos == 'normal', 0,
        np.where(tipos == 'falla_bomba', np.random.uniform(4, 24, n_eventos),
        np.where(tipos == 'mantenimiento_preventivo', np.random.uniform(2, 8, n_eventos),
        np.random.uniform(0.5, 4, n_eventos)))
    ).round(1),
    'impacto_produccion_pct': np.where(
        tipos == 'normal', 0,
        np.where(tipos == 'falla_bomba', np.random.uniform(50, 100, n_eventos),
        np.where(tipos == 'obstruccion', np.random.uniform(20, 60, n_eventos),
        np.random.uniform(0, 20, n_eventos)))
    ).round(1),
    'costo_intervencion': np.where(
        tipos == 'normal', 0,
        np.where(tipos == 'falla_bomba', np.random.uniform(50000, 200000, n_eventos),
        np.where(tipos == 'mantenimiento_preventivo', np.random.uniform(5000, 20000, n_eventos),
        np.random.uniform(1000, 10000, n_eventos)))
    ).round(0)
})

df_eventos.to_csv('eventos_operacionales.csv', index=False)
print(f"✓ eventos_operacionales.csv generado ({len(df_eventos)} registros)")


# DATASET 3: Parámetros de Pozos
# ===============================
print("\nGenerando datos de parámetros de pozos...")

# Generar datos diarios con más variables
df_parametros = pd.DataFrame({
    'fecha': fechas,
    'well_id': np.random.choice(['POZO-001', 'POZO-002', 'POZO-003'], n_dias),
    'presion_fondo_psi': presion_base * 1.3 + np.random.normal(0, 150, n_dias),
    'presion_cabeza_psi': presion_boca_psi * 0.9 + np.random.normal(0, 50, n_dias),
    'temperatura_fondo_f': temperatura_base * 1.2 + np.random.normal(0, 20, n_dias),
    'flujo_total_bpd': produccion_oil_bbl * 1.5 + np.random.normal(0, 100, n_dias),
    'corte_agua_pct': 20 + dias_operacion * 0.02 + np.random.normal(0, 5, n_dias),
    'gor_scf_bbl': 500 + np.random.normal(0, 100, n_dias),
    'indice_productividad': 10 - dias_operacion * 0.005 + np.random.normal(0, 1, n_dias),
    'dias_desde_mantenimiento': np.random.randint(1, 180, n_dias),
    'horas_operacion_continua': np.random.randint(1, 720, n_dias)
})

# Asegurar valores positivos y límites realistas
df_parametros['corte_agua_pct'] = df_parametros['corte_agua_pct'].clip(0, 100)
df_parametros['indice_productividad'] = df_parametros['indice_productividad'].clip(1, 20)

df_parametros = df_parametros.round(2)
df_parametros.to_csv('parametros_pozos.csv', index=False)
print(f"✓ parametros_pozos.csv generado ({len(df_parametros)} registros)")


# DATASET 4: Anomalías en Sensores
# =================================
print("\nGenerando datos de anomalías en sensores...")

n_lecturas = 2000

# Generar datos normales (80%) y anómalos (20%)
es_anomalia = np.random.choice([0, 1], n_lecturas, p=[0.8, 0.2])

# Datos base normales
temperatura_normal = np.random.normal(180, 10, n_lecturas)
vibracion_normal = np.random.normal(2.5, 0.5, n_lecturas)
presion_normal = np.random.normal(1500, 100, n_lecturas)
ruido_normal = np.random.normal(85, 5, n_lecturas)

# Modificar para anomalías
temperatura = np.where(es_anomalia, 
                       temperatura_normal + np.random.choice([-50, 50], n_lecturas),
                       temperatura_normal)
vibracion = np.where(es_anomalia,
                     vibracion_normal * np.random.uniform(2, 5, n_lecturas),
                     vibracion_normal)
presion = np.where(es_anomalia,
                   presion_normal + np.random.choice([-500, 500], n_lecturas),
                   presion_normal)
ruido_db = np.where(es_anomalia,
                    ruido_normal + np.random.uniform(20, 40, n_lecturas),
                    ruido_normal)

df_anomalias = pd.DataFrame({
    'timestamp': pd.date_range(end=datetime.now(), periods=n_lecturas, freq='H'),
    'sensor_id': np.random.choice(['SENS-001', 'SENS-002', 'SENS-003', 'SENS-004'], n_lecturas),
    'temperatura': temperatura.round(1),
    'vibracion': vibracion.round(2),
    'presion': presion.round(1),
    'ruido_db': ruido_db.round(1),
    'es_anomalia': es_anomalia,
    'tipo_anomalia': np.where(
        es_anomalia == 0, 'normal',
        np.random.choice(['sobrecalentamiento', 'vibracion_excesiva', 'presion_anormal', 'falla_sensor'],
                        n_lecturas, p=[0.3, 0.3, 0.2, 0.2])
    )
})

# Asegurar valores positivos
df_anomalias['temperatura'] = df_anomalias['temperatura'].clip(100, 300)
df_anomalias['vibracion'] = df_anomalias['vibracion'].clip(0.1, 20)
df_anomalias['presion'] = df_anomalias['presion'].clip(500, 3000)
df_anomalias['ruido_db'] = df_anomalias['ruido_db'].clip(60, 150)

df_anomalias.to_csv('anomalias_sensores.csv', index=False)
print(f"✓ anomalias_sensores.csv generado ({len(df_anomalias)} registros)")


# RESUMEN
# =======
print("\n" + "="*50)
print("RESUMEN DE DATASETS GENERADOS:")
print("="*50)
print(f"1. produccion_historica.csv: {n_dias} días de datos")
print(f"2. eventos_operacionales.csv: {n_eventos} eventos registrados")
print(f"3. parametros_pozos.csv: {n_dias} días de parámetros")
print(f"4. anomalias_sensores.csv: {n_lecturas} lecturas de sensores")
print(f"\nDistribución de anomalías: {(es_anomalia.sum()/len(es_anomalia)*100):.1f}%")
print(f"Distribución de eventos críticos: {((df_eventos['tipo_evento'].isin(['falla_bomba', 'obstruccion', 'fuga_menor'])).sum()/len(df_eventos)*100):.1f}%")
print("\n✓ Todos los datasets han sido generados exitosamente")