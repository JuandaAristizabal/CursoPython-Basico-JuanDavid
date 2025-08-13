"""
Script para generar datasets con problemas típicos de calidad
para la industria petrolera - Sesión 14
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

def generar_produccion_diaria():
    """
    Genera datos de producción diaria con valores faltantes típicos
    """
    fechas = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_dias = len(fechas)
    
    # Generar datos base
    datos = {
        'fecha': fechas,
        'pozo_id': np.random.choice(['POZO-001', 'POZO-002', 'POZO-003', 'POZO-004'], n_dias),
        'produccion_oil_bbl': np.random.normal(500, 50, n_dias),
        'produccion_gas_mcf': np.random.normal(2000, 200, n_dias),
        'produccion_agua_bbl': np.random.normal(100, 20, n_dias),
        'presion_boca_psi': np.random.normal(1500, 100, n_dias),
        'temperatura_f': np.random.normal(180, 10, n_dias),
        'horas_operacion': np.random.uniform(20, 24, n_dias)
    }
    
    df = pd.DataFrame(datos)
    
    # Introducir valores faltantes de forma realista
    # Fallas de sensor - valores faltantes consecutivos
    for i in range(10):
        inicio = np.random.randint(0, n_dias - 5)
        duracion = np.random.randint(1, 4)
        columna = np.random.choice(['presion_boca_psi', 'temperatura_f'])
        df.loc[inicio:inicio+duracion, columna] = np.nan
    
    # Valores faltantes aleatorios (5% de los datos)
    for col in ['produccion_oil_bbl', 'produccion_gas_mcf', 'produccion_agua_bbl']:
        indices_nan = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[indices_nan, col] = np.nan
    
    # Algunos valores negativos incorrectos
    indices_negativos = np.random.choice(df.index, size=5, replace=False)
    df.loc[indices_negativos, 'produccion_agua_bbl'] = -abs(df.loc[indices_negativos, 'produccion_agua_bbl'])
    
    # Valores cero cuando el pozo está cerrado
    indices_cero = np.random.choice(df.index, size=15, replace=False)
    df.loc[indices_cero, ['produccion_oil_bbl', 'produccion_gas_mcf', 'horas_operacion']] = 0
    
    # Redondear valores apropiadamente
    df['produccion_oil_bbl'] = df['produccion_oil_bbl'].round(1)
    df['produccion_gas_mcf'] = df['produccion_gas_mcf'].round(0)
    df['produccion_agua_bbl'] = df['produccion_agua_bbl'].round(1)
    df['presion_boca_psi'] = df['presion_boca_psi'].round(0)
    df['temperatura_f'] = df['temperatura_f'].round(1)
    df['horas_operacion'] = df['horas_operacion'].round(2)
    
    return df

def generar_sensores_pozos():
    """
    Genera datos de sensores con outliers y valores anómalos
    """
    # Generar timestamps cada hora por 30 días
    fechas = pd.date_range(start='2023-11-01', end='2023-11-30', freq='H')
    n_registros = len(fechas)
    
    datos = {
        'timestamp': fechas,
        'sensor_id': np.random.choice(['SENS-A1', 'SENS-A2', 'SENS-B1', 'SENS-B2'], n_registros),
        'presion_fondo_psi': np.random.normal(3000, 150, n_registros),
        'temperatura_fondo_f': np.random.normal(250, 15, n_registros),
        'caudal_instantaneo_bbl_h': np.random.normal(20, 3, n_registros),
        'vibracion_hz': np.random.normal(50, 5, n_registros),
        'nivel_tanque_ft': np.random.normal(15, 2, n_registros)
    }
    
    df = pd.DataFrame(datos)
    
    # Agregar outliers extremos (picos de sensor)
    n_outliers = 20
    indices_outliers = np.random.choice(df.index, size=n_outliers, replace=False)
    
    # Outliers en presión (valores imposibles)
    df.loc[indices_outliers[:5], 'presion_fondo_psi'] = np.random.uniform(5000, 10000, 5)
    
    # Outliers en temperatura (picos de sensor)
    df.loc[indices_outliers[5:10], 'temperatura_fondo_f'] = np.random.uniform(400, 500, 5)
    
    # Valores negativos incorrectos en caudal
    df.loc[indices_outliers[10:15], 'caudal_instantaneo_bbl_h'] = np.random.uniform(-50, -10, 5)
    
    # Vibraciones anormales (posible falla mecánica)
    df.loc[indices_outliers[15:], 'vibracion_hz'] = np.random.uniform(150, 300, 5)
    
    # Agregar ruido de medición
    ruido_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[ruido_indices, 'nivel_tanque_ft'] += np.random.normal(0, 5, 50)
    
    # Algunos valores faltantes
    for col in df.columns[2:]:
        indices_nan = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[indices_nan, col] = np.nan
    
    # Redondear apropiadamente
    for col in ['presion_fondo_psi', 'temperatura_fondo_f', 'vibracion_hz']:
        df[col] = df[col].round(1)
    df['caudal_instantaneo_bbl_h'] = df['caudal_instantaneo_bbl_h'].round(2)
    df['nivel_tanque_ft'] = df['nivel_tanque_ft'].round(2)
    
    return df

def generar_mantenimiento():
    """
    Genera registros de mantenimiento con inconsistencias y errores de entrada
    """
    tipos_mantenimiento = ['Preventivo', 'Correctivo', 'Predictivo', 'Emergencia']
    equipos = ['Bomba', 'Válvula', 'Compresor', 'Separador', 'Motor']
    tecnicos = ['Juan Pérez', 'Maria Garcia', 'Carlos Lopez', 'Ana Martinez', 'Pedro Rodriguez']
    
    registros = []
    fecha_inicio = datetime(2023, 1, 1)
    
    for i in range(200):
        fecha = fecha_inicio + timedelta(days=np.random.randint(0, 365))
        
        registro = {
            'id_mantenimiento': f'MNT-{i+1:04d}',
            'fecha': fecha,
            'equipo': random.choice(equipos),
            'tipo_mantenimiento': random.choice(tipos_mantenimiento),
            'horas_trabajo': np.random.uniform(1, 8),
            'costo_usd': np.random.uniform(100, 5000),
            'tecnico_responsable': random.choice(tecnicos),
            'descripcion': f'Mantenimiento rutinario del equipo',
            'pozo_id': random.choice(['POZO-001', 'POZO-002', 'POZO-003', 'POZO-004'])
        }
        registros.append(registro)
    
    df = pd.DataFrame(registros)
    
    # Introducir inconsistencias
    
    # Duplicados
    duplicados_idx = np.random.choice(df.index, size=5, replace=False)
    df = pd.concat([df, df.loc[duplicados_idx]], ignore_index=True)
    
    # Errores de tipeo en nombres
    df.loc[10, 'tecnico_responsable'] = 'Juan Perez'  # Sin tilde
    df.loc[20, 'tecnico_responsable'] = 'Maria  Garcia'  # Doble espacio
    df.loc[30, 'tecnico_responsable'] = 'carlos lopez'  # Minúsculas
    
    # Inconsistencias en tipos de mantenimiento
    df.loc[15, 'tipo_mantenimiento'] = 'preventivo'  # Minúscula
    df.loc[25, 'tipo_mantenimiento'] = 'CORRECTIVO'  # Mayúscula
    df.loc[35, 'tipo_mantenimiento'] = 'Pred.'  # Abreviado
    
    # Fechas futuras (error de entrada)
    df.loc[40, 'fecha'] = datetime(2024, 6, 15)
    df.loc[41, 'fecha'] = datetime(2025, 1, 1)
    
    # Valores faltantes
    df.loc[np.random.choice(df.index, 10), 'descripcion'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'costo_usd'] = np.nan
    
    # Valores negativos incorrectos
    df.loc[50, 'horas_trabajo'] = -3
    df.loc[51, 'costo_usd'] = -500
    
    # Valores extremos poco probables
    df.loc[60, 'horas_trabajo'] = 48  # 2 días seguidos
    df.loc[61, 'costo_usd'] = 50000  # Costo muy alto
    
    # Mezclar el orden para simular entrada real
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Redondear valores
    df['horas_trabajo'] = df['horas_trabajo'].round(1)
    df['costo_usd'] = df['costo_usd'].round(2)
    
    return df

def main():
    """
    Genera todos los datasets y los guarda en archivos CSV
    """
    print("Generando datasets con problemas de calidad...")
    
    # Generar datasets
    df_produccion = generar_produccion_diaria()
    df_sensores = generar_sensores_pozos()
    df_mantenimiento = generar_mantenimiento()
    
    # Guardar en archivos CSV
    df_produccion.to_csv('produccion_diaria.csv', index=False)
    print(f"✓ produccion_diaria.csv - {len(df_produccion)} registros")
    print(f"  Valores faltantes: {df_produccion.isnull().sum().sum()} total")
    
    df_sensores.to_csv('sensores_pozos.csv', index=False)
    print(f"✓ sensores_pozos.csv - {len(df_sensores)} registros")
    print(f"  Outliers introducidos: ~20 valores extremos")
    
    df_mantenimiento.to_csv('mantenimiento.csv', index=False)
    print(f"✓ mantenimiento.csv - {len(df_mantenimiento)} registros")
    print(f"  Duplicados e inconsistencias introducidas")
    
    print("\n¡Datasets generados exitosamente!")
    print("Los archivos contienen problemas típicos de calidad para práctica.")

if __name__ == "__main__":
    main()