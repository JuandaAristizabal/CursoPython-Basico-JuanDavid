"""
Script simplificado para generar datos sintéticos sin dependencias externas
===========================================================================
"""

import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

def generar_fecha_str(dias_atras):
    """Genera una fecha como string"""
    fecha = datetime.now() - timedelta(days=dias_atras)
    return fecha.strftime("%Y-%m-%d")

def normal_random(mean, std):
    """Aproximación de distribución normal usando método Box-Muller"""
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mean + std * z0

# DATASET 1: Producción Histórica
print("Generando datos de producción histórica...")

n_dias = 500
pozos = ['POZO-001', 'POZO-002', 'POZO-003']
choke_sizes = [24, 28, 32, 36, 40]

with open('produccion_historica.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fecha', 'well_id', 'presion_boca_psi', 'temperatura_f', 
                    'dias_operacion', 'choke_size', 'produccion_oil_bbl',
                    'produccion_gas_mcf', 'produccion_agua_bbl'])
    
    for i in range(n_dias):
        fecha = generar_fecha_str(n_dias - i - 1)
        well_id = random.choice(pozos)
        
        # Variables con tendencias y correlaciones
        presion = max(800, 1500 + normal_random(0, 100) - i * 0.1)
        temperatura = max(150, 180 + normal_random(0, 15) + math.sin(i/30) * 10)
        choke = random.choices(choke_sizes, weights=[0.1, 0.2, 0.4, 0.2, 0.1])[0]
        
        # Producción correlacionada
        produccion = max(100, 800 + presion * 0.3 + temperatura * 0.5 + 
                        choke * 5 - i * 0.2 + normal_random(0, 50))
        
        # Agregar ocasionalmente valores vacíos (2% de probabilidad)
        if random.random() < 0.02:
            presion = ''
        if random.random() < 0.02:
            temperatura = ''
            
        writer.writerow([
            fecha,
            well_id,
            round(presion, 1) if presion else '',
            round(temperatura, 1) if temperatura else '',
            i + 1,
            choke,
            round(produccion, 1),
            round(produccion * 0.5 + normal_random(0, 20), 1),
            round(produccion * 0.3 + normal_random(0, 10), 1)
        ])

print(f"✓ produccion_historica.csv generado ({n_dias} registros)")

# DATASET 2: Eventos Operacionales
print("\nGenerando datos de eventos operacionales...")

n_eventos = 1000
tipos_evento = [
    ('normal', 0.6),
    ('mantenimiento_preventivo', 0.15),
    ('falla_bomba', 0.05),
    ('obstruccion', 0.05),
    ('ajuste_parametros', 0.08),
    ('fuga_menor', 0.04),
    ('alarma_falsa', 0.03)
]

with open('eventos_operacionales.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fecha', 'well_id', 'tipo_evento', 'duracion_horas',
                    'impacto_produccion_pct', 'costo_intervencion'])
    
    for i in range(n_eventos):
        fecha = (datetime.now() - timedelta(hours=(n_eventos - i - 1) * 6)).strftime("%Y-%m-%d %H:%M:%S")
        well_id = random.choice(pozos)
        
        # Seleccionar tipo de evento según probabilidades
        tipo = random.choices([t[0] for t in tipos_evento], 
                             weights=[t[1] for t in tipos_evento])[0]
        
        # Generar características según el tipo
        if tipo == 'normal':
            duracion = 0
            impacto = 0
            costo = 0
        elif tipo == 'falla_bomba':
            duracion = random.uniform(4, 24)
            impacto = random.uniform(50, 100)
            costo = random.uniform(50000, 200000)
        elif tipo == 'mantenimiento_preventivo':
            duracion = random.uniform(2, 8)
            impacto = random.uniform(5, 20)
            costo = random.uniform(5000, 20000)
        elif tipo == 'obstruccion':
            duracion = random.uniform(2, 12)
            impacto = random.uniform(20, 60)
            costo = random.uniform(10000, 50000)
        else:
            duracion = random.uniform(0.5, 4)
            impacto = random.uniform(0, 20)
            costo = random.uniform(1000, 10000)
        
        writer.writerow([
            fecha,
            well_id,
            tipo,
            round(duracion, 1),
            round(impacto, 1),
            round(costo, 0)
        ])

print(f"✓ eventos_operacionales.csv generado ({n_eventos} registros)")

# DATASET 3: Parámetros de Pozos
print("\nGenerando datos de parámetros de pozos...")

with open('parametros_pozos.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fecha', 'well_id', 'presion_fondo_psi', 'presion_cabeza_psi',
                    'temperatura_fondo_f', 'flujo_total_bpd', 'corte_agua_pct',
                    'gor_scf_bbl', 'indice_productividad', 'dias_desde_mantenimiento',
                    'horas_operacion_continua'])
    
    for i in range(n_dias):
        fecha = generar_fecha_str(n_dias - i - 1)
        well_id = random.choice(pozos)
        
        presion_fondo = 1950 + normal_random(0, 150)
        presion_cabeza = 1350 + normal_random(0, 50)
        temp_fondo = 216 + normal_random(0, 20)
        flujo = 1200 + normal_random(0, 100)
        corte_agua = max(0, min(100, 20 + i * 0.02 + normal_random(0, 5)))
        gor = 500 + normal_random(0, 100)
        ip = max(1, min(20, 10 - i * 0.005 + normal_random(0, 1)))
        
        writer.writerow([
            fecha,
            well_id,
            round(presion_fondo, 2),
            round(presion_cabeza, 2),
            round(temp_fondo, 2),
            round(flujo, 2),
            round(corte_agua, 2),
            round(gor, 2),
            round(ip, 2),
            random.randint(1, 180),
            random.randint(1, 720)
        ])

print(f"✓ parametros_pozos.csv generado ({n_dias} registros)")

# DATASET 4: Anomalías en Sensores
print("\nGenerando datos de anomalías en sensores...")

n_lecturas = 2000
sensores = ['SENS-001', 'SENS-002', 'SENS-003', 'SENS-004']
tipos_anomalia = ['sobrecalentamiento', 'vibracion_excesiva', 'presion_anormal', 'falla_sensor']

with open('anomalias_sensores.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'sensor_id', 'temperatura', 'vibracion',
                    'presion', 'ruido_db', 'es_anomalia', 'tipo_anomalia'])
    
    anomalias_count = 0
    for i in range(n_lecturas):
        timestamp = (datetime.now() - timedelta(hours=n_lecturas - i - 1)).strftime("%Y-%m-%d %H:%M:%S")
        sensor_id = random.choice(sensores)
        
        # 20% de probabilidad de anomalía
        es_anomalia = 1 if random.random() < 0.2 else 0
        if es_anomalia:
            anomalias_count += 1
        
        # Generar valores según si es anomalía o no
        if es_anomalia:
            temperatura = 180 + random.choice([-50, 50]) + normal_random(0, 10)
            vibracion = 2.5 * random.uniform(2, 5) + normal_random(0, 0.5)
            presion = 1500 + random.choice([-500, 500]) + normal_random(0, 100)
            ruido_db = 85 + random.uniform(20, 40) + normal_random(0, 5)
            tipo = random.choice(tipos_anomalia)
        else:
            temperatura = 180 + normal_random(0, 10)
            vibracion = 2.5 + normal_random(0, 0.5)
            presion = 1500 + normal_random(0, 100)
            ruido_db = 85 + normal_random(0, 5)
            tipo = 'normal'
        
        # Asegurar valores dentro de rangos
        temperatura = max(100, min(300, temperatura))
        vibracion = max(0.1, min(20, vibracion))
        presion = max(500, min(3000, presion))
        ruido_db = max(60, min(150, ruido_db))
        
        writer.writerow([
            timestamp,
            sensor_id,
            round(temperatura, 1),
            round(vibracion, 2),
            round(presion, 1),
            round(ruido_db, 1),
            es_anomalia,
            tipo
        ])

print(f"✓ anomalias_sensores.csv generado ({n_lecturas} registros)")

# RESUMEN
print("\n" + "="*50)
print("RESUMEN DE DATASETS GENERADOS:")
print("="*50)
print(f"1. produccion_historica.csv: {n_dias} días de datos")
print(f"2. eventos_operacionales.csv: {n_eventos} eventos registrados")
print(f"3. parametros_pozos.csv: {n_dias} días de parámetros")
print(f"4. anomalias_sensores.csv: {n_lecturas} lecturas de sensores")
print(f"\nDistribución de anomalías: {(anomalias_count/n_lecturas*100):.1f}%")
print("\n✓ Todos los datasets han sido generados exitosamente")
print("\n💡 Nota: Estos datos son sintéticos para propósitos educativos.")