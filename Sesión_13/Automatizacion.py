import pandas as pd

# Cargar datos
ruta = r'/workspaces/CursoPython-Basico-JuanDavid/Sesión_13/datos/datos_produccion_automatizada.csv'
df = pd.read_csv(ruta, parse_dates=['fecha'])

# Limpieza básica: eliminar duplicados y valores nulos
df = df.drop_duplicates()
df = df.dropna()

# Análisis básico
total_produccion = df['produccion_bpd'].sum()
promedio_presion = df['presion_psi'].mean()
print(f'Total producción (bpd): {total_produccion}')
print(f'Presión promedio (psi): {promedio_presion:.2f}')

# Guardar datos limpios
df.to_csv('/workspaces/CursoPython-Basico-JuanDavid/Sesión_13/datos/datos_produccion_automatizada_limpio.csv', index=False) 