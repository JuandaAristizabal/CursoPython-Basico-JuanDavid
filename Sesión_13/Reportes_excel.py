import pandas as pd

# Cargar datos limpios
df = pd.read_csv('/workspaces/CursoPython-Basico-JuanDavid/Sesión_13/datos/datos_produccion_automatizada_limpio.csv', parse_dates=['fecha'])

# Crear resumen
resumen = {
    'Total producción (bpd)': [df['produccion_bpd'].sum()],
    'Promedio presión (psi)': [df['presion_psi'].mean()],
    'Promedio temperatura (F)': [df['temperatura_f'].mean()]
}
df_resumen = pd.DataFrame(resumen)

# Escribir a Excel con varias hojas
with pd.ExcelWriter('/workspaces/CursoPython-Basico-JuanDavid/Sesión_13/datos/reporte_produccion.xlsx', engine='openpyxl') as writer:
    df_resumen.to_excel(writer, sheet_name='Resumen', index=False)
    df.to_excel(writer, sheet_name='Datos', index=False)
print('Reporte Excel generado.') 