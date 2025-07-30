import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Cargar datos limpios
df = pd.read_csv(r'/workspaces/CursoPython-Basico-JuanDavid/Sesión_13/datos/datos_produccion_automatizada_limpio.csv', parse_dates=['fecha'])

# Crear gráfico de producción
plt.figure(figsize=(8,4))
plt.plot(df['fecha'], df['produccion_bpd'], marker='o')
plt.title('Producción diaria')
plt.xlabel('Fecha')
plt.ylabel('BPD')
plt.tight_layout()
grafico_path = '/workspaces/CursoPython-Basico-JuanDavid/Sesión_13/datos/grafico_produccion.png'
plt.savefig(grafico_path)
plt.close()

# Crear PDF e insertar gráfico
pdf_path = '/workspaces/CursoPython-Basico-JuanDavid/Sesión_13/datos/reporte_produccion.pdf'
c = canvas.Canvas(pdf_path, pagesize=letter)
c.drawString(100, 750, 'Reporte de Producción')
c.drawImage(grafico_path, 100, 500, width=400, height=200)
c.save()
print('Reporte PDF generado.') 