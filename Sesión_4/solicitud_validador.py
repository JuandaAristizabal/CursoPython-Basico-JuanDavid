from validador import calcular_volumen_total_produccion

#presiones = [99, 100, 150, 200, 250, 300, 350]

#for presion in presiones:
#    try:
#        resultado = validar_presion(presion)
#        print(f"Presión: {presion} - Resultado: {resultado}")
#    except ValueError as e:
#        print(f"Presión: {presion} - Error: {e}")

#while True:
#    presion = float(input("Ingrese la presión: "))
#    print(f"Presión: {presion} - Resultado: {resultado}")
#    break

produccion = [10, 200, 300, 400, 500]
total = calcular_volumen_total_produccion(produccion)
print(f"Volumen total de producción: {total}")
