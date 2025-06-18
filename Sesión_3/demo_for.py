pozos = [10,20,30,40,50,60,70,80,90,100,0.1,0.2,0.3,0.4,0.5,200]
#for elemento in pozos:
#    print(elemento)
#    if elemento >= 30:
#        break
    #print("Fin del ciclo")

#for pozos in pozos:
#    print(pozos)
#    if pozos == 30:
#        #print("Saltando valor 30")
#        continue
#    if pozos == 40:
#        break
#    print(pozos)
        
#for pozos in pozos:
#    if pozos % 2 == 0:
#        print(f"El pozo {pozos} es par")
#    else:
#        print(f"El pozo  {pozos} es impar")

frutas = ["manzana", "banana"] 
colores = ["rojo", "verde"]
for fruta in frutas:
    if fruta == "banana":
        continue
    for color in colores:
        if color == "verde":
            break
        print(f"La fruta {fruta} es de color {color}")

campos = ["Campo A", "Campo B"]
tipos_lectura = ["Temperatura", "Humedad", "pH"]
for campo in campos:
    for lectura in tipos_lectura:
        # Simulamos la toma de una lectura
        valor_lectura = f"Valor de {lectura} para {campo}" 
        print(f"Lectura en {campo} - Tipo: {lectura}, Valor: {valor_lectura}")

valores = [1, 2, 3, 4, 5]
suma = 0
for valor in valores:
    suma += valor
    print(f"La suma de los valores es: {suma}")

valores = [1, 2, 3, 4, 5]
print(valores[len(valores)-1])

valores = ["Pozo A", "Pozo B", "Pozo C", "Pozo D"]
print(valores[-1])
print(valores[len(valores)-1])

lista = [5,4,3,2,1]
for elemento in lista:
    print(f"indice: {lista.index(elemento)} -> elemento: {elemento}")

