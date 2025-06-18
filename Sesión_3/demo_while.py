# Contar del 1 al 5
# Variable Contador
# Validar que el contador sea menor o igual a 5
#contador = 1
#while contador  <= 5:
#    print(contador)
#    contador += 1

#def monitereo_presion(y_pressure, valor_minimo, delta):
#    presion = y_pressure
#    while presion > valor_minimo:
#        print (MENSAJE_PRESION, presion)
        #print(f"La Presión es: {presion}")
#        presion -= delta

#if __name__ == "__main__":
#    MENSAJE_PRESION = "La presión es: "
#    pressure_initial = 2000
#    valor_minimo = 100
#    delta = 200
#    monitereo_presion(pressure_initial, valor_minimo, delta)


import argparse

def monitorear_caida_presion(v_presion, valor_minimo, delta, mensaje_presion):
    presion = v_presion

    while presion > valor_minimo:
        print(mensaje_presion, presion)
        presion -= delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitorea la caída de presión.")
    parser.add_argument("presion_inicial", type=int, help="Presión inicial")
    parser.add_argument("valor_minimo", type=int, help="Valor mínimo de presión")
    parser.add_argument("delta", type=int, help="Decremento de presión en cada iteración")
    parser.add_argument("--mensaje_presion", type=str, default="The current presion is: ", help="Mensaje a mostrar")

    args = parser.parse_args()

    monitorear_caida_presion(args.presion_inicial, args.valor_minimo, args.delta, args.mensaje_presion)
