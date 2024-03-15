import sys
import random

sys.setrecursionlimit(10000000)
estados_visitados = set()

# Genera una matriz con datos random del 0 al 8 para el 8-puzzle
inicio = random.sample(range(9), 9)
objetivo = [1, 2, 3, 4, 5, 6, 7, 8, 0]

# Retorna True si ambas listas coinciden en algún punto, False de lo contrario
def goal_test(lista1, lista2):
    return any(estado in lista2 for estado in lista1)

# Expande el estado actual del 8-puzzle
def expand(F):
    lista_nueva = []
    global estados_visitados

    # Definir movimientos posibles del espacio vacío (0)
    movimientos = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4, 6],
        4: [1, 3, 5, 7],
        5: [2, 4, 8],
        6: [3, 7],
        7: [4, 6, 8],
        8: [5, 7]
    }

    # Encuentra la posición del espacio vacío
    espacio_vacio = F.index(0)

    # Genera los nuevos estados después de mover el espacio vacío
    for movimiento in movimientos[espacio_vacio]:
        nuevo_estado = F[:]
        nuevo_estado[espacio_vacio], nuevo_estado[movimiento] = nuevo_estado[movimiento], nuevo_estado[espacio_vacio]
        lista_nueva.append(nuevo_estado)

    return lista_nueva

# Algoritmo bidireccional para resolver el 8-puzzle
def bidirectional_search():
    global estados_visitados

    # Inicializa ambas fronteras de búsqueda
    frontera_inicio = [(inicio, 0)]  # Se agrega el costo del camino desde el inicio
    frontera_objetivo = [(objetivo, 0)]  # Se agrega el costo del camino desde el objetivo

    while frontera_inicio and frontera_objetivo:
        # Realiza la búsqueda desde el estado inicial
        estado_actual_inicio, costo_inicio = frontera_inicio.pop(0)
        estados_visitados.add(tuple(estado_actual_inicio))
        print("Desde el inicio:", estado_actual_inicio)

        # Chequea si el estado actual coincide con algún estado en la frontera objetivo
        for estado_objetivo, costo_objetivo in frontera_objetivo:
            if estado_actual_inicio == estado_objetivo:
                print("¡Solución encontrada!")
                print("Ruta óptima:", estado_actual_inicio)
                return

        for hijo in expand(estado_actual_inicio):
            if tuple(hijo) not in estados_visitados:
                frontera_inicio.append((hijo, costo_inicio + 1))

        # Realiza la búsqueda desde el estado objetivo
        estado_actual_objetivo, costo_objetivo = frontera_objetivo.pop(0)
        estados_visitados.add(tuple(estado_actual_objetivo))
        print("Desde el objetivo:", estado_actual_objetivo)

        # Chequea si el estado actual coincide con algún estado en la frontera inicio
        for estado_inicio, costo_inicio in frontera_inicio:
            if estado_actual_objetivo == estado_inicio:
                print("¡Solución encontrada!")
                print("Ruta óptima:", estado_actual_objetivo)
                return

        for hijo in expand(estado_actual_objetivo):
            if tuple(hijo) not in estados_visitados:
                frontera_objetivo.append((hijo, costo_objetivo + 1))

    print("No se encontró una solución.")

bidirectional_search()