import sys, random
sys.setrecursionlimit(10000000)
estados_visitados = set()

# genera una matriz con datos random del 0 al 8 para el 8-puzzle
F = [random.sample(range(9), 9)]

# retorna la cantidad de valores desordenados en el 8-puzzle
def goal_test(V:list[int]):
    desordenados = 0
    if (V[0] != 1):
        desordenados += 1
    if (V[1] != 2):
        desordenados += 1
    if (V[2] != 3):
        desordenados += 1
    if (V[3] != 4):
        desordenados += 1
    if (V[4] != 5):
        desordenados += 1
    if (V[5] != 6):
        desordenados += 1
    if (V[6] != 7):
        desordenados += 1
    if (V[7] != 8):
        desordenados += 1
    if (V[8] != 0):
        desordenados += 1

    return desordenados

def expand(F:list[int]):
    lista_nueva = []
    global estados_visitados

    # arriba a la izquierda (derecha y abajo)
    if (F[0] == 0):
        derecha = F.copy()
        dato = derecha[1]
        derecha[0] = dato
        derecha[1] = 0

        abajo = F.copy()
        dato = abajo[3]
        abajo[0] = dato
        abajo[3] = 0

        lista_nueva.append(derecha)
        lista_nueva.append(abajo)
    
    # arriba en el centro (izquierda, abajo y derecha)
    if (F[1] == 0):
        izquierda = F.copy()
        dato = izquierda[0]
        izquierda[1] = dato
        izquierda[0] = 0

        abajo = F.copy()
        dato = abajo[4]
        abajo[1] = dato
        abajo[4] = 0

        derecha = F.copy()
        dato = derecha[2]
        derecha[1] = dato
        derecha[2] = 0

        lista_nueva.append(izquierda)
        lista_nueva.append(abajo)
        lista_nueva.append(derecha)

    # arriba a la derecha (izquierda y abajo)
    if (F[2] == 0):
        izquierda = F.copy()
        dato = izquierda[1]
        izquierda[2] = dato
        izquierda[1] = 0

        abajo = F.copy()
        dato = abajo[5]
        abajo[2] = dato
        abajo[5] = 0

        lista_nueva.append(izquierda)
        lista_nueva.append(abajo)

    # en medio a la izquierda (arriba, derecha y abajo)
    if (F[3] == 0):
        arriba = F.copy()
        dato = arriba[0]
        arriba[3] = dato
        arriba[0] = 0

        derecha = F.copy()
        dato = derecha[4]
        derecha[3] = dato
        derecha[4] = 0

        abajo = F.copy()
        dato = abajo[6]
        abajo[3] = dato
        abajo[6] = 0

        lista_nueva.append(arriba)
        lista_nueva.append(derecha)
        lista_nueva.append(abajo)

    # en medio (izquierda, arriba, derecha y abajo)
    if (F[4] == 0):
        izquierda = F.copy()
        dato = izquierda[3]
        izquierda[4] = dato
        izquierda[3] = 0

        arriba = F.copy()
        dato = arriba[1]
        arriba[4] = dato
        arriba[1] = 0

        derecha = F.copy()
        dato = derecha[5]
        derecha[4] = dato
        derecha[5] = 0

        abajo = F.copy()
        dato = abajo[7]
        abajo[4] = dato
        abajo[7] = 0

        lista_nueva.append(izquierda)
        lista_nueva.append(arriba)
        lista_nueva.append(derecha)
        lista_nueva.append(abajo)

    # en medio a la derecha (arriba, izquierda y abajo)
    if (F[5] == 0):
        arriba = F.copy()
        dato = arriba[2]
        arriba[5] = dato
        arriba[2] = 0

        izquierda = F.copy()
        dato = izquierda[4]
        izquierda[5] = dato
        izquierda[4] = 0

        abajo = F.copy()
        dato = abajo[8]
        abajo[5] = dato
        abajo[8] = 0

        lista_nueva.append(arriba)
        lista_nueva.append(izquierda)
        lista_nueva.append(abajo)

    # abajo a la izquierda (arriba y derecha)
    if (F[6] == 0):
        arriba = F.copy()
        dato = arriba[3]
        arriba[6] = dato
        arriba[3] = 0

        derecha = F.copy()
        dato = derecha[7]
        derecha[6] = dato
        derecha[7] = 0

        lista_nueva.append(arriba)
        lista_nueva.append(derecha)
    
    # abajo y en medio (izquierda, arriba y derecha)
    if (F[7] == 0):
        izquierda = F.copy()
        dato = izquierda[6]
        izquierda[7] = dato
        izquierda[6] = 0

        arriba = F.copy()
        dato = arriba[4]
        arriba[7] = dato
        arriba[4] = 0

        derecha = F.copy()
        dato = derecha[8]
        derecha[7] = dato
        derecha[8] = 0

        lista_nueva.append(izquierda)
        lista_nueva.append(arriba)
        lista_nueva.append(derecha)

    # abajo a la derecha (izquierda y arriba)
    if (F[8] == 0):
        izquierda = F.copy()
        dato = izquierda[7]
        izquierda[8] = dato
        izquierda[7] = 0

        arriba = F.copy()
        dato = arriba[5]
        arriba[8] = dato
        arriba[5] = 0
        
        lista_nueva.append(izquierda)
        lista_nueva.append(arriba)

    return lista_nueva

def evaluate(F:list[list[int]]):
    evaluados = []
    for i in F:

        desordenados = goal_test(i)
        evaluados.append([desordenados, i])

    return evaluados

def Star(F:list[list[int]]):
    global estados_visitados

    if len(F) == 0:
        print("Solucion no encontrada")
        return None
    
    # los ordena priorizando al que tenga la menor heurística
    F.sort(key=lambda x: goal_test(x))
    estado_actual = F.pop(0)

    estados_visitados.add(tuple(estado_actual))
    print(estado_actual)

    gt = goal_test(estado_actual)
    if gt == 0:
        print("Solucion encontrada: ", estado_actual)
        return None

    hijos = expand(estado_actual)

    # añade todos los nodos hijos al arreglo (los que no se han visitado)
    for h in hijos:
        if tuple(h) not in estados_visitados:
            F.append(h)

    Star(F)

Star(F)