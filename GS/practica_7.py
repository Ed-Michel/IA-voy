import sys, random
sys.setrecursionlimit(100000)
estados_generados = []

# genera una matriz con ataques random del 0 al 3
F = [[random.randint(0, 49) for _ in range(50)]]

def goal_test(V:list[int]):
    ataques = 0
    n = len(V)
    for i in range(n - 1):
        for j in range (i + 1, n):
            if V[i] == V[j]:
                ataques += 2
            elif abs(i - j) == abs(V[i] - V[j]):
                ataques += 2
    return ataques

def expand(F:list[int]):
    lista_nueva = []
    global estados_generados

    for i in range(len(F)):
        for j in range(len(F)):

            # para no repetir el estado actual
            if j != F[i]:

                auxiliar = F.copy()
                auxiliar[i] = j

                # para no volver a meter los estados anteriormente generados
                if (auxiliar not in estados_generados):
                    estados_generados.append(auxiliar)
                    lista_nueva.append(auxiliar)

    return lista_nueva

def evaluate(F:list[list[int]]):
    evaluados = []
    for i in F:

        ataques = goal_test(i)
        evaluados.append([ataques, i])

    return evaluados

def GS(F:list[list[int]]):
    if len(F) == 0:
        print("Solucion no encontrada")
        return None

    estado_actual = F.pop(0)
    print(estado_actual)
    
    gt = goal_test(estado_actual)
    if gt == 0:
        print("Solucion encontrada: ", estado_actual)
        return None


    os = expand(estado_actual)
    os = evaluate(os)
    os = sorted(os, key=lambda x:x[0])

    # agarra al primero de la lista
    os = [os[0][1]]

    GS(os)

GS(F)