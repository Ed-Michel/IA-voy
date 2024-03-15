import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.setrecursionlimit(10000000)
F = [[0, 0, 0, 0]]

def goal_test(V:list[int]):
    ataques = 0
    n = len(V)
    for i in range(n - 1):
        for j in range (i + 1, n):
            if V[i] == V[j]:
                ataques += 2
            elif abs(i - j) == abs(V[i] - V[j]):
                ataques += 2
    return ataques == 0

def expand(F:list[int]):
    lista_nueva = []
    
    for i in range(len(F)):
        nuevo_estado = F.copy()

        if F[i] < len(F)-1:
            nuevo_estado[i] += 1
            lista_nueva.append(nuevo_estado)

    return lista_nueva

def dibujar_tablero(estado, ax, nivel, limite):
    ax.clear()
    n = len(estado)
    ax.scatter(range(0, n), estado, color='red', marker='o')
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(range(0, n))
    ax.set_yticks(range(0, n))
    ax.grid(True)
    ax.set_title('Posición de las reinas en el tablero')
    ax.set_xlabel('Columnas')
    # ax.set_label('Nivel: '+str(nivel)+'    Limite: '+str(limite))
    ax.text(0.5, -0.14, f'Nivel: {nivel}     Límite: {limite}', horizontalalignment='center', transform=ax.transAxes)
    ax.set_ylabel('Filas')

def LDFS(F:list[list[int]], nivel:int, limite:int):
    if len(F) == 0:
        print("Solucion no encontrada")
        return None
    
    estado_actual = F.pop(0)
    print(estado_actual)
    
    dibujar_tablero(estado_actual, ax, nivel, limite)  # dibujar el estado actual del tablero
    plt.pause(0.001)  # pausa para visualizar el estado antes de avanzar
    
    if goal_test(estado_actual):
        print("Solucion encontrada: ", estado_actual)
        return None
        
    # F = append (os, F)
    if (nivel < limite-1):
        F[0:0] = expand(estado_actual)
        nivel += 1
    else:
        print("Se llego al limite, no hay solucion")
        return None

    LDFS(F, nivel, limite)

fig, ax = plt.subplots()
LDFS(F, nivel=0, limite=10)
plt.show()