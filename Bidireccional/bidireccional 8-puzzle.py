import pygame
from pygame.locals import *

class Nodo:
    def __init__(self, estado, padre=None):
        self.estado = estado
        self.padre = padre

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Set up the display
WIDTH, HEIGHT = 300, 300
TILE_SIZE = WIDTH // 3

pygame.init()
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8 Puzzle")
font = pygame.font.Font(None, 36)

def expandir_nodo(nodo):
    hijos = []
    estado = nodo.estado
    # Encontrar la posición de la celda vacía
    vacio_i, vacio_j = None, None
    for i in range(3):
        for j in range(3):
            if estado[i][j] is None:
                vacio_i, vacio_j = i, j
                break
        if vacio_i is not None:
            break

    # Movimientos posibles: arriba, abajo, izquierda, derecha
    movimientos = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for di, dj in movimientos:
        nueva_i, nueva_j = vacio_i + di, vacio_j + dj
        if 0 <= nueva_i < 3 and 0 <= nueva_j < 3:
            # Copiar el estado actual para evitar modificarlo
            nuevo_estado = [fila[:] for fila in estado]
            # Intercambiar la celda vacía con la celda adyacente
            nuevo_estado[vacio_i][vacio_j], nuevo_estado[nueva_i][nueva_j] = nuevo_estado[nueva_i][nueva_j], nuevo_estado[vacio_i][vacio_j]
            hijos.append(Nodo(nuevo_estado, padre=nodo))

    return hijos

goalPath = []
initialPath = []

def add_EApath(elemento):
    initialPath.append(elemento)

def add_goalPath(elemento):
    goalPath.append(elemento)

def goal_test_bidireccional(nodo, nodos_visitados_inicio, nodos_visitados_meta, type):
    estado_str = ''.join(''.join(map(str, fila)) for fila in nodo.estado)
    if type == 0:
        if estado_str in nodos_visitados_inicio:
            print("Solución encontrada:")
            imprimir_camino(initialPath.pop(),goalPath.pop())
            return True
    elif type == 1:
        if estado_str in nodos_visitados_meta:
            print("Solución encontrada:")
            imprimir_camino(initialPath.pop(),goalPath.pop())
            return True
    return False

def bs_EA(nodos_visitados_meta, nodos_visitados_inicio, nodos_inicio):
    #print("bs_EA")
    if not nodos_inicio:
        return False
    
    EA = nodos_inicio.pop(0)
    #imprimir_matriz(EA.estado)
    estado_str = ''.join(''.join(map(str, fila)) for fila in EA.estado)  # Convertir el estado a una cadena de texto    

    if goal_test_bidireccional(EA, nodos_visitados_inicio, nodos_visitados_meta, 1):
        print("Solución encontrada")
        return True
    nodos_visitados_inicio.add(estado_str)  # Agregar la cadena de texto al conjunto
    add_EApath(EA) #se agregan a una lista global

    OS = expandir_nodo(EA)
    nodos_inicio.extend(OS)
    return False

def bs_META(nodos_visitados_meta, nodos_visitados_inicio, nodos_meta):
    if not nodos_meta:
        return False
    
    EA = nodos_meta.pop(0)
    #imprimir_matriz(EA.estado)
    estado_str = ''.join(''.join(map(str, fila)) for fila in EA.estado)  # Convertir el estado a una cadena de texto

    if goal_test_bidireccional(EA, nodos_visitados_inicio, nodos_visitados_meta, 0):
        print("Solución encontrada")
        return True
    nodos_visitados_meta.add(estado_str)  # Agregar la cadena de texto al conjunto
    add_goalPath(EA) #se agregan a una lista global

    OS = expandir_nodo(EA)
    nodos_meta.extend(OS)
    return False

# Display the current state of the 8-puzzle
def display_puzzle(puzzle):
    window.fill(WHITE)
    for i in range(3):
        for j in range(3):
            pygame.draw.rect(window, GRAY, (j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE), 2)
            if puzzle[i * 3 + j] != 0:
                text = font.render(str(puzzle[i * 3 + j]), True, BLACK)
                text_rect = text.get_rect(center=(j * TILE_SIZE + TILE_SIZE // 2, i * TILE_SIZE + TILE_SIZE // 2))
                window.blit(text, text_rect)
    pygame.display.update()

def imprimir_camino(nodo_desde_inicio, nodo_desde_meta):
    # Encuentra el camino desde el estado inicial hasta el estado meta
    camino_inicio = []
    camino_meta = []
    nodo_actual = nodo_desde_inicio
    nodo_meta = nodo_desde_meta
    while nodo_actual is not None:
        camino_inicio.insert(0, nodo_actual)
        nodo_actual = nodo_actual.padre

    while nodo_meta is not None:
        camino_meta.append(nodo_meta)
        nodo_meta = nodo_meta.padre

    camino_inicio.extend(camino_meta)

   # Imprime el camino desde el estado meta hasta el estado inicial
    print("Camino desde el estado meta al estado inicial:")
    for i, nodo in enumerate(camino_inicio):  # Sin invertir el camino
        print(f"Paso {i+1}:")
        imprimir_matriz(nodo.estado)
        lista = [0 if elemento is None else elemento for sublista in nodo.estado for elemento in sublista]
        display_puzzle(lista)
        pygame.time.delay(500)  # Delay for visualization
        print()  # Línea en blanco entre pasos

def bs():
    print("empezando a chambear")
    # Estado inicial
    estado_inicial = [
        [1, 2, 3],
        [4, 5, 8],
        [6, 7, None]
    ]

    estado_meta = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, None]
    ]

    nodo_raiz = Nodo(estado_inicial)
    nodo_meta = Nodo(estado_meta)

    nodos_raiz = [nodo_raiz]
    nodos_meta = [nodo_meta]

    nodos_visitados_meta = set([''.join(''.join(map(str, fila)) for fila in estado_meta)])  # Usar una cadena de texto
    nodos_visitados_inicio = set([''.join(''.join(map(str, fila)) for fila in estado_inicial)])  # Usar una cadena de texto

    add_EApath(nodo_raiz)
    add_goalPath(nodos_meta)

    while True:
        if bs_EA(nodos_visitados_meta, nodos_visitados_inicio, nodos_raiz):
            break
        if bs_META(nodos_visitados_meta, nodos_visitados_inicio, nodos_meta):
            break

def imprimir_matriz(matriz):
    for fila in matriz:
        print(fila)

def funcion():
    if __name__ == "__main__":
        bs()

funcion()

# Event loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

pygame.quit()