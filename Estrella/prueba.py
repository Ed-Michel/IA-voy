import sys
import random
import pygame
from pygame.locals import *

sys.setrecursionlimit(10000000)
estados_visitados = set()

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

# Generate a random initial state for the 8-puzzle
F = [random.sample(range(9), 9)]

# Returns the number of misplaced tiles in the 8-puzzle
def goal_test(V):
    misplaced = sum(1 for i, j in zip(V, range(1, 9)) if i != j)
    return misplaced

# Expands the current state of the 8-puzzle
def expand(F):
    lista_nueva = []
    global estados_visitados

    empty_index = F.index(0)
    row, col = empty_index // 3, empty_index % 3

    # Move the empty tile to the left
    if col > 0:
        left = F.copy()
        left[empty_index], left[empty_index - 1] = left[empty_index - 1], left[empty_index]
        lista_nueva.append(left)

    # Move the empty tile to the right
    if col < 2:
        right = F.copy()
        right[empty_index], right[empty_index + 1] = right[empty_index + 1], right[empty_index]
        lista_nueva.append(right)

    # Move the empty tile up
    if row > 0:
        up = F.copy()
        up[empty_index], up[empty_index - 3] = up[empty_index - 3], up[empty_index]
        lista_nueva.append(up)

    # Move the empty tile down
    if row < 2:
        down = F.copy()
        down[empty_index], down[empty_index + 3] = down[empty_index + 3], down[empty_index]
        lista_nueva.append(down)

    return lista_nueva

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

# Performs the A* search to solve the 8-puzzle
def Star(F):
    global estados_visitados

    if len(F) == 0:
        print("Solucion no encontrada")
        return None
    
    # Sorts the list prioritizing the one with the smallest heuristic
    F.sort(key=lambda x: goal_test(x))
    estado_actual = F.pop(0)

    estados_visitados.add(tuple(estado_actual))
    display_puzzle(estado_actual)

    gt = goal_test(estado_actual)
    if gt == 0:
        print("Solucion encontrada: ", estado_actual)
        return None

    hijos = expand(estado_actual)

    # Adds all child nodes to the list (those that have not been visited)
    for h in hijos:
        if tuple(h) not in estados_visitados:
            F.append(h)

    pygame.time.delay(500)  # Delay for visualization
    Star(F)

# Start the A* search
Star(F)

# Event loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

pygame.quit()
