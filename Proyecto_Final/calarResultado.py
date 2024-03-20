import math

def fitness_distancia(individuo):
    # Diccionario de coordenadas
    coordenadas = {
        0: [1, 7],
        1: [3, 7],
        2: [5, 7],
        3: [7, 7],
        4: [1, 5],
        5: [3, 5],
        6: [5, 5],
        7: [7, 5],
        8: [1, 3],
        9: [3, 3],
        10: [5, 3],
        11: [7, 3],
        12: [1, 1],
        13: [3, 1],
        14: [5, 1],
        15: [7, 1]
    }

    # Calcular la distancia total recorrida
    distancia_total = 0
    cont = 1
    while(cont < 9):
        if(cont == 1):
             distancia_total +=  2 * (distancia_euclidiana(coordenadas[individuo[0]], coordenadas[0]))
        if(cont == 2):
             distancia_total += distancia_euclidiana(coordenadas[individuo[0]], coordenadas[individuo[3]])
        if(cont == 3):
             distancia_total += distancia_euclidiana(coordenadas[individuo[3]], coordenadas[individuo[4]])
        if(cont == 4):
             distancia_total += distancia_euclidiana(coordenadas[individuo[4]], coordenadas[individuo[2]])
        if(cont == 5):
             distancia_total += 2* (distancia_euclidiana(coordenadas[individuo[3]], coordenadas[individuo[2]]))
        if(cont == 6):
             distancia_total += distancia_euclidiana(coordenadas[individuo[2]], coordenadas[individuo[0]])
        if(cont == 7):
             distancia_total += 2* ( distancia_euclidiana(coordenadas[4], coordenadas[individuo[0]]))
        if(cont == 8):
             distancia_total += 2* ( distancia_euclidiana(coordenadas[2], coordenadas[individuo[0]]))
        if(cont == 9):
             distancia_total += distancia_euclidiana(coordenadas[individuo[0]], coordenadas[7])
        cont += 1
    return distancia_total

def distancia_euclidiana(punto1, punto2):
        """
        Calcula la distancia euclidiana entre dos puntos en un espacio n-dimensional.

        Args:
        punto1 (list): Coordenadas del primer punto.
        punto2 (list): Coordenadas del segundo punto.

        Returns:
        float: La distancia euclidiana entre los dos puntos.
        """
        if len(punto1) != len(punto2):
            raise ValueError("Los puntos deben tener la misma cantidad de dimensiones.")

        suma_cuadrados = sum((q - p) ** 2 for p, q in zip(punto1, punto2))
        distancia = math.sqrt(suma_cuadrados)
        return distancia

# Ejemplo de uso:
individuo_ejemplo = [15, 13, 12, 3, 14, 4, 7]
aptitud = fitness_distancia(individuo_ejemplo)
print("La aptitud del individuo es:", aptitud)