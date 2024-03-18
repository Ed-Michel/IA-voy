import csv
import heapq

# función para cargar datos desde un archivo CSV y devolver una lista de tuplas
def cargar_csv(archivo):
    datos = []
    with open(archivo, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:  # saltar la primera fila que contiene encabezados
                continue
            if len(row) == 3:  # si hay tres columnas, se asume que es el primer tipo de archivo (distancias)
                datos.append((row[0], row[1], int(row[2])))
            elif len(row) == 2:  # si hay dos columnas, se asume que es el segundo tipo de archivo (linea recta)
                try:
                    datos.append((row[0], int(row[1])))
                except ValueError:
                    print(f"Error: valor no válido en la fila {i+1}: {row}")
    return datos

# cargar datos del primer archivo CSV
archivo1 = "C:/Users/edgar/Desktop/Inteligencia Artificial/IA/Bucharest/linea recta a bucharest.csv"
datos_ciudades = cargar_csv(archivo1)

# cargar datos del segundo archivo CSV
archivo2 = "C:/Users/edgar/Desktop/Inteligencia Artificial/IA/Bucharest/distancias entre ciudades.csv"
datos_distancias = cargar_csv(archivo2)

def obtener_distancia_recta_a_bucarest(ciudad):
    for dato in datos_ciudades:
        if dato[0] == ciudad:
            return dato[1]  # acceder al segundo elemento de la tupla
    return None

def obtener_distancia_entre_ciudades(ciudad1, ciudad2):
    for dato in datos_distancias:
        if (dato[0] == ciudad1 and dato[1] == ciudad2) or (dato[0] == ciudad2 and dato[1] == ciudad1):
            return dato[2]  # la tercera columna contiene la distancia entre ciudades
    return None

def obtener_ciudades_conectadas(ciudad):
    ciudades_conectadas = []
    for dato in datos_distancias:
        if dato[0] == ciudad:
            ciudades_conectadas.append(dato[1])
        elif dato[1] == ciudad:
            ciudades_conectadas.append(dato[0])
    return ciudades_conectadas

def star(ciudad_inicio, ciudad_meta):
    # estructuras de datos para almacenar los nodos a explorar y los nodos ya explorados por completo
    nodos_a_explorar = []
    nodos_explorados = {}
    
    # inicializar el nodo de inicio
    g = {ciudad_inicio: 0}
    h = {ciudad_inicio: obtener_distancia_recta_a_bucarest(ciudad_inicio)}
    f = {ciudad_inicio: h[ciudad_inicio]}
    heapq.heappush(nodos_a_explorar, (f[ciudad_inicio], ciudad_inicio))
    
    while nodos_a_explorar:
        _, ciudad_actual = heapq.heappop(nodos_a_explorar)
        print("Explorando ciudad actual:", ciudad_actual)
        
        # si ya se llega a la ciudad destino
        if ciudad_actual == ciudad_meta:
            # reconstruir el camino óptimo
            camino_optimo = [ciudad_actual]
            while ciudad_actual in nodos_explorados:
                ciudad_actual = nodos_explorados[ciudad_actual]
                camino_optimo.append(ciudad_actual)
            return camino_optimo[::-1]  # devolver el camino en orden correcto (ya que estaba al revés)
        
        for ciudad_vecina in obtener_ciudades_conectadas(ciudad_actual):
            nueva_g = g[ciudad_actual] + obtener_distancia_entre_ciudades(ciudad_actual, ciudad_vecina)
            print("Explorando vecino:", ciudad_vecina)
            
            if ciudad_vecina in g and nueva_g >= g[ciudad_vecina]:
                print("No es un camino mejor")
                continue  # no es mejor
                
            # si se ha encontrado un mejor camino
            nodos_explorados[ciudad_vecina] = ciudad_actual
            g[ciudad_vecina] = nueva_g
            h[ciudad_vecina] = obtener_distancia_recta_a_bucarest(ciudad_vecina)
            f[ciudad_vecina] = g[ciudad_vecina] + h[ciudad_vecina]
            heapq.heappush(nodos_a_explorar, (f[ciudad_vecina], ciudad_vecina))
            print("Vecino añadido a nodos abiertos:", ciudad_vecina)
    
    # si no se puede encontrar un camino, devolver None
    return None

# aquí se escribe la ciudad de donde se inicia y hacia a dónde se quiere ir
ciudad_inicio = "Arad"
ciudad_meta = "Bucharest"
ruta_optima = star(ciudad_inicio, ciudad_meta)
print("\n"+"La ruta óptima desde", ciudad_inicio, "a", ciudad_meta, "es", ruta_optima)