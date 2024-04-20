import random
import osmnx as ox
import folium

PuntoInicioyFin = [24.750986500213106, -107.45732566971502]  # cedis coppel
PuntosInteres = [[24.8259647322571, -107.42613555285189], [24.834542368635553, -107.39806426389583], [24.763531, -107.453989], [24.80085076980745, -107.39122225493001], [24.808862282043563, -107.39730792212923], [24.806291588486456, -107.39628727881541], [24.769247, -107.371943], [24.777955, -107.408095], [24.747421, -107.426716], [24.739089, -107.39311], [24.798349, -107.39361], [24.797711, -107.423402], [24.81342495074406, -107.40072622765123], [24.756552, -107.403092], [24.795033, -107.410673], [24.801446389514645, -107.36533391787977], [24.777126, -107.432649], [24.808594949094665, -107.39575474973914], [24.816005114020253, -107.42619802671574]]
#[[24.777955, -107.408095], [24.8259647322571, -107.42613555285189], [24.808862282043563, -107.39730792212923], [24.756552, -107.403092], [24.808594949094665, -107.39575474973914], [24.797711, -107.423402], [24.769247, -107.371943], [24.80085076980745, -107.39122225493001], [24.834542368635553, -107.39806426389583], [24.806291588486456, -107.39628727881541], [24.816005114020253, -107.42619802671574], [24.81342495074406, -107.40072622765123], [24.739089, -107.39311], [24.801446389514645, -107.36533391787977], [24.777126, -107.432649], [24.763531, -107.453989], [24.795033, -107.410673], [24.747421, -107.426716], [24.798349, -107.39361]]# Crear el gráfico de la red vial de la región de interés 
graph = ox.graph_from_place("Culiacan Sinaloa", network_type='drive')
longitud_total = 0
coordenadas_ruta = []

# calcular desde el punto incio al primer punto de interes:
nodo_inicio = ox.nearest_nodes(graph, PuntoInicioyFin[1], PuntoInicioyFin[0])
nodo_fin = ox.nearest_nodes(graph, PuntosInteres[0][1], PuntosInteres[0][0])
rutacorta = ox.shortest_path(graph, nodo_inicio, nodo_fin, weight='length')

for nodo in rutacorta:
    coordenadas_ruta.append((graph.nodes[nodo]['y'], graph.nodes[nodo]['x']))

longitud_total += sum(ox.utils_graph.get_route_edge_attributes(graph, rutacorta, 'length'))

# se calculan los puntos intermedios
for i in range(len(PuntosInteres) - 1):  # iterar solo hasta el penúltimo punto
    nodo_inicio = ox.nearest_nodes(graph, PuntosInteres[i][1], PuntosInteres[i][0])
    nodo_fin = ox.nearest_nodes(graph, PuntosInteres[i + 1][1], PuntosInteres[i + 1][0])

    print("nodo 1 coordenadas: " + str(PuntosInteres[i][1]) + ", " + str(PuntosInteres[i][0]))
    print("nodo 2 coordenadas: " + str(PuntosInteres[i + 1][1]) + ", " + str(PuntosInteres[i + 1][0]))

    rutacorta = ox.shortest_path(graph, nodo_inicio, nodo_fin, weight='length')

    for nodo in rutacorta:
        coordenadas_ruta.append((graph.nodes[nodo]['y'], graph.nodes[nodo]['x']))

    longitud_total += sum(ox.utils_graph.get_route_edge_attributes(graph, rutacorta, 'length'))
    print(longitud_total)

# calcular desde el punto final al ultimo punto interes:
nodo_fin = ox.nearest_nodes(graph, PuntoInicioyFin[1], PuntoInicioyFin[0])
nodo_inicio = ox.nearest_nodes(graph, PuntosInteres[len(PuntosInteres)-1][1], PuntosInteres[len(PuntosInteres)-1][0])
rutacorta = ox.shortest_path(graph, nodo_inicio, nodo_fin, weight='length')

for nodo in rutacorta:
    coordenadas_ruta.append((graph.nodes[nodo]['y'], graph.nodes[nodo]['x']))

longitud_total += sum(ox.utils_graph.get_route_edge_attributes(graph, rutacorta, 'length'))

print("La distancia total entre los puntos de interés es:", longitud_total, "metros.")

# Crear un mapa centrado en el punto de inicio
mymap = folium.Map(location=[PuntoInicioyFin[0], PuntoInicioyFin[1]], zoom_start=14)

# Agregar marcador para el punto de inicio y fin
folium.Marker(location=[PuntoInicioyFin[0], PuntoInicioyFin[1]], popup='Punto de inicio y fin').add_to(mymap)

# Convertir los nodos de los puntos de interés en coordenadas
# Agregar marcadores para los puntos de interés

for i, punto in enumerate(PuntosInteres):
    folium.Marker(location=punto, icon=folium.DivIcon(html=f'<div style="font-size: 14pt; color: white; background-color: blue; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center;">{i+1}</div>')).add_to(mymap)

colores_ruta = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']

# Crear la ruta en el mapa
for i in range(len(coordenadas_ruta) - 1):
    # Obtener las coordenadas de la sección de la ruta
    inicio = coordenadas_ruta[i]
    fin = coordenadas_ruta[i + 1]

    # Seleccionar un color aleatorio
    color = random.choice(colores_ruta)

    # Agregar la sección de la ruta al mapa
    folium.PolyLine(locations=[inicio, fin], color=color, weight=5, opacity=0.7).add_to(mymap)

# Guardar el mapa como archivo HTML
mymap.save('ruta1.html')

# Mostrar el mapa en el navegador
mymap