import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from collections import deque

def extract_skeleton(image_path):
    """Estrae lo skeleton dall'immagine"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skeleton = morphology.skeletonize(binary > 0)
    return skeleton.astype(np.uint8) * 255

def get_neighbors_8(y, x, shape):
    """Restituisce i vicini in 8-connettività"""
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                neighbors.append((ny, nx))
    return neighbors

def count_neighbors(skeleton, y, x):
    """Conta i vicini attivi di un pixel"""
    count = 0
    for ny, nx in get_neighbors_8(y, x, skeleton.shape):
        if skeleton[ny, nx] > 0:
            count += 1
    return count

def find_critical_points_simple(skeleton):
    """Trova endpoint e branch points in modo semplice e robusto"""
    endpoints = []
    branch_points = []

    # Analizza ogni pixel dello skeleton
    coords = np.argwhere(skeleton > 0)

    for y, x in coords:
        n_neighbors = count_neighbors(skeleton, y, x)

        if n_neighbors == 1:
            endpoints.append((y, x))
        elif n_neighbors >= 3:
            branch_points.append((y, x))

    return np.array(branch_points), np.array(endpoints)

def merge_close_points(points, min_distance=10):
    """Unisce punti troppo vicini tra loro"""
    if len(points) == 0:
        return points

    if len(points) == 1:
        return points

    # Calcola matrice delle distanze
    distances = squareform(pdist(points))

    # Clustering greedy
    merged = []
    used = set()

    for i in range(len(points)):
        if i in used:
            continue

        # Trova tutti i punti vicini
        close_indices = [i]
        for j in range(len(points)):
            if j != i and j not in used and distances[i, j] < min_distance:
                close_indices.append(j)
                used.add(j)

        # Calcola il centroide
        cluster_points = points[close_indices]
        centroid = np.mean(cluster_points, axis=0).astype(int)
        merged.append(tuple(centroid))
        used.add(i)

    return np.array(merged)

def simplify_skeleton_pruning(skeleton, min_branch_length=15):
    """Semplifica lo skeleton rimuovendo rami corti"""
    result = skeleton.copy()

    # Itera fino a quando non ci sono più rami da rimuovere
    for iteration in range(20):
        # Trova gli endpoint
        endpoints = []
        coords = np.argwhere(result > 0)

        for y, x in coords:
            if count_neighbors(result, y, x) == 1:
                endpoints.append((y, x))

        if len(endpoints) == 0:
            break

        removed_any = False

        for y, x in endpoints:
            if result[y, x] == 0:  # Già rimosso
                continue

            # Traccia il ramo
            path = [(y, x)]
            visited = {(y, x)}
            current = (y, x)

            for step in range(min_branch_length):
                # Trova il prossimo pixel
                found_next = False
                for ny, nx in get_neighbors_8(current[0], current[1], result.shape):
                    if result[ny, nx] > 0 and (ny, nx) not in visited:
                        n_neighbors = count_neighbors(result, ny, nx)

                        # Se è un branch point, fermati
                        if n_neighbors >= 3:
                            break

                        path.append((ny, nx))
                        visited.add((ny, nx))
                        current = (ny, nx)
                        found_next = True
                        break

                if not found_next:
                    break

            # Se il ramo è più corto del minimo, rimuovilo
            if len(path) < min_branch_length:
                for py, px in path:
                    result[py, px] = 0
                removed_any = True

        if not removed_any:
            break

    return result

def build_graph_from_skeleton(skeleton, branch_points, endpoints, max_trace_length=500):
    """Costruisce il grafo tracciando i percorsi tra punti critici"""
    G = nx.Graph()

    # Combina tutti i punti critici
    all_critical = []

    for pt in branch_points:
        all_critical.append(('branch', tuple(pt)))

    for pt in endpoints:
        all_critical.append(('endpoint', tuple(pt)))

    if len(all_critical) == 0:
        print("  ATTENZIONE: Nessun punto critico trovato!")
        return G, {}

    # Trova la radice (punto più basso)
    skel_coords = np.argwhere(skeleton > 0)
    if len(skel_coords) == 0:
        return G, {}

    root_idx = np.argmax(skel_coords[:, 0])
    root = tuple(skel_coords[root_idx])

    # Trova il punto critico più vicino alla radice
    min_dist = float('inf')
    root_critical = None
    for node_type, pt in all_critical:
        dist = np.sqrt((pt[0] - root[0])**2 + (pt[1] - root[1])**2)
        if dist < min_dist:
            min_dist = dist
            root_critical = pt

    # Crea mapping punti -> ID
    point_to_id = {}
    for i, (node_type, pt) in enumerate(all_critical):
        point_to_id[pt] = i

        # Determina il tipo corretto
        if pt == root_critical:
            final_type = 'root'
        else:
            final_type = node_type

        G.add_node(i, pos=pt, type=final_type, coordinates=(pt[1], pt[0]))

    print(f"  Nodi creati: {G.number_of_nodes()}")

    # Funzione per tracciare percorsi tra punti critici
    def trace_path_between_points(start, end, skeleton, critical_set):
        """BFS per trovare il percorso più corto tra due punti"""
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            # Se abbiamo raggiunto la destinazione
            if current == end:
                return path

            # Se il percorso è troppo lungo, abbandona
            if len(path) > max_trace_length:
                continue

            # Esplora i vicini
            for neighbor in get_neighbors_8(current[0], current[1], skeleton.shape):
                if neighbor in visited:
                    continue

                if skeleton[neighbor[0], neighbor[1]] == 0:
                    continue

                visited.add(neighbor)
                new_path = path + [neighbor]

                # Se abbiamo trovato un altro punto critico
                if neighbor in critical_set and neighbor != start:
                    if neighbor == end:
                        return new_path
                    # Non attraversare altri punti critici
                    continue

                queue.append((neighbor, new_path))

        return None

    # Set di tutti i punti critici
    critical_set = set(pt for _, pt in all_critical)

    # Traccia connessioni tra tutti i punti critici
    print("  Tracciamento connessioni...")
    edges_found = 0

    for i, (type1, pt1) in enumerate(all_critical):
        # Per ogni punto critico, esplora i vicini immediati
        for neighbor in get_neighbors_8(pt1[0], pt1[1], skeleton.shape):
            if skeleton[neighbor[0], neighbor[1]] == 0:
                continue

            # Se il vicino è un altro punto critico, collega direttamente
            if neighbor in critical_set and neighbor != pt1:
                id1 = point_to_id[pt1]
                id2 = point_to_id[neighbor]

                if not G.has_edge(id1, id2):
                    G.add_edge(id1, id2, length=1, path=[pt1, neighbor])
                    edges_found += 1
                continue

            # Altrimenti, traccia il percorso verso altri punti critici
            for j, (type2, pt2) in enumerate(all_critical):
                if i >= j:  # Evita duplicati
                    continue

                id1 = point_to_id[pt1]
                id2 = point_to_id[pt2]

                if G.has_edge(id1, id2):
                    continue

                # Traccia il percorso
                path = trace_path_between_points(pt1, pt2, skeleton, critical_set)

                if path is not None:
                    G.add_edge(id1, id2, length=len(path), path=path)
                    edges_found += 1

    print(f"  Connessioni trovate: {edges_found}")

    return G, point_to_id

def analyze_tree_structure(G):
    """Analizza la struttura gerarchica dell'albero"""
    print("\n=== ANALISI STRUTTURA ALBERO ===")
    print(f"Numero totale di nodi: {G.number_of_nodes()}")
    print(f"Numero totale di rami: {G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        print("ATTENZIONE: Grafo vuoto!")
        return G

    # Conta i tipi di nodi
    node_types = nx.get_node_attributes(G, 'type')
    type_counts = {}
    for node_type in node_types.values():
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    print(f"\nTipologia nodi:")
    print(f"  - Radice (tronco): {type_counts.get('root', 0)}")
    print(f"  - Biforcazioni: {type_counts.get('branch', 0)}")
    print(f"  - Estremità (foglie): {type_counts.get('endpoint', 0)}")

    # Analisi dei percorsi
    if G.number_of_edges() > 0:
        edge_lengths = [data['length'] for _, _, data in G.edges(data=True)]
        print(f"\nLunghezza rami:")
        print(f"  - Media: {np.mean(edge_lengths):.1f} pixel")
        print(f"  - Min: {min(edge_lengths)} pixel")
        print(f"  - Max: {max(edge_lengths)} pixel")
        print(f"  - Totale: {sum(edge_lengths)} pixel")

        # Gradi dei nodi
        degrees = dict(G.degree())
        print(f"\nConnettività:")
        print(f"  - Grado medio: {np.mean(list(degrees.values())):.2f}")
        print(f"  - Grado massimo: {max(degrees.values())}")

    # Trova la radice
    root_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'root']

    if len(root_nodes) > 0 and G.number_of_edges() > 0:
        root_id = root_nodes[0]

        try:
            # Calcola le distanze dalla radice
            if nx.is_connected(G):
                distances = nx.single_source_shortest_path_length(G, root_id)
                max_depth = max(distances.values())
                print(f"\nProfondità massima: {max_depth} livelli")

                # Conta nodi per livello
                level_counts = {}
                for node, depth in distances.items():
                    level_counts[depth] = level_counts.get(depth, 0) + 1

                print(f"Distribuzione per livello:")
                for level in sorted(level_counts.keys())[:10]:
                    print(f"  - Livello {level}: {level_counts[level]} nodi")
            else:
                print("\nATTENZIONE: Grafo non connesso!")
                print(f"Componenti connesse: {nx.number_connected_components(G)}")
        except Exception as e:
            print(f"\nErrore nel calcolo profondità: {e}")

    return G

def visualize_results(image_path, skeleton, simplified_skeleton, G):
    """Visualizza i risultati"""
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    fig = plt.figure(figsize=(20, 12))

    # 1. Skeleton originale
    ax1 = plt.subplot(231)
    ax1.imshow(skeleton, cmap='gray')
    ax1.set_title('Skeleton Originale', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. Skeleton semplificato
    ax2 = plt.subplot(232)
    ax2.imshow(simplified_skeleton, cmap='gray')
    ax2.set_title('Skeleton Semplificato', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 3. Punti critici sul skeleton
    ax3 = plt.subplot(233)
    ax3.imshow(simplified_skeleton, cmap='gray')

    if G.number_of_nodes() > 0:
        for node in G.nodes():
            y, x = G.nodes[node]['pos']
            node_type = G.nodes[node]['type']

            if node_type == 'root':
                ax3.scatter(x, y, c='green', s=300, marker='s',
                           edgecolors='white', linewidths=3, zorder=5, label='Radice')
            elif node_type == 'branch':
                ax3.scatter(x, y, c='red', s=200, marker='o',
                           edgecolors='white', linewidths=2, zorder=5, label='Biforcazione')
            else:
                ax3.scatter(x, y, c='blue', s=150, marker='^',
                           edgecolors='white', linewidths=2, zorder=5, label='Estremità')

        # Rimuovi label duplicati
        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    ax3.set_title('Punti Critici Identificati', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # 4. Grafo sovrapposto all'immagine
    ax4 = plt.subplot(234)
    ax4.imshow(original, cmap='gray', alpha=0.6)

    if G.number_of_edges() > 0:
        # Disegna i percorsi
        for u, v, data in G.edges(data=True):
            if 'path' in data and len(data['path']) > 1:
                path = np.array(data['path'])
                ax4.plot(path[:, 1], path[:, 0], 'yellow', linewidth=3, alpha=0.8)

    if G.number_of_nodes() > 0:
        # Disegna i nodi
        for node in G.nodes():
            y, x = G.nodes[node]['pos']
            node_type = G.nodes[node]['type']

            if node_type == 'root':
                ax4.scatter(x, y, c='green', s=300, marker='s',
                           edgecolors='white', linewidths=3, zorder=10)
            elif node_type == 'branch':
                ax4.scatter(x, y, c='red', s=200, marker='o',
                           edgecolors='white', linewidths=2, zorder=10)
            else:
                ax4.scatter(x, y, c='blue', s=150, marker='^',
                           edgecolors='white', linewidths=2, zorder=10)

    ax4.set_title('Grafo Sovrapposto', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # 5. Struttura ad albero
    ax5 = plt.subplot(235)

    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        try:
            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

            # Colori
            node_colors = []
            node_sizes = []
            for node in G.nodes():
                node_type = G.nodes[node]['type']
                if node_type == 'root':
                    node_colors.append('green')
                    node_sizes.append(800)
                elif node_type == 'branch':
                    node_colors.append('red')
                    node_sizes.append(600)
                else:
                    node_colors.append('blue')
                    node_sizes.append(400)

            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                  node_size=node_sizes, ax=ax5,
                                  edgecolors='black', linewidths=2)
            nx.draw_networkx_edges(G, pos, edge_color='gray',
                                  width=3, ax=ax5, alpha=0.6)

            # Etichette
            labels = {node: str(node) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10,
                                   font_weight='bold', ax=ax5)

        except Exception as e:
            ax5.text(0.5, 0.5, f'Errore visualizzazione: {str(e)}',
                    ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'Nessun grafo da visualizzare',
                ha='center', va='center', transform=ax5.transAxes, fontsize=14)

    ax5.set_title('Grafo come Rete', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # 6. Statistiche
    ax6 = plt.subplot(236)
    ax6.axis('off')

    stats_text = f"""STATISTICHE ALBERO

Nodi: {G.number_of_nodes()}
Rami: {G.number_of_edges()}

Radici: {sum(1 for n in G.nodes() if G.nodes[n].get('type')=='root')}
Biforcazioni: {sum(1 for n in G.nodes() if G.nodes[n].get('type')=='branch')}
Estremità: {sum(1 for n in G.nodes() if G.nodes[n].get('type')=='endpoint')}
"""

    if G.number_of_edges() > 0:
        edge_lengths = [d['length'] for _, _, d in G.edges(data=True)]
        stats_text += f"""
Lunghezza rami:
  Media: {np.mean(edge_lengths):.1f} px
  Min: {min(edge_lengths)} px
  Max: {max(edge_lengths)} px
  Totale: {sum(edge_lengths)} px
"""

    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        stats_text += f"""
Connettività:
  Grado medio: {np.mean(list(degrees.values())):.2f}
  Grado max: {max(degrees.values())}
"""

    ax6.text(0.05, 0.5, stats_text, fontsize=11,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax6.set_title('Statistiche', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

def analyze_tree(image_path, min_branch_length=20, min_distance_merge=10):
    """Funzione principale di analisi"""
    print(f"=== ANALISI ALBERO ===")
    print(f"Immagine: {image_path}")
    print(f"Parametri:")
    print(f"  - Lunghezza minima rami: {min_branch_length} px")
    print(f"  - Distanza minima merge punti: {min_distance_merge} px")

    # 1. Estrai skeleton
    print("\n1. Estrazione skeleton...")
    skeleton = extract_skeleton(image_path)
    print(f"   Pixel skeleton: {np.sum(skeleton > 0)}")

    # 2. Semplifica skeleton
    print("\n2. Semplificazione skeleton...")
    simplified_skeleton = simplify_skeleton_pruning(skeleton, min_branch_length)
    print(f"   Pixel dopo semplificazione: {np.sum(simplified_skeleton > 0)}")

    # 3. Trova punti critici
    print("\n3. Identificazione punti critici...")
    branch_points, endpoints = find_critical_points_simple(simplified_skeleton)
    print(f"   Branch points iniziali: {len(branch_points)}")
    print(f"   Endpoints iniziali: {len(endpoints)}")

    # 4. Unisci punti vicini
    print("\n4. Merge punti vicini...")
    if len(branch_points) > 0:
        branch_points = merge_close_points(branch_points, min_distance_merge)
        print(f"   Branch points dopo merge: {len(branch_points)}")

    if len(endpoints) > 0:
        endpoints = merge_close_points(endpoints, min_distance_merge)
        print(f"   Endpoints dopo merge: {len(endpoints)}")

    # 5. Costruisci grafo
    print("\n5. Costruzione grafo...")
    G, point_to_id = build_graph_from_skeleton(
        simplified_skeleton, branch_points, endpoints
    )

    # 6. Analizza
    analyze_tree_structure(G)

    # 7. Visualizza
    print("\n6. Visualizzazione...")
    visualize_results(image_path, skeleton, simplified_skeleton, G)

    return G, simplified_skeleton

# Utilizzo
if __name__ == "__main__":
    try:
        #image_path = "/content/drive/MyDrive/branch/segmentazioni_ottimizzate/Zelkova serrata_branch_1 (55)_mask_opt.png"
        #image_path= "/content/drive/MyDrive/branch/segmentazioni_ottimizzate/Koelreuteria paniculata_branch_1 (78)_mask_opt.png"
        image_path= "/content/drive/MyDrive/branch/segmentazioni_ottimizzate/Lagerstroemia indica_branch_1 (41)_mask_opt.png"
        G, skeleton = analyze_tree(
            image_path,
            min_branch_length=20,  # Aumenta per semplificare
            min_distance_merge=15   # Distanza minima tra punti critici
        )

        print("\n" + "="*50)
        print("ANALISI COMPLETATA!")
        print("="*50)
        print("\nEsporta il grafo con:")
        print("  nx.write_graphml(G, 'tree_graph.graphml')")
        print("  nx.write_gexf(G, 'tree_graph.gexf')")

    except Exception as e:
        print(f"\nERRORE: {e}")
        import traceback
        traceback.print_exc()
