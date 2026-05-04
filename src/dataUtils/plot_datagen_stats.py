import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from src.utils.utils import export_svg


def parse_stats(filepath):
    # Diccionario para almacenar las distintas métricas extraídas
    data = {
        "boards": {},
        "moves_by_type": {},
        "moves_by_piece": {},
        "eval_imbalances": {},
        "results": {},
        "positions_by_piece_count": {},
        "score_distribution": {},
        "endgame_configurations": {},
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parseo de frecuencias en los tableros (matrices 5x5)
        matched_board = False
        for board_type in [
            "King square distribution:",
            "Move from square distribution:",
            "Move to square distribution:",
        ]:
            if line.startswith(board_type):
                matched_board = True
                for color in ["White", "Black"]:
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith(color):
                        i += 1

                    if i < len(lines):
                        board_name = f"{board_type.replace(':', '')} - {color}"
                        board_matrix = []
                        for _ in range(5):
                            i += 1
                            row = [int(x) for x in lines[i].split()]
                            board_matrix.append(row)
                        data["boards"][board_name] = board_matrix
                break

        if matched_board:
            continue

        # Parseo de datos categóricos simples
        if line.startswith("Number of moves by type:"):
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                parts = lines[i].strip().split(": ")
                if len(parts) == 2 and parts[0] != "Total":
                    data["moves_by_type"][parts[0]] = int(parts[1])
                i += 1
            continue

        if line.startswith("Number of moves by piece type:"):
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                parts = lines[i].strip().split(": ")
                if len(parts) == 2:
                    data["moves_by_piece"][parts[0]] = int(parts[1])
                i += 1
            continue

        if line.startswith('Number of "simple eval" imbalances'):
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                parts = lines[i].strip().split(":")
                if len(parts) == 2:
                    data["eval_imbalances"][int(parts[0].strip())] = int(
                        parts[1].strip()
                    )
                i += 1
            continue

        if line.startswith("Distribution of results:"):
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                parts = lines[i].strip().split(": ")
                if len(parts) == 2:
                    data["results"][parts[0]] = int(parts[1])
                i += 1
            continue

        if line.startswith("Number of positions by piece count:"):
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                parts = lines[i].strip().split(": ")
                if len(parts) == 2:
                    data["positions_by_piece_count"][int(parts[0])] = int(parts[1])
                i += 1
            continue

        if line.startswith("Score distribution:"):
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                parts = lines[i].strip().split(": ")
                if len(parts) == 2:
                    key = parts[0]
                    # Solo nos interesan los valores numéricos, Min y Max los ignoramos 
                    # porque podemos sacarlos de las keys.
                    if key not in ["Min Score", "Max Score"]:
                        data["score_distribution"][int(key)] = int(parts[1])
                i += 1
            continue

        if line.startswith("Distribution of endgame configurations"):
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                parts = lines[i].split("):")
                if len(parts) == 2:
                    config_part = parts[0].split("(")[0].strip()
                    config = " ".join(config_part.split())
                    counts = parts[1].strip().split()
                    if counts:
                        data["endgame_configurations"][config] = {
                            "count": int(counts[0]),
                            "perf": counts[-1]
                        }
                i += 1
            continue

        i += 1

    return data


def plot_3d_histogram(matrix, title, flip=False):
    '''histograma 3d (barras en 3d) para los tableros porque asimila topológicamente
    el tablero 2D; donde la altura y el color denotan la densidad conjunta.
    Es ideal para identificar "hotspots" (puntos calientes) de forma bastante intuitiva.

    flip: si es True, invierte el eje y para que el histograma se vea como el tablero desde la perspectiva del jugador de negras
    '''
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    matrix = np.array(matrix)
    rows, cols = matrix.shape

    # x e y definen el plano 2D del tablero 5x5
    x_data, y_data = np.meshgrid(np.arange(cols), np.arange(rows))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = np.zeros(rows * cols)

    dz = matrix.flatten()
    dx = np.ones(rows * cols) * 0.8
    dy = np.ones(rows * cols) * 0.8

    
    # TODO esto hay que mirarlo, me parece que los datos vienen en 
    # espejo, así que no se muy bien como hacerlo
    
    # si flip es True, invertimos el eje y para que el histograma se vea como 
    # el tablero desde la perspectiva del jugador de negras
    # if flip:
    #     # set x ticks to match columns a, b, c, d, e
    #     ax.set_xticks(np.arange(cols))
    #     ax.set_xticklabels(["a", "b", "c", "d", "e"])
    #     y_data = rows - y_data - 1
    #     # and set y ticks to be inverted
    #     ax.set_yticks(np.arange(rows))
    #     ax.set_yticklabels(["1", "2", "3", "4", "5"])
    #     ax.invert_yaxis()
    # else:
    #     # set x ticks to match columns a, b, c, d, e
    #     ax.set_xticks(np.arange(cols))
    #     ax.set_xticklabels(["e", "d", "c", "b", "a"])
    #     ax.set_yticks(np.arange(rows))
    #     ax.set_yticklabels(["5", "4", "3", "2", "1"])

    # colormap 'jet' mapeado a las alturas (emulando la imagen pedida)
    cmap = plt.get_cmap("jet")
    max_val = np.max(dz) if np.max(dz) > 0 else 1
    colors = cmap(dz / max_val)

    ax.bar3d(x_data, y_data, z_data, dx, dy, dz, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Columnas (X)")
    ax.set_ylabel("Filas (Y)")
    ax.set_zlabel("Frecuencia")


def plot_bar_chart(data_dict, title, xlabel, ylabel):
    # Los gráficos de barras son sencillos y claros para enumerar datos puramente categóricos
    # que no tienen orden continuo, nos permite comparar frecuencias independientemente.
    fig = plt.figure(figsize=(8, 6))
    categories = list(data_dict.keys())
    values = list(data_dict.values())

    # add the exact values inside the bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
    plt.bar(categories, values, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

def plot_endgame_bar_chart(endgames_dict, title, xlabel, ylabel):
    fig = plt.figure(figsize=(10, 6))
    categories = list(endgames_dict.keys())
    counts = [v["count"] for v in endgames_dict.values()]
    perfs = [v["perf"] for v in endgames_dict.values()]

    bars = plt.bar(categories, counts, color="skyblue", edgecolor="black")
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # perf% en el centro
        plt.text(bar.get_x() + bar.get_width() / 2, height / 2, perfs[i], ha='center', va='center', color='black', fontweight='bold', fontsize=10)
        # conteo arriba
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{counts[i]}", ha='center', va='bottom', fontsize=9)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


def plot_line_chart(data_dict, title, xlabel, ylabel):
    # Un gráfico de líneas con gradiente funciona perfectamente para distribuciones secuenciales,
    # ya que resalta visualmente la forma de campana o caída y su continuidad paramétrica a lo largo
    # del eje de evaluación (o del número de piezas).
    fig = plt.figure(figsize=(8, 6))

    sorted_items = sorted(data_dict.items())
    x = [item[0] for item in sorted_items]
    y = [item[1] for item in sorted_items]

    # use integer ticks and rotate them for better visibility
    plt.xticks(np.arange(min(x), max(x)+1, 1))
    plt.xticks(rotation=45, ha="right")
    plt.plot(x, y, marker="o", linestyle="-", color="indigo")
    plt.fill_between(x, y, color="indigo", alpha=0.1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()


def plot_binned_histogram(data_dict, title, xlabel, ylabel, num_bins=30):
    fig = plt.figure(figsize=(10, 6))
    
    if not data_dict:
        return
        
    # Extraer todos los scores y multiplicarlos por su frecuencia
    # (data_dict tiene formato {score: frecuencia})
    scores = list(data_dict.keys())
    min_score = min(scores)
    max_score = max(scores)
    
    # Asegurarnos de que los bins estén centrados en 0 simétricamente
    abs_max = max(abs(min_score), abs(max_score))
    
    # Creamos un array plano repitiendo los valores para matplotlib.hist
    # o mejor aún, calculamos los bines directamente con numpy usando weights.
    bins = np.linspace(-abs_max, abs_max, num_bins + 1)
    
    values = list(data_dict.values())
    
    n, bins_out, patches = plt.hist(scores, bins=bins, weights=values, color="coral", edgecolor="black", alpha=0.8)
    
    # Anotar el número exacto de conteos en cada bin
    for i in range(len(patches)):
        count = int(n[i])
        if count > 0:
            x_center = patches[i].get_x() + patches[i].get_width() / 2
            plt.text(x_center, count, f'{count}', ha='center', va='bottom', fontsize=8, rotation=90)
            
    plt.title(f"{title} (Min: {min_score}, Max: {max_score})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()



def get_endgame_str(fen):
    board_part = fen.split()[0]
    white, black = [], []
    for char in board_part:
        if char.isalpha():
            if char.isupper():
                white.append(char)
            else:
                black.append(char.upper())
    
    order = {'K': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5}
    white.sort(key=lambda x: order[x])
    black.sort(key=lambda x: order[x])
    
    return "".join(white) + " v" + "".join(black)

def generate_fen_visualizations(data_dict, stats_filepath):
    from src.utils.utils import get_svg_board
    if not data_dict.get("endgame_configurations"):
        return
        
    output_dir = os.path.dirname(stats_filepath)
    dataset_file = None
    for f in os.listdir(output_dir):
        if f.endswith(".txt") and "stats" not in f:
            dataset_file = os.path.join(output_dir, f)
            break
            
    if not dataset_file:
        print("No se encontró archivo de dataset raw para exportar FENs.")
        return
        
    print(f"Buscando FENs en {dataset_file}...")
    
    # 1. Los top 10 endgame configurations
    top_endgames = sorted(data_dict["endgame_configurations"].items(), key=lambda x: x[1]["count"], reverse=True)[:10]
    top_endgame_names = [x[0] for x in top_endgames]
    found_endgames = {name: [] for name in top_endgame_names}
    random_fens = []
    
    with open(dataset_file, "r") as f:
        for line in f:
            if line.startswith("fen "):
                fen = line[4:].strip()
                
                if random.random() < 0.001 and len(random_fens) < 6:
                    random_fens.append(fen)
                    
                endgame_str = get_endgame_str(fen)
                if endgame_str in found_endgames and len(found_endgames[endgame_str]) < 1:
                    found_endgames[endgame_str].append(fen)
                    
                all_done = len(random_fens) >= 6 and all(len(v) >= 1 for v in found_endgames.values())
                if all_done:
                    break
                    
    svg_out_dir = os.path.join(output_dir, "fens_svg")
    os.makedirs(svg_out_dir, exist_ok=True)
    
    def save_svg_grid(items_list, filename, main_title):
        board_size = 300
        margin = 60
        text_height = 80
        cell_width = board_size + margin
        cell_height = board_size + text_height + margin

        total_items = len(items_list)
        cols = 4 
        if total_items == 6: cols = 3
        elif total_items == 10: cols = 5
        rows = (total_items + cols - 1) // cols

        width = cols * cell_width
        height = rows * cell_height + 100

        svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: white;">']
        svg.append(f'<text x="{width/2}" y="60" font-size="36" font-family="Arial" text-anchor="middle" font-weight="bold">{main_title}</text>')

        for idx, item in enumerate(items_list):
            fen = item["fen"]
            title = item["title"]
            subtitle = item.get("subtitle", "")
            
            x_idx = idx % cols
            y_idx = idx // cols
            
            x_pos = margin/2 + x_idx * cell_width
            y_pos = 120 + y_idx * cell_height
            
            svg.append(f'<text x="{x_pos + board_size/2}" y="{y_pos - 35}" font-size="22" font-family="Arial" text-anchor="middle" font-weight="bold">{title}</text>')
            if subtitle:
                svg.append(f'<text x="{x_pos + board_size/2}" y="{y_pos - 10}" font-size="16" font-family="Arial" text-anchor="middle" fill="#555">{subtitle}</text>')
            
            board_svg = get_svg_board(fen)
            svg.append(f'<g transform="translate({x_pos}, {y_pos})">')
            svg.append(board_svg)
            svg.append('</g>')

        svg.append('</svg>')
        
        with open(filename, 'w') as f:
            f.write("\n".join(svg))
        print(f"[*] Imagen conjunta guardada en: {filename}")

    endgame_items = []
    for endgame, fens in found_endgames.items():
        perf = data_dict["endgame_configurations"][endgame]["perf"]
        count = data_dict["endgame_configurations"][endgame]["count"]
        for fen in fens:
            endgame_items.append({"fen": fen, "title": f"{endgame} (WR: {perf})", "subtitle": f"Muestras: {count:,}"})
            
    random_items = []
    for i, fen in enumerate(random_fens):
        random_items.append({"fen": fen, "title": f"Random #{i+1}"})
        
    save_svg_grid(endgame_items, os.path.join(svg_out_dir, "top_endgames.svg"), "Top 10 Finales Frecuentes")
    save_svg_grid(random_items, os.path.join(svg_out_dir, "random_fens.svg"), "Posiciones Aleatorias (Random)")

def main():
    # Intenta leer argumento de linea de comandos, sino buscará el archivo por defecto.
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("Ingrese la ruta del archivo de estadísticas: ")

    if not os.path.exists(filepath):
        print(f"Error: {filepath} no existe.")
        sys.exit(1)

    data = parse_stats(filepath)

    # Graficar histogramas 3d para los tableros
    for name, matrix in data["boards"].items():
        if "Black" in name:
            plot_3d_histogram(matrix, name, flip=True)
        else:
            plot_3d_histogram(matrix, name)

    # Graficar datos categóricos puros
    if data["moves_by_type"]:
        plot_bar_chart(
            data["moves_by_type"], "Movimientos por Tipo", "Tipo", "Cantidad"
        )
    if data["moves_by_piece"]:
        plot_bar_chart(
            data["moves_by_piece"], "Movimientos por Pieza", "Pieza", "Cantidad"
        )
    if data["results"]:
        plot_bar_chart(
            data["results"], "Distribución de Resultados", "Resultado", "Cantidad"
        )

    # Graficar distribuciones en eje continuo
    if data["eval_imbalances"]:
        plot_line_chart(
            data["eval_imbalances"],
            "Evaluaciones Simples (Imbalances)",
            "Diferencia de Eval",
            "Cantidad",
        )
    if data["positions_by_piece_count"]:
        plot_line_chart(
            data["positions_by_piece_count"],
            "Posiciones por Cantidad de Piezas",
            "Nº de Piezas",
            "Cantidad",
        )
    if data["score_distribution"]:
        plot_binned_histogram(
            data["score_distribution"],
            "Distribución de Evaluación (Score)",
            "Score",
            "Frecuencia",
            num_bins=30
        )

    if data.get("endgame_configurations"):
        top_endgames = dict(sorted(data["endgame_configurations"].items(), key=lambda x: x[1]["count"], reverse=True)[:10])
        plot_endgame_bar_chart(
            top_endgames,
            "Top 10 Finales Frecuentes",
            "Final",
            "Cantidad"
        )

    # Extraer FENs de muestra y visualizarlos en SVG
    generate_fen_visualizations(data, filepath)

    # guardar en imágenes en la carpeta de origen de los datos
    output_dir = os.path.dirname(filepath)
    if not os.path.exists(os.path.join(output_dir, "stat_plots")):
        os.mkdir(os.path.join(output_dir, "stat_plots"))
    output_dir = os.path.join(output_dir, "stat_plots")
    
    for i, fig in enumerate(plt.get_fignums()):
        print(f"Guardando figura {i} en {output_dir}")
        plt.figure(fig)
        plt.savefig(os.path.join(output_dir, f"plot_{i}.png"))

    plt.show()

if __name__ == "__main__":
    main()
