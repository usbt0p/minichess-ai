import sys
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_stats(filepath):
    # Diccionario para almacenar las distintas métricas extraídas
    data = {
        "boards": {},
        "moves_by_type": {},
        "moves_by_piece": {},
        "eval_imbalances": {},
        "results": {},
        "positions_by_piece_count": {},
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

        i += 1

    return data


def plot_3d_histogram(matrix, title):
    # Utilizamos un histograma 3d (barras en 3d) para los tableros porque asimila topológicamente
    # nuestro tablero 2D; donde la altura y el color denotan la densidad conjunta.
    # Es ideal para identificar "hotspots" (puntos calientes) de forma bastante intuitiva.
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

    # Colormap 'jet' mapeado a las alturas (emulando la imagen pedida)
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

    plt.bar(categories, values, color="skyblue", edgecolor="black")
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

    plt.plot(x, y, marker="o", linestyle="-", color="indigo")
    plt.fill_between(x, y, color="indigo", alpha=0.1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()


def main():
    # Intenta leer argumento de linea de comandos, sino buscará el archivo por defecto.
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Fallback a la ruta actual que ha devuelto el comando \`find\`
        # Ajustamos a la ruta específica o buscamos global si no la hay
        filepath = "data/gardner_depth4/stats_corrected_names.txt"
        if not os.path.exists(filepath):
            # Posible estructura alternativa
            filepath = "logs/stats_corrected_names.txt"

    if not os.path.exists(filepath):
        print(f"Error: {filepath} no existe.")
        sys.exit(1)

    data = parse_stats(filepath)

    # Graficar histogramas 3d para los tableros
    for name, matrix in data["boards"].items():
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
