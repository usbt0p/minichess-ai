import pyffish

UNICODE_PIECES = {
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟'
}

def print_board(fen: str):
    """Muestra el tablero 5x5 en la terminal usando caracteres Unicode."""
    board_part = fen.split()[0]
    rows = board_part.split('/')
    print("\n  +---+---+---+---+---+")
    for i, row in enumerate(rows):
        rank = 5 - i
        row_str = f"{rank} |"
        for char in row:
            if char.isdigit():
                for _ in range(int(char)):
                    row_str += "   |"
            else:
                row_str += f" {UNICODE_PIECES.get(char, char)} |"
        print(row_str)
        print("  +---+---+---+---+---+")
    print("    a   b   c   d   e  \n")

import base64
import os

def get_piece_svg_b64(char):
    color_code = 'l' if char.isupper() else 'd'
    piece_code = char.lower()
    filename = f"Chess_{piece_code}{color_code}t45.svg"
    filepath = os.path.join("src", "chess", "svg", filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/svg+xml;base64,{data}"

def export_svg(fen: str, filename: str = "tablero.svg"):
    """Exporta el FEN actual a un archivo de imagen vectorial SVG con las piezas oficiales."""
    board_part = fen.split()[0]
    rows = board_part.split('/')
    
    sq = 60 # Tamaño de la casilla
    width, height = 5 * sq, 5 * sq
    
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    
    # Dibujar el tablero a cuadros
    for r in range(5):
        for c in range(5):
            color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
            svg.append(f'<rect x="{c * sq}" y="{r * sq}" width="{sq}" height="{sq}" fill="{color}" />')

    # Añadir las piezas (incrustando los SVG)
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                b64_data = get_piece_svg_b64(char)
                if b64_data:
                    # Lo escalamos para que llene la casilla
                    svg.append(f'<image href="{b64_data}" x="{c * sq}" y="{r * sq}" width="{sq}" height="{sq}" />')
                else:
                    # Fallback a Unicode si no se encuentra el SVG
                    piece_char = UNICODE_PIECES.get(char, char)
                    color = "#000000" if char.islower() else "#ffffff"
                    svg.append(
                        f'<text x="{c * sq + sq/2}" y="{r * sq + sq/2 + 20}" '
                        f'font-size="{sq * 0.8}" text-anchor="middle" font-family="Arial" '
                        f'fill="{color}">{piece_char}</text>'
                    )
                c += 1
                
    svg.append('</svg>')
    
    with open(filename, "w") as f:
        f.write("\n".join(svg))
    print(f"[*] Imagen SVG guardada en: {filename}")


def play_cli():
    pyffish.set_option("UCI_Variant", "gardner")
    fen = pyffish.start_fen("gardner")
    
    print("="*40)
    print("♟️  MINICHESS GARDNER 5x5 (Pyffish) ♟️")
    print("="*40)
    print("Comandos disponibles:")
    print("  - Mover: escribe la jugada UCI (ej: a2a3)")
    print("  - Exportar: escribe 'export' para guardar imagen")
    print("  - Salir: escribe 'quit'")
    
    while True:
        print_board(fen)
        legal = pyffish.legal_moves("gardner", fen, [])
        
        if not legal:
            print("¡Juego terminado!")
            break
            
        turn = 'Blancas' if ' w ' in fen else 'Negras'
        move = input(f"[{turn}] Mueve > ").strip().lower()
        
        if move in ['quit', 'exit', 'q']:
            print("Saliendo...")
            break
        elif move == 'export':
            export_svg(fen, "minichess_export.svg")
            continue
            
        if move not in legal:
            print(f"\n[!] Movimiento inválido. Legales: {', '.join(legal[:10])}...")
            continue
            
        # pyffish actualiza el FEN si le pasamos el fen inicial y la lista con LA jugada nueva.
        fen = pyffish.get_fen("gardner", fen, [move])

if __name__ == "__main__":
    play_cli()
