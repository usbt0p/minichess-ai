import pyffish
from src.utils.utils import print_board, export_svg


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
