import pyffish

# 1. Configurar la variante a Gardner (5x5) usando el motor de opciones UCI
pyffish.set_option("UCI_Variant", "gardner")

# 2. Conseguir el FEN inicial de esta variante
initial_fen = pyffish.start_fen("gardner")
print("FEN inicial:", initial_fen)

# 3. Obtener los movimientos legales desde un FEN (el segundo argumento es una lista de movimientos UCI previos, puedes dejarla vacía al principio)
legal_moves = pyffish.legal_moves("gardner", initial_fen, [])
print("Movimientos legales:", legal_moves)
