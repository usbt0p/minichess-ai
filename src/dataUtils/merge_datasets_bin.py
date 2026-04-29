import os
import sys
import time

def merge_and_deduplicate_bins(input_files, output_file):
    """
    Lee archivos binarios de Fairy-Stockfish (PackedSfenValue = 72 bytes),
    usa los primeros 64 bytes (PackedSfen) como identificador único de posición,
    y guarda solo las posiciones únicas en output_file.
    Esto permite generar datasets haciendo distintas runs de generación para distintas profundidades
    y luego unirlos para tener un dataset más grande, y también cancelar generaciones cuando esta decelera
    por el hash de filtro para sumarlas todas.
    
    El orden de input_files importa: si pasas depth4 antes que depth2,
    se guardará la evaluación de depth4 y se ignorará la duplicada de depth2.
    """
    seen_positions = set()
    total_read = 0
    total_written = 0
    CHUNK_SIZE = 72
    
    start_time = time.time()
    
    with open(output_file, 'wb') as f_out:
        for file_path in input_files:
            if not os.path.exists(file_path):
                print(f"[!] Archivo no encontrado, saltando: {file_path}")
                continue
                
            print(f"[*] Procesando {file_path}...")
            file_read = 0
            file_written = 0
            
            with open(file_path, 'rb') as f_in:
                while True:
                    chunk = f_in.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    
                    if len(chunk) != CHUNK_SIZE:
                        print(f"[!] Warning: Chunk final de {len(chunk)} bytes (incompleto) ignorado.")
                        break
                    
                    total_read += 1
                    file_read += 1
                    
                    # Los primeros 64 bytes son la estructura PackedSfen (el tablero)
                    position_hash = chunk[:64]
                    
                    if position_hash not in seen_positions:
                        seen_positions.add(position_hash)
                        f_out.write(chunk)
                        total_written += 1
                        file_written += 1
                        
            print(f"    -> Leídas {file_read} posiciones, escritas {file_written} posiciones (ignoradas {file_read - file_written} duplicadas).")
            
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("Resumen de Fusión (Binario):")
    print("="*50)
    print(f"Posiciones totales procesadas:  {total_read:,}")
    print(f"Posiciones únicas guardadas:    {total_written:,}")
    print(f"Duplicados eliminados:          {total_read - total_written:,}")
    print(f"Tiempo total:                   {elapsed:.2f} segundos")
    print(f"Archivo guardado en:            {output_file}")
    
    print("\n[+] Siguiente paso recomendado:")
    print(f"    1. Extraer stats: ./variant-nnue-tools/build/tools/stats {output_file}")
    print(f"    2. Convertir a texto para entrenamiento: ./variant-nnue-tools/build/tools/convert_plain {output_file} data/merged_dataset.txt")

if __name__ == "__main__":
    # Ordenados de mayor a menor calidad (depth4 > depth3 > depth2)
    # Así, si una posición está repetida, nos quedamos con la mejor evaluación.
    INPUTS = [
        "data/gardner_depth4/gardner.bin",
        "data/gardner_depth3_(incomplete)/gen_gardner_d3.bin",
        "data/gardner_depth2/gen_gardner_d2.bin"
    ]
    
    OUTPUT = "data/merged_gardner.bin"
    
    merge_and_deduplicate_bins(INPUTS, OUTPUT)
