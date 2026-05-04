import os
import subprocess
import time
import re
import signal
import sys
import matplotlib.pyplot as plt

def handle_interrupt(sig, frame):
    print("\n[!] Interrupción del usuario (Ctrl+C). Saliendo del benchmark limpiamente...")
    sys.exit(0)

# Manejar el Ctrl+C
signal.signal(signal.SIGINT, handle_interrupt)

def create_subsets(source_file, target_dir, sizes_in_samples):
    os.makedirs(target_dir, exist_ok=True)
    subset_files = []
    
    for size in sizes_in_samples:
        lines_to_read = size * 6  # Cada instancia ocupa 6 líneas en el dataset
        target_file = os.path.join(target_dir, f"subset_{size}.txt")
        subset_files.append((size, target_file))
        
        if os.path.exists(target_file):
            print(f"[*] El subset {size} ya existe en {target_file}")
            continue
            
        print(f"[*] Creando subset con {size} instancias ({lines_to_read} líneas)...")
        # Usamos head porque es rapidísimo para leer el principio del archivo
        subprocess.run(f"head -n {lines_to_read} {source_file} > {target_file}", shell=True, check=True)
        
    return subset_files

def run_benchmark(subset_files):
    results = []
    
    for size, file_path in subset_files:
        print(f"\n{'='*50}\nBenchmarking tamaño del dataset: {size}\n{'='*50}")
        
        log_file = f"src/benchmarks/logs_{size}.txt"
        
        # Ejecutamos el baseline, asumiendo que el venv se carga con PYTHONPATH=. 
        cmd = ["python3", "src/models/baseline.py", file_path]
        
        start_time = time.time()
        
        best_mean_acc, best_move_acc, best_res_acc = 0.0, 0.0, 0.0
        total_time_str = ""
        epochs = 0
        
        # Guardamos en un log pero también imprimimos por pantalla (como 'tee')
        with open(log_file, "w") as f_log:
            # Usamos PYTHONPATH=. por si hace falta para que Python encuentre los módulos
            env = os.environ.copy()
            env["PYTHONPATH"] = "." 
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                env=env
            )
            
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    f_log.write(line)
                    
                    # Parseamos la salida para extraer métricas
                    if "Best mean accuracy:" in line:
                        m = re.search(r"([\d\.]+)%", line)
                        if m: best_mean_acc = float(m.group(1))
                    elif "Best move accuracy:" in line:
                        m = re.search(r"([\d\.]+)%", line)
                        if m: best_move_acc = float(m.group(1))
                    elif "Best result accuracy:" in line:
                        m = re.search(r"([\d\.]+)%", line)
                        if m: best_res_acc = float(m.group(1))
                    elif "took" in line and "train_model" in line:
                        m = re.search(r"took ([\d:\.]+)", line)
                        if m: total_time_str = m.group(1)
                    elif "Epoch" in line and "/" in line:
                        epochs += 1
                        
                process.wait()
            except KeyboardInterrupt:
                print("\n[!] Matando el proceso hijo de entrenamiento...")
                process.kill()
                process.wait()
                raise KeyboardInterrupt
                
        # Parsear el tiempo (formato H:MM:SS.xxx)
        try:
            h, m, s = total_time_str.split(':')
            total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            # Fallback por si acaso
            total_seconds = time.time() - start_time
            
        time_per_epoch = total_seconds / epochs if epochs > 0 else 0
        
        results.append({
            'size': size,
            'move_acc': best_move_acc,
            'res_acc': best_res_acc,
            'mean_acc': best_mean_acc,
            'total_time': total_seconds,
            'time_per_epoch': time_per_epoch
        })
        
    return results

def plot_results(results):
    sizes = [r['size'] for r in results]
    move_accs = [r['move_acc'] for r in results]
    res_accs = [r['res_acc'] for r in results]
    times_per_epoch = [r['time_per_epoch'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Accuracy
    ax1.plot(sizes, move_accs, marker='o', label='Move Accuracy (%)')
    ax1.plot(sizes, res_accs, marker='s', label='Result Accuracy (%)')
    #ax1.set_xscale('log')
    ax1.set_xlabel('Tamaño del Dataset (instancias)')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Precisión vs Datos de Entrenamiento')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    
    # Plot Time
    ax2.plot(sizes, times_per_epoch, marker='^', color='r', label='Tiempo por Época (s)')
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.set_xlabel('Tamaño del Dataset (instancias)')
    ax2.set_ylabel('Tiempo (segundos)')
    ax2.set_title('Escalado del Tiempo de Entrenamiento')
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("src/benchmarks/scaling_results.png")
    print("\n[*] Gráficas guardadas en src/benchmarks/scaling_results.png")

if __name__ == "__main__":
    # --- AJUSTA ESTOS PARÁMETROS SI ES NECESARIO ---
    # SOURCE_FILE = "data/merged/merged_gardner.txt"
    # TARGET_DIR = "data/subsets_merged"
    
    # Tamaños de los subsets en instancias.
    # SIZES = [50_000, 100_000, 500_000, 1_000_000, 1_800_000, 3_000_000, 6_000_000, 10_000_000, 
    # 20_000_000, 25_445_963]

    SOURCE_FILE = "data/gardner_depth2/gen_gardner_d2.txt"
    TARGET_DIR = "data/subsets_d2"

    SIZES = [50_000, 100_000, 500_000, 1_000_000, 1_800_000, 3_000_000, 6_000_000, 10_000_000]
    
    print("[1] Creando subsets de datos...")
    subset_files = create_subsets(SOURCE_FILE, TARGET_DIR, SIZES)
    
    print("\n[2] Ejecutando entrenamientos en serie...")
    results = run_benchmark(subset_files)
    
    print("\n[3] Generando gráficas de resultados...")
    plot_results(results)
    
    print("\n" + "="*50)
    print("Resumen Final:")
    print("="*50)
    for r in results:
        print(f"Size: {r['size']:<8} | Mean Acc: {r['mean_acc']:5.2f}% | Best Move Acc: {r['move_acc']:5.2f}% | Best Result Acc: {r['res_acc']:5.2f}% | Time/Epoch: {r['time_per_epoch']:.2f}s")
