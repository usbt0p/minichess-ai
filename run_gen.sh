#!/usr/bin/env bash

# example execution
# ./run_gen.sh -v gardner -d 2 -t 20 -m auto; ./run_gen.sh -v gardner -d 3 -t 20 -m auto

set -euo pipefail

# Defaults
variant="gardner"
depth=2
out_file=""   # if empty, we'll set it from variant
log_file=""   # if empty, we'll set it from variant

usage() {
  cat <<'USAGE'
Usage: ./run_gen.sh [-v VARIANT] [-d DEPTH] [-t THREADS] [-m MEMORY] [-o OUTPUT_FILE] [--] [extra stockfish args...]

Options:
  -v VARIANT       UCI_Variant value (default: your_variant)
  -d DEPTH         generate_training_data depth (default: 2)
  -o OUTPUT_FILE   output_file_name (default: <variant>.bin)
  -t THREADS       Number of threads for Stockfish (default: 16)
  -m MEMORY        Hash size in GB, or "auto" to use available RAM - 2GB. (default: 2)
  -h               show help

Examples:
  ./run_gen.sh
  ./run_gen.sh -v fairy -d 3
  ./run_gen.sh -v crazyhouse -o crazyhouse.bin
  ./run_gen.sh -v gardner -t 8 -m auto
USAGE
}

# Parse args
threads=16
memory="2"

while getopts ":v:d:o:t:m:h" opt; do
  case "$opt" in
    v) variant="$OPTARG" ;;
    d) depth="$OPTARG" ;;
    o) out_file="$OPTARG" ;;
    t) threads="$OPTARG" ;;
    m) memory="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Missing argument for -$OPTARG" >&2; usage; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

# Derived defaults
if [[ -z "${out_file}" ]]; then # true if the string is null or empty
  base_name="gen_${variant}_d${depth}"
  out_file="${base_name}.bin"
else
  base_name="${out_file%.*}"
fi

log_file="${base_name}.log"

# If files exist, append a suffix to prevent overwriting
i=1
while [[ -f "$out_file" || -f "$log_file" ]]; do
  out_file="${base_name}_${i}.bin"
  log_file="${base_name}_${i}.log"
  ((i++))
done

# Basic validation
if ! [[ "$depth" =~ ^[0-9]+$ ]]; then
  echo "Error: depth must be an integer, got: $depth" >&2
  exit 2
fi

stockfish_bin="/home/usbt0p/TFG/variant-nnue-tools/src/stockfish"
if [[ ! -x "$stockfish_bin" ]]; then
  echo "Error: $stockfish_bin not found or not executable." >&2
  echo "Tip: put stockfish in the current directory or edit stockfish_bin in this script." >&2
  exit 1
fi

# Determine Hash Size in MB
if [[ "$memory" == "auto" ]]; then
  # Get available memory in MB (from column 'available' in `free -m`)
  avail_mb=$(free -m | awk '/^Mem:/{print $7}')
  
  # Leave a 2GB (2048 MB) safety margin
  hash_val=$(( avail_mb - 2048 ))
  
  # Ensure minimum viable size
  if (( hash_val < 16 )); then
    hash_val=16
  fi
else
  if ! [[ "$memory" =~ ^[0-9]+$ ]]; then
    echo "Error: memory must be an integer (GB) or 'auto', got: $memory" >&2
    exit 2
  fi
  hash_val=$(( memory * 1024 ))
fi

echo "== Running Stockfish training data generation =="
echo "variant   : $variant"
echo "depth     : $depth"
echo "threads   : $threads"
echo "memory    : ${memory} (Hash: ${hash_val} MB)"
echo "out_file  : $out_file"
echo "log_file  : $log_file"
echo "stockfish : $stockfish_bin"
echo

# Create a FIFO to feed stdin while we pipe stdout to tee.
fifo="$(mktemp -u "${TMPDIR:-/tmp}/stockfish-uci.XXXXXX")"
mkfifo "$fifo"
cleanup() { rm -f "$fifo"; }
trap cleanup EXIT

# Run stockfish: stdin from fifo, stdout+stderr to terminal+log
"$stockfish_bin" <"$fifo" 2>&1 | tee -a "$log_file" &
sf_pid=$!

# Feed commands into the fifo
cat >"$fifo" <<EOF
uci
setoption name Use NNUE value false
setoption name Threads value $threads
setoption name Hash value $hash_val
setoption name UCI_Variant value $variant
isready
generate_training_data depth $depth count 10000000 random_multi_pv 4 random_multi_pv_diff 100 random_move_count 8 random_move_max_ply 20 write_min_ply 5 eval_limit 10000 set_recommended_uci_options data_format bin output_file_name $out_file
quit
EOF

# Wait for stockfish to exit
wait "$sf_pid"