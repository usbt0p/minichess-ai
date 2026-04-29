# utility script to get stats since the command line of stockfish is so shitty

# get input file and out file from arguments

input_file=$1
output_file=$2

# if no output file, by default write it to the same directory as input file
if [ -z "$output_file" ]; then
    output_file="$(dirname "$input_file")/stats.txt"
fi

./variant-nnue-tools/src/stockfish << EOF
setoption name UCI_Variant value gardner
gather_statistics all input_file $input_file output_file $output_file
quit
EOF