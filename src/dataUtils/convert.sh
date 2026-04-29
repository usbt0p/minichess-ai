# convert a file from .bin to .txt, bypassing the shitty command line
# we use eof, which is for the shell to understand the multi-line input

# first, check if the files already exist to not append innecesary data

input_file=$1
output_file=$2
overwrite=$3

if [ -f "$output_file" ] && [ "$overwrite" = "" ]; then
    echo "$output_file already exists. Skipping..."
    exit 0
fi 

if [ "$overwrite" = "overwrite" ]; then
    echo "$output_file already exists. Overwriting..."
    rm "$output_file"
fi

echo "Converting $input_file to $output_file..."

./variant-nnue-tools/src/stockfish << EOF
setoption name UCI_Variant value gardner
convert_plain targetfile $input_file output_file_name $output_file
quit
EOF