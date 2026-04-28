# convert the three binary files to txt automatically. in order for the stockfish script to read all the args
# we use eof, which is for the shell to understand the multi-line input

# first, check if the files already exist to not append innecesary data

if [ -f "data/gardner_depth2/gen_gardner_d2.txt" ]; then
    echo "data/gardner_depth2/gen_gardner_d2.txt already exists. Skipping..."
    exit 0
fi

./variant-nnue-tools/src/stockfish << 'EOF'
setoption name UCI_Variant value gardner
convert_plain targetfile data/gardner_depth2/gen_gardner_d2.bin output_file_name data/gardner_depth2/gen_gardner_d2.txt
quit
EOF

./variant-nnue-tools/src/stockfish << 'EOF'
setoption name UCI_Variant value gardner
convert_plain targetfile data/gardner_depth3_(incomplete)/gen_gardner_d3.bin output_file_name data/gardner_depth3_(incomplete)/gen_gardner_d3.txt
quit
EOF

./variant-nnue-tools/src/stockfish << 'EOF'
setoption name UCI_Variant value gardner
convert_plain targetfile data/gardner_depth4/gardner.bin output_file_name data/gardner_depth4/gen_gardner_d4.txt
quit
EOF