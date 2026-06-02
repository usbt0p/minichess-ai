python3 src/training/train_transformer.py data/gardner_depth3_incomplete/gen_gardner_d3.txt 32 | tee logs/exps/trnsf_d3_32.log; 
python3 src/training/train_transformer.py data/gardner_depth3_incomplete/gen_gardner_d3.txt 128 | tee logs/exps/trnsf_d3_128.log; 
python3 src/training/train_transformer.py data/gardner_depth3_incomplete/gen_gardner_d3.txt 64 | tee logs/exps/trnsf_d3_64.log; 
python3 src/training/train_transformer.py data/gardner_depth3_incomplete/gen_gardner_d3.txt 256 | tee logs/exps/trnsf_d3_256.log; 
python3 src/training/train_transformer.py data/gardner_depth3_incomplete/gen_gardner_d3.txt 512 | tee logs/exps/trnsf_d3_512.log; 
python3 src/training/train_transformer.py data/gardner_depth3_incomplete/gen_gardner_d3.txt 768 | tee logs/exps/trnsf_d3_768.log;   

python3 src/training/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 32 | tee logs/exps/trnsf_d2_32.log; 
python3 src/training/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 64 | tee logs/exps/trnsf_d2_64.log; 
python3 src/training/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 128 | tee logs/exps/trnsf_d2_128.log; 
python3 src/training/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 256 | tee logs/exps/trnsf_d2_256.log; 
python3 src/training/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 512 | tee logs/exps/trnsf_d2_512.log; 
python3 src/training/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 768 | tee logs/exps/trnsf_d2_768.log;

python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 32 | tee logs/exps/trnsf_d4_32.log; 
python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 64 | tee logs/exps/trnsf_d4_64.log; 
python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 128 | tee logs/exps/trnsf_d4_128.log; 
python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 256 | tee logs/exps/trnsf_d4_256.log; 
python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 512 | tee logs/exps/trnsf_d4_512.log; 
python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 768 | tee logs/exps/trnsf_d4_768.log;   

python3 src/training/train_transformer.py data/merged/merged_gardner.txt 32 | tee logs/exps/trnsf_merged_32.log; 
python3 src/training/train_transformer.py data/merged/merged_gardner.txt 64 | tee logs/exps/trnsf_merged_64.log; 
python3 src/training/train_transformer.py data/merged/merged_gardner.txt 128 | tee logs/exps/trnsf_merged_128.log; 
python3 src/training/train_transformer.py data/merged/merged_gardner.txt 256 | tee logs/exps/trnsf_merged_256.log; 
python3 src/training/train_transformer.py data/merged/merged_gardner.txt 512 | tee logs/exps/trnsf_merged_512.log; 
python3 src/training/train_transformer.py data/merged/merged_gardner.txt 768 | tee logs/exps/trnsf_merged_768.log;   
