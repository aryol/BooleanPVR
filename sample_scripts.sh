python3 neural_net.py -model transformer -func PVR -w 3 -agg parity -epochs 100 > parity_transformer_w_3.txt
python3 neural_net.py -model mlp -func PVR -w 3 -agg majority -epochs 100 > majority_mlp_w_3.txt

python3 linear_exp.py -depth 4 > depth4.txt
python3 linear_exp.py -alpha-init 1.5 > init15.txt

