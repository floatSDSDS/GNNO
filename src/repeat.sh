#python main.py --model_name STRec  --random_seed 0 --batch_size 1024 --gamma_st 0 --min_st_freq 64 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7
#python main.py --model_name STRec  --random_seed 1 --batch_size 1024 --gamma_st 0 --min_st_freq 64 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7
#python main.py --model_name STRec  --random_seed 2 --batch_size 1024 --gamma_st 0 --min_st_freq 64 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7
#python main.py --model_name STRec  --random_seed 3 --batch_size 1024 --gamma_st 0.5 --min_st_freq 1 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7
#python main.py --model_name STRec  --random_seed 4 --batch_size 1024 --gamma_st 0.5 --min_st_freq 1 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7


#python main.py --model_name STRec  --random_seed 0 --batch_size 1024 --gamma_st 0.5 --min_st_freq 8 --emb_size 64 --num_layers 1 --num_heads 16 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Gowalla --gpu 7
#python main.py --model_name STRec  --random_seed 1 --batch_size 1024 --gamma_st 0 --min_st_freq 8 --emb_size 64 --num_layers 1 --num_heads 16 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Gowalla --gpu 7
#python main.py --model_name STRec  --random_seed 2 --batch_size 1024 --gamma_st 0 --min_st_freq 8 --emb_size 64 --num_layers 1 --num_heads 16 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Gowalla --gpu 7
#python main.py --model_name STRec  --random_seed 3 --batch_size 1024 --gamma_st 0 --min_st_freq 8 --emb_size 64 --num_layers 1 --num_heads 16 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Gowalla --gpu 7
#python main.py --model_name STRec  --random_seed 4 --batch_size 1024 --gamma_st 0 --min_st_freq 8 --emb_size 64 --num_layers 1 --num_heads 16 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Gowalla --gpu 7

python main.py --model_name ContraRec --random_seed 0 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --encoder BERT4Rec --gamma 1 --temp 0.2 --batch_size 1024 --dataset 'Grocery_and_Gourmet_Food' --gpu 1
#python main.py --model_name ContraRec --random_seed 1 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --encoder BERT4Rec --gamma 1 --temp 0.2 --batch_size 1024 --dataset 'Grocery_and_Gourmet_Food' --gpu 7
#python main.py --model_name ContraRec --random_seed 2 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --encoder BERT4Rec --gamma 1 --temp 0.2 --batch_size 1024 --dataset 'Grocery_and_Gourmet_Food' --gpu 7
#python main.py --model_name ContraRec --random_seed 3 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --encoder BERT4Rec --gamma 1 --temp 0.2 --batch_size 1024 --dataset 'Grocery_and_Gourmet_Food' --gpu 7
#python main.py --model_name ContraRec --random_seed 4 --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --encoder BERT4Rec --gamma 1 --temp 0.2 --batch_size 1024 --dataset 'Grocery_and_Gourmet_Food' --gpu 7

#python main.py --model_name STRec  --random_seed 0 --batch_size 1024 --gamma_st 0.5 --min_st_freq 1 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7
#python main.py --model_name STRec  --random_seed 1 --batch_size 1024 --gamma_st 0.5 --min_st_freq 1 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7
#python main.py --model_name STRec  --random_seed 2 --batch_size 1024 --gamma_st 0.5 --min_st_freq 1 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7
#python main.py --model_name STRec  --random_seed 3 --batch_size 1024 --gamma_st 0.5 --min_st_freq 1 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7
#python main.py --model_name STRec  --random_seed 4 --batch_size 1024 --gamma_st 0.5 --min_st_freq 1 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --gpu 7

#python main.py --model_name STRec  --random_seed 0 --batch_size 2048 --gamma_st 0.1 --min_st_freq 32 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Beauty --gpu 4
#python main.py --model_name STRec  --random_seed 1 --batch_size 2048 --gamma_st 0.1 --min_st_freq 32 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Beauty --gpu 4
#python main.py --model_name STRec  --random_seed 2 --batch_size 2048 --gamma_st 0.1 --min_st_freq 32 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Beauty --gpu 4

#python main.py --model_name TopoRec  --random_seed 1 --batch_size 4096 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 10 --dataset Gowalla --gpu 7 --num_neg 3 --gamma_st 0
#python main.py --model_name TopoRec  --random_seed 2 --batch_size 4096 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 10 --dataset Gowalla --gpu 7 --num_neg 3 --gamma_st 0

#python main.py --model_name TopoRec  --batch_size 4096 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 10 --dataset Gowalla --gpu 6 --num_neg 3 --gamma_st 0

#python main.py --model_name HardNei --random_seed 0 --batch_size 4096 --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 10 --dataset Grocery_and_Gourmet_Food --gpu 0 --num_neg 3 --fn_split 20 --n_hn 5 --f_hn g --hn0 10
