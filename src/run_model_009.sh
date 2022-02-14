python train_nn.py \
    --fold 0 \
    --max_len 256 \
    --num_folds 5 \
    --model bert-base-multilingual-cased \
    --custom_header concat \
    --lr 1e-5
