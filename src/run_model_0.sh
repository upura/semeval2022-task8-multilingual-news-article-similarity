python train_nn.py \
    --fold 0 \
    --max_len 512 \
    --num_folds 20 \
    --model bert-base-multilingual-cased \
    --custom_header concat \
    --lr 1e-5
