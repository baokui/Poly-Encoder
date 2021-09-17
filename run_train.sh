export CUDA_VISIBLE_DEVICES="0"
nohup python train.py \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model_small \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/train_data/ \
    --use_pretrain \
    --architecture poly \
    --max_contexts_length 32 \
    --max_response_length 64 \
    --poly_m 16 >> log/train.log 2>&1 &

export CUDA_VISIBLE_DEVICES="1"
nohup python train.py \
    --gpu 1 \
    --bert_model /search/odin/guobk/data/model/bert-base-chinese \
    --output_dir /search/odin/guobk/data/data_polyEncode/vpa/model_small_all \
    --train_dir /search/odin/guobk/data/data_polyEncode/vpa/train_data_all/ \
    --use_pretrain \
    --architecture poly \
    --max_contexts_length 32 \
    --max_response_length 64 \
    --poly_m 16 >> log/train-all.log 2>&1 &