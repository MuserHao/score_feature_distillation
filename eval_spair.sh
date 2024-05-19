python eval_spair.py \
    --dataset_path ./SPair-71k \
    --save_path ./spair_ft \ # a path to save features
    --dift_model sd \
    --img_size 768 768 \
    --t 261 \
    --up_ft_index 2 \
    --ensemble_size 8