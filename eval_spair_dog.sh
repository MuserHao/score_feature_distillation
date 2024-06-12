python eval_spair_dog.py \
    --dataset_path ./SPair-71k \
    --save_path ./spair_ft \
    --dift_model sd \
    --img_size 768 768 \
    --t 261 \
    --up_ft_index 1 \
    --ensemble_size 12


## e4: dog per image PCK@0.1: 50.71
## e4: dog per point PCK@0.1: 54.29
