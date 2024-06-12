python eval_spair.py \
    --dataset_path ./SPair-71k \
    --save_path ./spair_ft \
    --dift_model sd \
    --img_size 768 768 \
    --t 261 \
    --up_ft_index 1 \
    --ensemble_size 12

## e1: dog per image PCK@0.1: 39.28
## e1: dog per point PCK@0.1: 42.13
## e4: dog per image PCK@0.1: 50.71
## e4: dog per point PCK@0.1: 54.29
## e8: dog per image PCK@0.1: 52.00
## e8: dog per point PCK@0.1: 55.58
## e12: dog per image PCK@0.1: 51.80
## e12: dog per point PCK@0.1: 55.35
