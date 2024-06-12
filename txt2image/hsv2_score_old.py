from hpsv2 import img_score
import hpsv2
import os
import json
from tqdm.auto import tqdm
import numpy as np

# os.makedirs(save_folder, exist_ok=True)
all_prompts = hpsv2.benchmark_prompts('all') 
prompt_keys = ['anime', 'concept-art', 'paintings', 'photo']
all_prompts = all_prompts['photo'][:10]

print(all_prompts)
result_dir = '/home/shilei/projects/score_feature_distillation/txt2image/guid7.5/old'
res_dicts = {}

for subdir in os.listdir(result_dir):

    scores_v21 = []
    scores_v211 = []
    # hpsv2.evaluate(os.path.join(result_dir, subdir), hps_version="v2.1") 
    for idx, prompt in tqdm(enumerate(all_prompts)):
        file_path = os.path.join(result_dir, subdir, f'{idx}.jpg')
        # print(prompt)
        assert os.path.exists(file_path), file_path
        score_v21 = img_score.score(file_path, prompt, hps_version="v2.1")[0]
        scores_v21.append(str(score_v21))
        scores_v211.append(score_v21)
    res_dicts[subdir] = scores_v21
    print(subdir, np.mean(scores_v211))

with open("hpsv2_scores.json", "w") as outfile: 
    json.dump(res_dicts, outfile)
    
    
""""
['A man taking a drink from a water fountain.', 
'Fruit in a jar filled with liquid sitting on a wooden table.', 
'A bathroom sink cluttered with multiple personal care items.', 
'A smiling man is cooking in his kitchen.', 
'A beautiful blue and pink sky overlooking the beach.', 
'A man smiles as he stirs his food in the pot.', 
'Several bikers are lined up in a parking lot.', 
'There is no picture or image sorry sorry', 
'A small car parked in by a vespa.', 
'Several people around some motorcycles on the side of a road.']
"""

"""
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w11z_w10.005t_w21z_w20.005 0.2477
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w21z_w20.005 0.2474
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Falset_w11z_w1-0.005t_w21z_w2-0.005 0.2328
h512w512timeline-1 0.2462
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w1-0.005t_w210z_w2-0.005 0.2451
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w11z_w1-0.005t_w21z_w2-0.005 0.2407
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w1-0.005t_w21z_w2-0.002 0.2512
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w210z_w20.002 0.2394
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w21z_w20.002 0.2349
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.002t_w210z_w20.002 0.2266
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w210z_w20.005 0.2015
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w11z_w10.005t_w21z_w20.005 0.2482
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Truet_w11z_w10.005t_w21z_w20.005 0.251
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w21z_w20.005 0.2314
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w210z_w20.005 0.2335
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.002t_w210z_w20.002 0.2219
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w1-0.005t_w210z_w2-0.002 0.2583
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w11z_w1-0.005t_w210z_w2-0.005 0.236
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Falset_w11z_w10.005t_w21z_w20.005 0.2479
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w210z_w20.002 0.2428
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w21z_w20.002 0.2311
"""