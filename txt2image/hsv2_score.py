from hpsv2 import img_score
import hpsv2
import os
import json
from tqdm.auto import tqdm
import numpy as np

# os.makedirs(save_folder, exist_ok=True)
all_prompts = hpsv2.benchmark_prompts('all') 
prompt_keys = ['anime', 'concept-art', 'paintings', 'photo']
all_prompts = all_prompts['photo'][:50]

print(all_prompts)
result_dir = '/home/shilei/projects/score_feature_distillation/txt2image/guid7.5'
res_dicts = {}

kwargs1 = {'timeline':-1}
kwargs2 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':1, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
kwargs3 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':1, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
kwargs4 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':1, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
kwargs5 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':1, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}

kwargs6 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
kwargs7 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':1, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
kwargs8 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
kwargs9 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':1, 'z_w1':0.002, 't_w2':10, 'z_w2':0.002}
kwargs10 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.002, 't_w2':10, 'z_w2':0.002}
kwargs11 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}

kwargs35 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0, 't_w2':1, 'z_w2':0}
kwargs36 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0, 't_w2':10, 'z_w2':0}

kwargs = [kwargs1, kwargs2, kwargs3, kwargs4, kwargs5, kwargs6, kwargs7, kwargs8, kwargs9, kwargs10, kwargs11, kwargs35, kwargs36]

"""
h512w512_timeline_-1 0.2455
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_1_z_w1_0.005_t_w2_1_z_w2_0.005 0.2228
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_True_t_w1_1_z_w1_0.005_t_w2_1_z_w2_0.005 0.2363
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_True_t_w1_1_z_w1_0.005_t_w2_1_z_w2_0.005 0.244
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_False_t_w1_1_z_w1_0.005_t_w2_1_z_w2_0.005 0.2242
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0.005_t_w2_1_z_w2_0.005 0.2097
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_1_z_w1_0.005_t_w2_10_z_w2_0.005 0.2115
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0.002_t_w2_1_z_w2_0.002 0.2311
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_1_z_w1_0.002_t_w2_10_z_w2_0.002 0.2201
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0.002_t_w2_10_z_w2_0.002 0.2153
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0.005_t_w2_10_z_w2_0.005 0.2135
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0_t_w2_1_z_w2_0 0.2192
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0_t_w2_10_z_w2_0 0.2141
"""

# kwargs12 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':1, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
# kwargs13 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':1, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
# kwargs14 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':1, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
# kwargs15 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':1, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}

# kwargs16 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':10, 'z_w1':0.005, 't_w2':1, 'z_w2':0.005}
# kwargs17 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':1, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
# kwargs18 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':10, 'z_w1':0.002, 't_w2':1, 'z_w2':0.002}
# kwargs19 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':1, 'z_w1':0.002, 't_w2':10, 'z_w2':0.002}
# kwargs20 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':10, 'z_w1':0.002, 't_w2':10, 'z_w2':0.002}
# kwargs21 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':True, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}

# kwargs = [kwargs12, kwargs13, kwargs14, kwargs15, kwargs16, kwargs17, kwargs18, kwargs19, kwargs20, kwargs21]

"""
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_1_z_w1_0.002_t_w2_1_z_w2_0.002 0.239
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_True_t_w1_1_z_w1_0.002_t_w2_1_z_w2_0.002 0.2363
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_True_t_w1_1_z_w1_0.002_t_w2_1_z_w2_0.002 0.2415
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_False_t_w1_1_z_w1_0.002_t_w2_1_z_w2_0.002 0.234
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_True_t_w1_10_z_w1_0.005_t_w2_1_z_w2_0.005 0.2355
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_True_t_w1_1_z_w1_0.005_t_w2_10_z_w2_0.005 0.2332
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_True_t_w1_10_z_w1_0.002_t_w2_1_z_w2_0.002 0.2311
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_True_t_w1_1_z_w1_0.002_t_w2_10_z_w2_0.002 0.2388
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_True_t_w1_10_z_w1_0.002_t_w2_10_z_w2_0.002 0.2344
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_True_t_w1_10_z_w1_0.005_t_w2_10_z_w2_0.005 0.2213
"""

# kwargs22 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':-0.005}
# kwargs23 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':-0.005, 't_w2':10, 'z_w2':0.005}
# kwargs24 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':0.002, 't_w2':10, 'z_w2':-0.002}
# kwargs25 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':True, 't_w1':10, 'z_w1':-0.002, 't_w2':10, 'z_w2':0.002}

# kwargs26 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':-0.005}
# kwargs27 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':-0.005, 't_w2':10, 'z_w2':0.005}
# kwargs28 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':0.002, 't_w2':10, 'z_w2':-0.002}
# kwargs29 = {'timeline':200, 'extrap1':True, 'extrap2':True, 'pos1':False, 'pos2':False, 't_w1':10, 'z_w1':-0.002, 't_w2':10, 'z_w2':0.002}

# kwargs30 = {'timeline':300, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
# kwargs31 = {'timeline':400, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
# kwargs32 = {'timeline':500, 'extrap1':True, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}        
# kwargs33 = {'timeline':200, 'extrap1':True, 'extrap2':False, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
# kwargs34 = {'timeline':200, 'extrap1':False, 'extrap2':True, 'pos1':True, 'pos2':False, 't_w1':10, 'z_w1':0.005, 't_w2':10, 'z_w2':0.005}
# kwargs = [kwargs22, kwargs23, kwargs24, kwargs25, kwargs26, kwargs27, kwargs28, kwargs29, kwargs30, kwargs31, kwargs32, kwargs33, kwargs34]

"""
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_True_t_w1_10_z_w1_0.005_t_w2_10_z_w2_-0.005 0.2297
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_True_t_w1_10_z_w1_-0.005_t_w2_10_z_w2_0.005 0.2325
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_True_t_w1_10_z_w1_0.002_t_w2_10_z_w2_-0.002 0.23
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_True_pos2_True_t_w1_10_z_w1_-0.002_t_w2_10_z_w2_0.002 0.2375
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_False_t_w1_10_z_w1_0.005_t_w2_10_z_w2_-0.005 0.2317
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_False_t_w1_10_z_w1_-0.005_t_w2_10_z_w2_0.005 0.2023
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_False_t_w1_10_z_w1_0.002_t_w2_10_z_w2_-0.002 0.2262
h512w512_timeline_200_extrap1_True_extrap2_True_pos1_False_pos2_False_t_w1_10_z_w1_-0.002_t_w2_10_z_w2_0.002 0.217
h512w512_timeline_300_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0.005_t_w2_10_z_w2_0.005 0.2104
h512w512_timeline_400_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0.005_t_w2_10_z_w2_0.005 0.2172
h512w512_timeline_500_extrap1_True_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0.005_t_w2_10_z_w2_0.005 0.2181
h512w512_timeline_200_extrap1_True_extrap2_False_pos1_True_pos2_False_t_w1_10_z_w1_0.005_t_w2_10_z_w2_0.005 0.2205
h512w512_timeline_200_extrap1_False_extrap2_True_pos1_True_pos2_False_t_w1_10_z_w1_0.005_t_w2_10_z_w2_0.005 0.2172
"""

pre_name = 'h512w512'
for args in kwargs:
    subdir = pre_name
    for k, v in args.items():
        subdir += '_' + k + '_' + str(v)

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