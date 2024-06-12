import os
import json
import matplotlib.pyplot as plt
folder = '/home/shilei/projects/score_feature_distillation/plot_res_tz/plots_dog_grad4_evaltz/jsons'
folder = '/home/shilei/projects/score_feature_distillation/plot_res_tz/twosides_acc_pos/plots_dog_grad4_evaltz/jsons'
files = []
for file in os.listdir(folder):
    files.append(file)
files = ['dog_Per_Point_evaluation_curve_-1_-1_10_0.005.json',
         'dog_Per_Point_evaluation_curve_5_-1_10_0.005.json', 
         'dog_Per_Point_evaluation_curve_8_-1_10_0.005.json', 
         'dog_Per_Point_evaluation_curve_11_-1_10_0.005.json', ]
plt.figure()
idx = 0
for file in files:
    # if 'Image' not in file:
    #     continue

    with open(os.path.join(folder, file)) as f:
        data = json.loads(f.read())
    
    keys = []
    values = []
    for k, v in data.items():
        keys.append(int(k))
        values.append(float(v))
    print(file.split('/')[-1], values[0], max(values))

    ss = file.split('curve_')[-1].split('.json')[0]

    plt.plot(keys, values, label=f'{ss}')
    # idx += 1
    
plt.xlabel('t')
plt.ylabel('PCK')
plt.title(f'PCK Metric vs. t - Category: dog')
plt.legend()
plt.savefig('temp0.png')


"""
temp1
dog_Per_Image_evaluation_curve_-1_-1_10_0.005.json 38.7370485995486 43.80157203907204
dog_Per_Image_evaluation_curve_-1_0_10_0.005.json 38.33303039553039 43.991118141118136
dog_Per_Image_evaluation_curve_0_-1_10_0.005.json 40.45110815110816 44.37929755429756
dog_Per_Image_evaluation_curve_0_0_10_0.005.json 39.54221935471935 44.57322492322493
temp2
dog_Per_Image_evaluation_curve_-1_-1_10_0.005.json 38.7370485995486 43.80157203907204
dog_Per_Image_evaluation_curve_-1_0_10_-0.005.json 38.17176897176898 43.968679468679476
dog_Per_Image_evaluation_curve_0_-1_10_-0.005.json 40.45110815110816 44.37929755429756
dog_Per_Image_evaluation_curve_0_0_10_-0.005.json 39.8945966070966 44.45803317053317
"""