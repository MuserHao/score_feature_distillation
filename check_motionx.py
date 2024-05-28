import numpy as np
import os
import collections

dir1 = '/home/shilei/projects/Motion-X/datasets/motion_data/vector_263/EgoBody/recording_20210918_S05_S06_01/body_idx_0'
dir2 = '/home/shilei/projects/MotionxV1-IDEA/motionx/vector_263/EgoBody/recording_20210918_S05_S06_01/body_idx_0'

# subdirs1 = os.listdir(dir1)
# subdirs2 = os.listdir(dir2)
# inters =  set(subdirs1).intersection(set(subdirs2))

files = os.listdir(dir1)
for file in files:
    f1 = os.path.join(dir1, file)
    f2 = os.path.join(dir2, file)
    motion1 = np.load(f1)
    motion2 = np.load(f2)
    print(motion1.shape[0], motion2.shape[0])
        
    print(np.sum(motion1-motion2))
    # break