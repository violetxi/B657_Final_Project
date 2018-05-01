# random sample N images to check detection accuracy

import sys
import os
import numpy as np
import random
import shutil

# Produce a randomly selected sample of size N (# of images sampled from NM and RS 
# is porportional to the number of NM vs. number of RS images)
def get_random_samples(img_dest1, img_dest2, sample_dest):
    imgs1 = os.listdir(img_dest1)
    imgs2 = os.listdir(img_dest2)
    imgs1_portion = float(len(imgs1)) / ( len(imgs1) + len(imgs2) )
    imgs2_portion = float(len(imgs2)) / ( len(imgs1) + len(imgs2) )
    #print imgs1_portion, imgs2_portion
    random.shuffle(imgs1)
    random.shuffle(imgs2)
    #samples = [] +imgs1[ 0: int(round(N*imgs1_portion)) ] + imgs2[ 0 : int(round(N*imgs2_portion)) ]
    samples1 = imgs1[ 0: int(round(N*imgs1_portion)) ]
    samples2 = imgs2[ 0 : int(round(N*imgs2_portion)) ]
    print len(samples1) + len(samples2)
    
    for s in samples1:
        shutil.move(img_dest1+s, sample_dest+s)
    for s in samples2:
        shutil.move(img_dest2+s, sample_dest+s)

# Number of samples
N = 100
# Paths to detection output images (NM and RS) and path to destination to put sampled images
(img_dest1, img_dest2, sample_dest) = sys.argv[1:]
get_random_samples(img_dest1, img_dest2, sample_dest)

