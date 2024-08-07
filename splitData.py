import os
import shutil
import random
import glob



os.chdir('/home/zyvasheikh/Desktop/work/images HITL/spheroid HITL 4C')
if os.path.isdir('train/0-20%') is False:
    os.makedirs('train/0-20%')
    os.makedirs('train/20-40%')
    os.makedirs('train/40-70%')
    os.makedirs('train/70-100%')
    os.makedirs('valid/0-20%')
    os.makedirs('valid/20-40%')
    os.makedirs('valid/40-70%')
    os.makedirs('valid/70-100%')
    os.makedirs('test/0-20%')
    os.makedirs('test/20-40%')
    os.makedirs('test/40-70%')
    os.makedirs('test/70-100%')
    
    for c in random.sample(glob.glob('0-20%*'), 107):
        shutil.move(c, 'train/0-20%')
    for c in random.sample(glob.glob('20-40%*'), 107):
        shutil.move(c, 'train/20-40%')
    for c in random.sample(glob.glob('40-70%*'), 107):
        shutil.move(c, 'train/40-70%')
    for c in random.sample(glob.glob('70-100%*'), 107):
        shutil.move(c, 'train/70-100%')
    for c in random.sample(glob.glob('0-20%*'), 13):
        shutil.move(c, 'valid/0-20%')
    for c in random.sample(glob.glob('20-40%*'), 13):
        shutil.move(c, 'valid/20-40%')
    for c in random.sample(glob.glob('40-70%*'), 13):
        shutil.move(c, 'valid/40-70%')
    for c in random.sample(glob.glob('70-100%*'), 13):
        shutil.move(c, 'valid/70-100%')
    for c in random.sample(glob.glob('0-20%*'), 13):
        shutil.move(c, 'test/0-20%')
    for c in random.sample(glob.glob('20-40%*'), 13):
        shutil.move(c, 'test/20-40%')
    for c in random.sample(glob.glob('40-70%*'), 13):
        shutil.move(c, 'test/40-70%')
    for c in random.sample(glob.glob('70-100%*'), 13):
        shutil.move(c, 'test/70-100%')