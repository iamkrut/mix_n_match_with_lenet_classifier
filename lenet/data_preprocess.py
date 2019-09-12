
import os
from os.path import isdir, exists, abspath, join
import numpy as np
from PIL import Image



def make_dirs(in_dir, not_exist=True):
    out_dir = in_dir
    try:
        os.makedirs(out_dir)
        print("Directory " , out_dir ,  " Created ")
    except FileExistsError:
        print("Directory " , out_dir ,  " already exists")
        not_exist = False

    return (join(out_dir), not_exist)
    

def conc_images(from_dirName, to_dirName, not_exist):
    if not_exist:
        idx = 0
        for filename in os.listdir(from_dirName):
            
            
            view = int(filename.split(".")[0]) 
            idx = view
            view = view % 3

            if view == 1:

                filename_1 = join(from_dirName,filename)
                filename_2 = join(from_dirName, '{}.bmp'.format(str(idx+1)))
                filename_3 = join(from_dirName, '{}.bmp'.format(str(idx+2)))
                try :
                    image_1 = Image.open(filename_1)
                except:
                    print("no such file : ",filename_1)
                    continue
                try:
                    image_2 = Image.open(filename_2)
                except:
                    print("no such file : ",filename_2)
                    continue
                try:
                    image_3 = Image.open(filename_3)
                except:
                    print("no such file : ",filename_3)
                    continue
                image_1 = np.array(image_1)
                image_2 = np.array(image_2)
                image_3 = np.array(image_3)
                image = np.concatenate((image_1,image_2,image_3), axis=1)
                image = Image.fromarray(image)
                image.save(join(to_dirName,filename))

    else:
        print("concatination is already done !!")


def preproc (root_dir='chairs-data'):

    root_dir = abspath(root_dir)
    dirName1 = join(root_dir,'positive')
    dirName2 = join(root_dir,'negative')

    dirName3, not_exist_3 = make_dirs(join(root_dir,'pos'))
    dirName4, not_exist_4 = make_dirs(join(root_dir,'neg'))

    conc_images(from_dirName=dirName1, to_dirName=dirName3, not_exist=not_exist_3)
    conc_images(from_dirName=dirName2, to_dirName=dirName4, not_exist=not_exist_4)


    print("Concatination is done!")

def preproc_2 (root_dir='chairs-data'):
    root_dir = abspath(root_dir) 
    dirName1 = join(root_dir,'train/positive')
    dirName2 = join(root_dir,'train/negative')
    dirName3 = join(root_dir,'test/positive')
    dirName4 = join(root_dir,'test/negative')

    ctr = 0
    try:
        os.makedirs(dirName1)    
        print("Directory " , dirName1 ,  " Created ")
    except FileExistsError:
        ctr += 1
        print("Directory " , dirName1 ,  " already exists")  

    try:
        os.makedirs(dirName2)    
        print("Directory " , dirName2 ,  " Created ")
    except FileExistsError:
        ctr += 1
        print("Directory " , dirName2 ,  " already exists")  

    try:
        os.makedirs(dirName3)    
        print("Directory " , dirName3 ,  " Created ")
    except FileExistsError:
        ctr += 1
        print("Directory " , dirName3 ,  " already exists")  

    try:
        os.makedirs(dirName4)    
        print("Directory " , dirName4 ,  " Created ")
    except FileExistsError:
        ctr += 1
        print("Directory " , dirName4 ,  " already exists")  

    if (ctr == 4):
        print("Train/Test devide is already done!")
        return


    test_percent = 0.20

    pos_dir = join(root_dir, 'pos')
    neg_dir = join(root_dir, 'neg')

    pos_files = os.listdir(pos_dir) 
    neg_files = os.listdir(neg_dir)

    pos_data_length = len(pos_files)
    neg_data_length = len(neg_files)

    train_pos_data_length = np.int_(pos_data_length - np.floor(pos_data_length * test_percent))
    train_neg_data_length = np.int_(neg_data_length - np.floor(neg_data_length * test_percent))

    if (ctr == 0) :
        for i in range(train_pos_data_length):
            img_name = pos_files[i]
            os.rename(join(pos_dir, img_name), join(dirName1, img_name))

    if (ctr == 0 or ctr == 1):
        for i in range(train_neg_data_length):
            img_name = neg_files[i]
            os.rename(join(neg_dir, img_name), join(dirName2, img_name))

    if (ctr == 0 or ctr == 1 or ctr == 2):
        for i in range(train_pos_data_length, pos_data_length):
            img_name = pos_files[i]
            os.rename(join(pos_dir, img_name), join(dirName3, img_name))

    if (ctr == 0 or ctr == 1 or ctr == 2 or ctr == 3):
        for i in range(train_neg_data_length, neg_data_length):
            img_name = neg_files[i]
            os.rename(join(neg_dir, img_name), join(dirName4, img_name))

    print("Train/Test devide done!")