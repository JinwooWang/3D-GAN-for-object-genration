import numpy as np
import os
import scipy.io

new_train_dir = './Voxel_Files/'
train_dir = './volumetric_data/chair/30/train/'

def data_preprocess():
    if not os.path.exists(new_train_dir):
        os.makedirs(new_train_dir)

    train_file_names = [f for f in os.listdir(train_dir) if f.endswith('_1.mat')]
    for filename in train_file_names:
        voxel_matrix = scipy.io.loadmat(train_dir+filename)['instance']
        voxel_matrix=np.pad(voxel_matrix,(1,1),'constant',constant_values=(0,0))
        voxel_matrix.dump(new_train_dir+filename[:-4])

    test_dir = './volumetric_data/chair/30/test/'
    test_file_names = [filename for filename in os.listdir(test_dir) if filename.endswith('_1.mat')]
    for filename in test_file_names:
        voxel_matrix = scipy.io.loadmat(test_dir+filename)['instance']
        voxel_matrix=np.pad(voxel_matrix,(1,1),'constant',constant_values=(0,0))
        voxel_matrix.dump(new_train_dir+filename[:-4])

if __name__ == "__main__":
    data_preprocess()
