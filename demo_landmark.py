import face_alignment
import glob
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
import random

import h5py

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

image_list = glob.glob("./data/demo/*.jpeg")
print(image_list)
num_image = len(image_list)

def prepare_dataset(name, size, _image_list):

    with h5py.File("./data/demo_{}.hdf5".format(name), 'w') as f:
        image_dset = f.create_dataset("img", shape=(size, 250, 250 , 3), dtype='uint8')
        landmark_dset = f.create_dataset("lmk_2D", shape=(size, 68, 2), dtype='uint8')
        for idx, image_path in tqdm(enumerate(_image_list), desc="Processing Landmark...", total=size):
            image = io.imread(image_path)
            image_dset[idx] = image
            lmk = fa.get_landmarks(image)
            if lmk is None:
                print ("Error in {}".format(image_path))
            else:
                landmark_dset[idx] = lmk[0][:,:2]

print ("Prepare demo set:")
prepare_dataset("demo", num_image, image_list[:num_image])

print ("Dataset prearation stopped for protecting the current data folder!")

