#!/usr/bin/env python

import scipy.misc, shutil, os, nibabel
from PIL import Image as im
import numpy as np
import sys

input = sys.argv[1]
output = sys.argv[2]
tag = sys.argv[3]

print('Input file is ', input)
print('Output folder is ', output)
print('Image tag is ', tag)


if tag == 'm':
    image_array = nibabel.load(input).get_fdata()
elif tag == 'i':
    image_array = nibabel.load(input).get_fdata(dtype=np.float64)
    image_array = image_array/np.max(image_array) * 255
# set destination folder
if not os.path.exists(output):
    os.makedirs(output)
    print("Created ouput directory: " + output)

print('Reading NIfTI file...')
total_slices = image_array.shape[1] #coronal slice
slice_counter = 0

# iterate through slices
for current_slice in range(0, total_slices):
    # alternate slices
    if (slice_counter % 1) == 0:
        # rotate or no rotate
        data = image_array[: , current_slice, :]
        data = np.rot90(data, 1, (0,1))
        #alternate slices and save as png
        if (slice_counter % 1) == 0:
            print('Saving image...')
            image_name = input[:-4] + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
            image_slice = im.fromarray(data).convert("L")
            image_slice.save(image_name)
            print('Saved.')

            #move images to folder
            print('Moving image...')
            src = image_name
            shutil.move(src, output)
            slice_counter += 1
            print('Moved.')

print('Finished converting images')