import ants
import numpy as np
import nibabel as nib
import sys
import pdb
import glob

base_path=sys.argv[1]
date=sys.argv[2]
ID=sys.argv[3]
type=sys.argv[4]

img1 = ants.image_read(f'{base_path}/{date}/{ID}/mri/T1.nii.gz')
img2 = ants.image_read(f'{base_path}/{date}/{ID}/mri/T2.nii.gz')

if len(glob.glob(f'{base_path}/{date}/{ID}/mri/rT2.nii.gz')) > 0:
    pass
else:
    mytx = ants.registration(fixed=img1 , moving=img2, type_of_transform=type)
    ants.image_write(mytx['warpedmovout'], f'{base_path}/{date}/{ID}/mri/rT2.nii.gz')

if len(glob.glob(f'{base_path}/{date}/{ID}/mri/rGRE.nii.gz')) > 0:
    pass
else:
    try:
        img2 = ants.image_read(f'{base_path}/{date}/{ID}/mri/GRE.nii.gz')
        mytx = ants.registration(fixed=img1 , moving=img2, type_of_transform=type)
        ants.image_write(mytx['warpedmovout'], f'{base_path}/{date}/{ID}/mri/rGRE.nii.gz')
    except:
        pass
