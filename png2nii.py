import nibabel as nib
import numpy as np
import glob
import sys 
import cv2 as cv

base_path=sys.argv[1]
date=sys.argv[2]
ID=sys.argv[3]

print(base_path)
print(ID)
print(date)

png_imgs=glob.glob(f'{base_path}/{date}/{ID}/rmbg_png/*.png')
png_imgs.sort()
nii_img=glob.glob(f'{base_path}/{date}/{ID}/mri/T1.nii.gz')
outpath1=f'{base_path}/{date}/{ID}/mri/rmbgT1.nii.gz'
outpath2=f'{base_path}/{date}/{ID}/mri/T1_mask.nii.gz'

imgs = [cv.cvtColor(cv.imread(p), cv.COLOR_BGR2GRAY) for p in png_imgs]
nii = nib.load(nii_img[0])
nii_img = nii.get_fdata()

out = np.array(imgs)
out = np.flip(out, axis=1)
out = np.transpose(out, [2,0,1])

out_nii = nib.Nifti1Image(out, affine=nii.affine)
nib.save(out_nii, outpath1)

##############################################################
png_imgs=glob.glob(f'{base_path}/{date}/{ID}/rmbg_mask/*.png')
png_imgs.sort()
imgs = [cv.cvtColor(cv.imread(p), cv.COLOR_BGR2GRAY) for p in png_imgs]
masks = []
for img in imgs:
    idx = np.argwhere(img != 0)
    img_temp = np.zeros_like(img)
    for ind in range(len(idx)):
        x,y = idx[ind]
        img_temp[x,y] = 1
    masks.append(img_temp)
    
out = np.array(masks)
out = np.flip(out, axis=1)
out = np.transpose(out, [2,0,1])
out_nii = nib.Nifti1Image(out, affine=nii.affine)
nib.save(out_nii, outpath2)

