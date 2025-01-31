import nibabel as nib
import numpy as np
import pdb
import sys
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import os
from utils import *
import torch 
import pandas as pd
import glob
import cv2
from natsort import natsorted


if len(sys.argv) != 4:
    print("Usage: python your_script.py <date> <ID> <CW_ID>")
    print("Please provide the required arguments: date, ID, and CW_ID.")
    sys.exit(1)

# Extract the command-line arguments
date = sys.argv[1]
ID = sys.argv[2]
CW_ID = sys.argv[3]

base_path='/ix1/tibrahim/shared/tibrahim_drj21/03-PMB'
mat = pd.read_table(f'{base_path}/{date}/angle_pos', header=None)
[x,y,z] = np.float16(mat.iloc[0][0].split(","))
rot_mat = get_rotation_matrix(x,y,z) #x y -z
best_pos = np.int16(mat.iloc[1][0].split(","))
trans_mat = get_translation_matrix(x=0, y=0, z=0)

cam_img = glob.glob(f'{base_path}/{date}/{ID}/rembg_cam/*.png')
cam_img = natsorted(cam_img)
ap_img = cv2.imread(glob.glob(f'{base_path}/{date}/{CW_ID}/*A.001*')[0])
ap_img = np.flip(ap_img, 1)
ap_img = cv2.cvtColor(ap_img, cv2.COLOR_BGR2RGB)
ap_img = np.array(ap_img, dtype='uint8')

T1 = nib.load(f'{base_path}/{date}/{ID}/mri/rmbgT1.nii.gz').get_fdata()
T1_img = nib.load(f'{base_path}/{date}/{ID}/mri/T1.nii.gz').get_fdata()
T2_img = nib.load(f'{base_path}/{date}/{ID}/mri/rT2.nii.gz').get_fdata()
GRE_img = nib.load(f'{base_path}/{date}/{ID}/mri/rGRE.nii.gz').get_fdata()

wmh_img = nib.load(f'{base_path}/{date}/{ID}/mri/T1_wmh.nii.gz').get_fdata()

img = [cv2.imread(path) for path in cam_img]
img_gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in img]

rotated = resample_image(T1, trans_mat, rot_mat)
rotated = np.einsum('ijk -> kij', rotated)
rotated = np.rot90(rotated, 2)

rotated_T2 = resample_image(T2_img, trans_mat, rot_mat)
rotated_T2 = np.einsum('ijk -> kij', rotated_T2)
rotated_T2 = np.rot90(rotated_T2, 2)

rotated_T1 = resample_image(T1_img, trans_mat, rot_mat)
rotated_T1 = np.einsum('ijk -> kij', rotated_T1)
rotated_T1 = np.rot90(rotated_T1, 2)

rotated_wmh = resample_image(wmh_img, trans_mat, rot_mat)
rotated_wmh = np.einsum('ijk -> kij', rotated_wmh)
rotated_wmh = np.rot90(rotated_wmh, 2)

rotated_gre = resample_image(GRE_img, trans_mat, rot_mat)
rotated_gre = np.einsum('ijk -> kij', rotated_gre)
rotated_gre = np.rot90(rotated_gre, 2)

sliced_brain = []
sliced_brain_T1 = []
sliced_brain_T2 = []
sliced_brain_GRE = []

best_score = []
best_img = []
best_img_T1 = []
best_img_T2 = []
best_wmh_T1 = []
best_img_GRE = []

for i in range(len(best_pos)):    
    best_img_T1.append(rotated_T1[:,:,best_pos[i]])
    best_img_T2.append(rotated_T2[:,:,best_pos[i]])
    best_wmh_T1.append(rotated_wmh[:,:,best_pos[i]])
    best_img_GRE.append(rotated_gre[:,:,best_pos[i]])

###############################################################################
img_rgb = [zeroCrop(im) for im in img]
img_rgb = [customResize(img, [400,228]) for img in img_rgb] # 400x228
ap_img = customResize(ap_img, np.int16([ap_img.shape[0]/5, ap_img.shape[1]/5]))

try:
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/match_GRE')
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/match_GRE_anno')
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/match_T1_anno')
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/match_T2_anno')
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/match_wmh')
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/resizedCam')
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/match_T1')
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/match_T2')
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/ap')
    
except:
    pass
Image.fromarray(ap_img).save(f'{base_path}/{date}/{ID}/rembg_cam/ap/ap.png')

for i in range(len(best_img_T1)):
    wmh_img = best_wmh_T1[i]*255
    T1_img = best_img_T1[i]/np.max(best_img_T1[i]) * 255    
    T2_img = best_img_T2[i]/np.max(best_img_T2[i]) * 255
    GRE_img = best_img_GRE[i]/np.max(best_img_GRE[i]) * 255
    resized_img = img_rgb[i]/np.max(img_rgb[i]) * 255
    
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/match_wmh/{i}.png', wmh_img)
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/match_T1/{i}.png', T1_img)
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/match_T2/{i}.png', T2_img)
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/match_GRE/{i}.png', GRE_img)
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/resizedCam/{i}.png', resized_img)

###############################################################################
img1_p = natsorted(glob.glob(f'{base_path}/{date}/{ID}/rembg_cam/resizedCam/*.png'))
img2_p = natsorted(glob.glob(f'{base_path}/{date}/{ID}/rembg_cam/match_T1/*.png'))
img3_p = natsorted(glob.glob(f'{base_path}/{date}/{ID}/rembg_cam/match_T2/*.png'))
img4_p = natsorted(glob.glob(f'{base_path}/{date}/{ID}/rembg_cam/match_GRE/*.png'))
img1 = [load_torch_image(imgp)[0]  for imgp in img1_p]
img2 = [load_torch_image(imgp)[0]  for imgp in img2_p]
img3 = [load_torch_image(imgp, False)[0]  for imgp in img3_p]
img4 = [load_torch_image(imgp, False)[0]  for imgp in img4_p]


img1_mask = [load_torch_image(imgp)[1]  for imgp in img1_p]
img2_mask = [load_torch_image(imgp)[1]  for imgp in img2_p]
gre_mask = [load_torch_image(imgp)[1]  for imgp in img4_p]


img1_np = [cv2.imread(imgp)  for imgp in img1_p]

matcher = KF.LoFTR(pretrained='outdoor')
warpped_cam = []
for i in tqdm(range(len(img1))):
    
    input_dict = {"image0": K.color.rgb_to_grayscale(img1[i]), # LofTR works on grayscale images only 
                "image1": K.color.rgb_to_grayscale(img2[i])}
    with torch.inference_mode():
        correspondences = matcher(input_dict)
    
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    try:
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.1, 0.999, 100000)
    except:
        pdb.set_trace()
    inliers = inliers > 0

    transform = cv2.estimateAffine2D(mkpts0, mkpts1)[0]
    _, _, cols, rows = img2[i].shape
    src = np.array(img2[i].detach().cpu())
    src = np.squeeze(src[0,0,:,:])

    target = np.array(img1_np[i])

    img_output = cv2.warpAffine(target, transform, (rows, cols))
    warpped_cam.append(img_output)

try:
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/resizedCam_adjusted_anno')
    os.mkdir(f'{base_path}/{date}/{ID}/rembg_cam/resizedCam_adjusted')
except:
    pass

for i in range(len(warpped_cam)):

    img2[i] = np.array(np.squeeze(img2[i]).detach().cpu())
    img3[i] = np.array(np.squeeze(img3[i]))
    img4[i] = np.array(np.squeeze(img4[i]))
    T1_img_anno = img2[i]/np.max(img2[i]) * 255
    T2_img_anno = img3[i]/np.max(img3[i]) * 255
    gre_img_anno = img4[i]/np.max(img4[i]) * 255
    T1_img_anno = np.transpose(T1_img_anno, (1,2,0))
    T2_img_anno = np.transpose(T2_img_anno, (1,2,0))    
    gre_img_anno = np.transpose(gre_img_anno, (1,2,0))   
    T2_img_anno = np.multiply(T2_img_anno[:,:,0] , (img2_mask[i]/np.max(img2_mask[i])))
    T2_img_anno = np.expand_dims(T2_img_anno, -1)
    T2_img_anno = np.repeat(T2_img_anno, 3, axis = 2)
    gre_img_anno = np.multiply(gre_img_anno[:,:,0] , (gre_mask[i]/np.max(gre_mask[i])))
    gre_img_anno = np.expand_dims(gre_img_anno, -1)
    gre_img_anno = np.repeat(gre_img_anno, 3, axis = 2)
    
    resized_img = warpped_cam[i]/np.max(warpped_cam[i]) * 255
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/resizedCam_adjusted/{i}.png', resized_img)
    resized_img_anno = resized_img
    # resized_img_anno[:,:,2] += best_wmh_T1[i] * 500
    resized_img_anno[:,:,1] += best_wmh_T1[i] * 500
    T1_img_anno[:,:,2] += best_wmh_T1[i] * 255
    T2_img_anno[:,:,2] += best_wmh_T1[i] * 255
    gre_img_anno[:,:,2] += best_wmh_T1[i] * 255
    
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/match_T1_anno/{i}.png', T1_img_anno)
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/match_T2_anno/{i}.png', T2_img_anno)
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/match_GRE_anno/{i}.png', gre_img_anno)
    cv2.imwrite( f'{base_path}/{date}/{ID}/rembg_cam/resizedCam_adjusted_anno/{i}.png', resized_img_anno)

#make ppt
os.system(f'python3 makePPT.py {ID} {date} {CW_ID}')
#reconstruct png_nii
os.system(f'python3 png2nii_cam.py {ID} {date} {best_pos[0]} {best_pos[-1]}')