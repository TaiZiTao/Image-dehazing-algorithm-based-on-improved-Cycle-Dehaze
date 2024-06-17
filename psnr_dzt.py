from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import math
import yaml
import json
import os
from skimage.metrics import structural_similarity as ssim
# file = '/home/sh/lzy/c2/myconfig/test/sots_test.yaml'
# opt = yaml.safe_load(open(file, 'r', encoding='utf-8').read())

# img1 = np.array(Image.open('original.jpg'))
# img2 = np.array(Image.open('compress.jpg'))


# def psnr(img1, img2):
#     mse = np.mean((img1 -img2 )**2)
#     if mse == 0:
#         return 100
#     else:
#         return 20*np.log10(255/np.sqrt(mse))
def _psnr(img1, img2):
    mse =np.mean((img1/255. - img2/255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20* math.log10(PIXEL_MAX / math.sqrt(mse))


# label_dir = 'data/Densehaze_datasets/test/clear'
# out_dir = 'pred_FFA_DENSE'
# label_dir = opt['label_dir']
# out_dir = opt['out_dir']
def psnr(out_dir, label_dir):
    # s1 = set(os.path.splitext(i)[0] for i in os.listdir(out_dir) if os.path.splitext(i)[1]=='.json' )

    save_dir, save_file = os.path.dirname(out_dir), os.path.basename(out_dir)
    save_file += '.json'
    save_path = os.path.join(save_dir, save_file)
    imgs = list(Path(out_dir).iterdir())
    format = Path(label_dir).iterdir().__next__().suffix
    psnrs = []
    ssims = []
    i=1
    for img in imgs:
        print(i)
        i=i+1
        label = img.name.split('/')[-1].split('_')[0] #+ format
        label = label_dir + '/' + label[:-10]+'targets.png'
        print(img)
        print(label)
        img = cv2.imread(str(img), 1)
        label = cv2.imread(str(label), 1)
        psnrs.append(_psnr(img, label))
        ssims.append(ssim(img,label,multichannel=True))
    dic = {}
    dic['eval_psnr'] = np.mean(np.array(psnrs))
    dic['psnrs'] = psnrs
    dic['eval_ssim'] = np.mean(np.array(ssims))
    dic['ssims'] = ssims
    print(save_path)
    json.dump(dic, open(save_path, 'w'))





if __name__ == "__main__":
    out_dir='myresult/outdoor_outdoor_latest'
    label_dir='./datasets/outdoor/testB'
    psnr(out_dir=out_dir,label_dir=label_dir)

