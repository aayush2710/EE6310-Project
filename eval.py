import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import transforms, models
import os
from tqdm import tqdm


model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
cmap = plt.cm.get_cmap('tab20c')
colors = (cmap(np.arange(cmap.N)) * 255).astype(np.int)[:, :3].tolist()
np.random.seed(2020)
np.random.shuffle(colors)
colors.insert(0, [0, 0, 0]) # background color must be black
colors = np.array(colors, dtype=np.uint8)

if torch.cuda.is_available():
        print("Working on CUDA")
def segment(net, img):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    output = model(input_batch)['out'][0] # (21, height, width)

    output_predictions = output.argmax(0).byte().cpu().numpy() # (height, width) 

    r = Image.fromarray(output_predictions).resize((img.shape[1], img.shape[0]))
    r.putpalette(colors)

    return r, output_predictions


def pixel_accuracy(y_pred,y_true):
    return np.mean(y_pred==y_true)*100

def mIOU(y_pred,y_true):
    intersection = np.sum(y_pred*y_true)
    union = y_pred+y_true
    union = np.sum(union>0)
    return np.mean(intersection/union)*100

def dice(y_pred, y_true. smooth=1):
    intersection = np.sum(y_pred*y_true)
    # union = y_pred+y_true
    # union = np.sum(union > 0)
    total_pixels = len(y_pred.flatten())+len(y_true.flatten())
    return intersection/total_pixels

def eval(img,gt):
    segment_map, pred = segment(model, img)
    gt[gt>0] = 1
    pred[pred>0] = 1
    return mIOU(pred,gt), pixel_accuracy(pred,gt), dice(pred, gt)


dataset="PascalVOC2012"
mIOUs = []
categorical_miou = {}
for i in os.listdir(dataset):
    if i[0]==".":
        continue
    imgs = os.listdir(dataset+"/" + i + "/input/")
    gts = os.listdir(dataset+"/" + i + "/groundtruth/" )
    imgs.sort()
    gts.sort()
    for j in tqdm(range(len(imgs))):
        img = np.array(Image.open(dataset+'/'+i +'/input/'+imgs[j]))
        gt = np.array(Image.open(dataset+'/'+i +'/groundtruth/'+gts[j]))
        mIOUs.append(eval(img,gt))
    mIOUs = np.array(mIOUs)
    categorical_miou[i]=np.nanmean(mIOUs, axis=0)
    
print(categorical_miou)


