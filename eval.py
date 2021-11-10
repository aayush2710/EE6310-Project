import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import transforms, models
import os

labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
cmap = plt.cm.get_cmap('tab20c')
colors = (cmap(np.arange(cmap.N)) * 255).astype(np.int)[:, :3].tolist()
np.random.seed(2020)
np.random.shuffle(colors)
colors.insert(0, [0, 0, 0]) # background color must be black
colors = np.array(colors, dtype=np.uint8)

palette_map = np.empty((10, 0, 3), dtype=np.uint8)
legend = []

for i in range(21):
    legend.append(mpatches.Patch(color=np.array(colors[i]) / 255., label='%d: %s' % (i, labels[i])))
    c = np.full((10, 10, 3), colors[i], dtype=np.uint8)
    palette_map = np.concatenate([palette_map, c], axis=1)

# plt.figure(figsize=(20, 2))
# plt.legend(handles=legend)
# plt.imshow(palette_map)

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

def eval(img,gt):
    fg_h, fg_w, _ = img.shape
    segment_map, pred = segment(model, img)
    gt[gt>0] = 1
    pred[pred>0] = 1
    return mIOU(pred,gt)


dataset="SBI2015_dataset"
mIOUs = []
for i in os.listdir(dataset):
    imgs = os.listdir(dataset+"/" + i + "/input/")
    gts = os.listdir(dataset+"/" + i + "/groundtruth/" )
    for j in range(len(imgs)):
        img = np.array(Image.open(dataset+'/'+i +'/input/'+imgs[j]))
        gt = np.array(Image.open(dataset+'/'+i +'/groundtruth/'+gts[j]))
        mIOUs.append(eval(img,gt))

print(np.mean(mIOUs))

# fig, axes = plt.subplots(1, 2, figsize=(20, 10))
# axes[0].imshow(img)
# axes[1].imshow(segment_map)

