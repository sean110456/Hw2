# model
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
# from reference
import transforms as T
from engine import train_one_epoch, evaluate
import utils
# input
import os
from skimage import io
import pandas as pd
# output
import json
import cv2


class TrainDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.annotations = pd.read_hdf(h5_file, 'table')
        self.transform = transform
        print(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image = self.annotations.iloc[index]['img']

        # get the numbers in the picture
        strs = self.annotations.iloc[index]['labels'].split('_')
        strs = ['0.0' if s == '10.0' else s for s in strs]
        labels = torch.tensor(list(map(int, map(float, strs))))

        # Divide box into boxes
        box = []
        for i in range(2, 6):
            box.append(int(self.annotations.iloc[index, i]))
        interval_x = float(box[3]-box[1])/len(labels)
        boxes = [[box[1], box[0], int(box[1]+interval_x), box[2]]]
        for i in range(1, len(labels)):
            boxes.append(
                [boxes[i-1][2], box[0], int(boxes[i-1][2]+interval_x), box[2]])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        img_id = torch.tensor(int(self.annotations.iloc[index, 0][0:-4]))

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = img_id
        target["area"] = area

        # Do transforms
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target


class TestDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform

        # Read all image from test_data directory
        self.images = []
        for filename in os.listdir(dir):
            tmp = io.imread(os.path.join(dir, filename))
            if tmp is not None:
                self.images.append(tmp)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        if self.transform:
            image = self.transform(image)

        return image


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# 10 classes 0~9
num_classes = 10
# Hyperparameters
learning_rate = 0.005
batch_size = 2
num_epochs = 10

# Dataset & Loader
H5_PATH = 'train/train_data_processed.h5'
train_data = TrainDataset(H5_PATH, get_transform(train=True))
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)
trns = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
test_data = TestDataset('test', trns)
test_loader = DataLoader(
    dataset=test_data, batch_size=1, shuffle=False)

# GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Faster RCNN model
model = models.detection.fasterrcnn_resnet50_fpn(
    pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
model.to(device)

# optimizer,scheduler
# params are the parameters to fine tune, put them in the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate,
                            momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Train Network
for epoch in range(num_epochs):
    print('epoch:', epoch)
    train_one_epoch(model, optimizer, train_loader,
                    device, epoch, print_freq=1000)
    scheduler.step()
print('---training done---')

torch.save(model, 'FasterRCNN.pth')
#model = torch.load('FasterRCNN.pth')
model.to(device)
model.eval()

result = []
idx = 1
with torch.no_grad():
    for img in test_loader:
        img = img.to(device)
        predictions = model(img)

        """ # this will delete those scores which are smaller then 'thres'
        tail_idx = 0
        thres = 0.7
        for i in range(len(predictions[0]['scores'])):
            if predictions[0]['scores'][i] < thres:
                tail_idx = i
                break
        """
        prdt = [{}]
        prdt[0]['bbox'] = predictions[0]['boxes'].cpu().numpy()  # [0:tail_idx]
        prdt[0]['score'] = predictions[0]['scores'].cpu().numpy()  # [0:tail_idx]
        t = predictions[0]['labels'].cpu().numpy()  # [0:tail_idx]
        prdt[0]['label'] = [10 if i == 0 else i for i in t]

        """ # show the result 
        filename = str(idx) + '.png'
        t_img = io.imread(os.path.join('test', filename))
        idx += 1
        for i in range(len(prdt[0]['label'])):
            cv2.rectangle(t_img, (int(prdt[0]['bbox'][i][0]), int(prdt[0]['bbox'][i][1])), (int(
                prdt[0]['bbox'][i][2]), int(prdt[0]['bbox'][i][3])), (255, 255, 255), thickness=2)
        cv2.imshow(str(prdt[0]['label']), t_img)
        cv2.waitKey()
        """

        # swap x, y for the output format
        for ar in prdt[0]['bbox']:
            ar[0], ar[1] = ar[1], ar[0]
            ar[2], ar[3] = ar[3], ar[2]

        result.extend(prdt)
print('prediction done')

# type convertion for json ouput
for r in result:
    r['bbox'] = r['bbox'].tolist()
    r['score'] = r['score'].tolist()
    for i in range(len(r['label'])):
        r['label'][i] = r['label'][i].item()  # convert numpy.int64 to int

with open('309551082.json', 'w') as f:
    json.dump(result, f)
print('output done')
