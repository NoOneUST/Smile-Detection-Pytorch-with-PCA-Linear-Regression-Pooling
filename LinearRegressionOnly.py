import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

device = 'cuda:0'
datasetFolder = '/home/cqiuac/Downloads/img_align_celeba'
mySmileDataPath = './mysmiledata.t7'

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()
dtype = torch.float
trainsetPercent = 3 / 4


def image_loader(image_name):
    image = Image.open(image_name).convert('L')
    image = loader(image).unsqueeze(0)
    return image.to(device, dtype)


if os.path.exists(mySmileDataPath):
    data = torch.load(mySmileDataPath)
    bTrain = data['bTrain']
    XTrain = data['XTrain']
    bTest = data['bTest']
    XTest = data['XTest']
    print('Loading successful')
else:
    f = open("list_attr_celeba.txt")

    Number_Of_Image = len(f.readlines())
    Number_Of_Used_Image = int(Number_Of_Image / 10)
    f.seek(0)
    line = f.readline()
    line = f.readline()
    Smile_Indicator_Index = line.split().index('Smiling')
    line = f.readline()
    array = line.split()
    img = Image.open(os.path.join(datasetFolder, array[0]))
    total_Pixel_Number_Per_Img = np.prod(img.size)
    b = torch.zeros(1, Number_Of_Used_Image, dtype=dtype, device=device)
    X = torch.zeros(
        total_Pixel_Number_Per_Img,
        Number_Of_Used_Image,
        dtype=dtype,
        device=device)
    imgNameList = []
    lineInd = 0
    Number_Of_Smile_Image_Choosen = 0
    Number_Of_None_Smile_Image_Choosen = 0
    lastlineInd = -1
    while line and Number_Of_Smile_Image_Choosen + \
            Number_Of_None_Smile_Image_Choosen < Number_Of_Used_Image:
        array = line.split()
        if int(array[Smile_Indicator_Index]
               ) == 1 and Number_Of_Smile_Image_Choosen < Number_Of_Used_Image / 2:
            Number_Of_Smile_Image_Choosen += 1
            img = image_loader(os.path.join(datasetFolder, array[0]))
            imgNameList.append(array[0])
            b[0, lineInd] = int(array[Smile_Indicator_Index])
            X[:, lineInd] = img.flatten()
            lineInd += 1
        if int(array[Smile_Indicator_Index]) == - \
                1 and Number_Of_None_Smile_Image_Choosen < Number_Of_Used_Image / 2:
            Number_Of_None_Smile_Image_Choosen += 1
            img = image_loader(os.path.join(datasetFolder, array[0]))
            imgNameList.append(array[0])
            b[0, lineInd] = int(array[Smile_Indicator_Index])
            X[:, lineInd] = img.flatten()
            lineInd += 1
        line = f.readline()
        if lineInd % 1000 == 0 and lastlineInd != lineInd:
            print('processed:', lineInd, '/', Number_Of_Used_Image)
            lastlineInd = lineInd
    bTrain = b[0, :int(trainsetPercent * Number_Of_Used_Image)].t()
    XTrain = X[:, :int(trainsetPercent * Number_Of_Used_Image)].t()
    bTest = b[0, int(trainsetPercent * Number_Of_Used_Image):].t()
    XTest = X[:, int(trainsetPercent * Number_Of_Used_Image):].t()
    print('===> Saving models...')
    torch.save({
        'bTrain': bTrain,
        'XTrain': XTrain,
        'bTest': bTest,
        'XTest': XTest
    }, './mysmiledata.t7')


A, LU = torch.lstsq(bTrain.unsqueeze(1).cpu(), XTrain.cpu())
A, LU = [x.cuda() for x in [A, LU]]

bPredicted = XTest@A
loss = torch.mean(torch.abs(bPredicted - bTest))
lossDiscrete = torch.mean(((bPredicted / bTest) > 0).float())
print('Loss on backtesting:', loss.item())
print('Accuracy on backtesting:', lossDiscrete.item() * 100, '%')
bPredictedOri = XTrain@A
loss = torch.mean(torch.abs(bPredictedOri - bTrain))
print('Loss on Trainingset:', loss.item())
