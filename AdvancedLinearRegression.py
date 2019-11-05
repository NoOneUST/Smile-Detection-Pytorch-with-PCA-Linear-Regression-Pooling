import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

device = 'cuda:0'
datasetFolder = '/home/cqiuac/Downloads/img_align_celeba'
mySmileDataPath = './mysmiledataPooling.t7'

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()
dtype = torch.float
trainsetPercent = 3 / 4
poolingSize = [100, 100]


def image_loader(image_name):
    image = Image.open(image_name).convert('L')
    image = loader(image).unsqueeze(0)
    return image.to(device, dtype)


def pca(X, k=10000):
    X_mean = torch.mean(X, 1).unsqueeze(1)
    X = X - X_mean
    U, S, V = torch.svd(X)
    S[k:] = 0
    return U@torch.diag(S)@V.t()


print('===> Loading Data...')

if os.path.exists(mySmileDataPath):
    data = torch.load(mySmileDataPath)
    A = data['A']
    bTest = data['bTest']
    XTest = data['XTest']
    lossTrain = data['lossTrain']
    imgNameTestList = data['imgNameTestList']
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
    total_Pixel_Number_Per_Img = np.prod(poolingSize)
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
    poolingOp = torch.nn.AdaptiveMaxPool2d((100, 100))

    while line and Number_Of_Smile_Image_Choosen + \
            Number_Of_None_Smile_Image_Choosen < Number_Of_Used_Image:
        array = line.split()
        if int(array[Smile_Indicator_Index]
               ) == 1 and Number_Of_Smile_Image_Choosen < Number_Of_Used_Image / 2:
            Number_Of_Smile_Image_Choosen += 1
            img = image_loader(os.path.join(datasetFolder, array[0]))
            img = poolingOp(img)
            imgNameList.append(array[0])
            b[0, lineInd] = int(array[Smile_Indicator_Index])
            X[:, lineInd] = img.flatten()
            lineInd += 1
        if int(array[Smile_Indicator_Index]) == - \
                1 and Number_Of_None_Smile_Image_Choosen < Number_Of_Used_Image / 2:
            Number_Of_None_Smile_Image_Choosen += 1
            img = image_loader(os.path.join(datasetFolder, array[0]))
            img = poolingOp(img)
            imgNameList.append(array[0])
            b[0, lineInd] = int(array[Smile_Indicator_Index])
            X[:, lineInd] = img.flatten()
            lineInd += 1
        line = f.readline()
        if lineInd % 1000 == 0 and lastlineInd != lineInd:
            print('processed:', lineInd, '/', Number_Of_Used_Image)
            lastlineInd = lineInd

    print('===> Doing PCA...')
    X = pca(X)
    bTrain = b[0, :int(trainsetPercent * Number_Of_Used_Image)].t()
    XTrain = X[:, :int(trainsetPercent * Number_Of_Used_Image)].t()
    bTest = b[0, int(trainsetPercent * Number_Of_Used_Image):].t()
    XTest = X[:, int(trainsetPercent * Number_Of_Used_Image):].t()
    A, LU = torch.lstsq(bTrain.cpu(), XTrain.cpu())
    A, LU = [x.cuda() for x in [A, LU]]
    A = A[:np.prod(poolingSize), 0]
    bPredictedOri = XTrain @ A
    lossTrain = torch.mean(torch.abs(bPredictedOri - bTrain))
    imgNameTestList = imgNameList[int(trainsetPercent * Number_Of_Used_Image):]

    print('===> Saving models...')
    torch.save({
        'A': A,
        'bTest': bTest,
        'XTest': XTest,
        'lossTrain': lossTrain,
        'imgNameTestList': imgNameTestList,
    }, './mysmiledataPooling.t7')


bPredicted = XTest@A
loss = torch.mean(torch.abs(bPredicted - bTest))
lossDiscrete = torch.mean(((bPredicted / bTest) > 0).float())
choosenIndexList = (torch.rand(5) * XTest.size()[0]).int()
print()

for i in range(choosenIndexList.size()[0]):
    choosenIndex = choosenIndexList[i]
    img = Image.open(os.path.join(datasetFolder,
                                  imgNameTestList[choosenIndex.item()]))
    plt.figure()
    plt.imshow(img)
    plt.show()
    if bPredicted[choosenIndex.item()] >= 0:
        print('Yes, I guess he is smiling.')
    else:
        print('No, I guess he is not smiling.')
    if bTest[choosenIndex.item()] / bPredicted[choosenIndex.item()] >= 0:
        print('You guess correctly.\n')
    else:
        print('You did not guess correctly.\n')

print('Accuracy on backtesting:', lossDiscrete.item() * 100, '%')
print('Loss on backtesting:', loss.item())
print('Loss on Trainingset:', lossTrain.item())
