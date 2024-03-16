from scipy import io
import torch
import random
import numpy as np
import os
from sklearn.decomposition import PCA
import torch.utils.data as dataf
from operator import truediv

from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, confusion_matrix

def selectdataset(name):
    if name == "Houston":
        print("Houston")
        return "./data/HSI.mat", "./data/LiDAR.mat", "./data/TRLabel.mat", "./data/TSLabel.mat"

def setseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def makeseed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark= False
    torch.backends.cudnn.deterministic = True


#3. Set parameters
def setfig(pathsize1, pathsize2, batchsize, epoch, lr, nc):
    return pathsize1, pathsize2, batchsize, epoch, lr, nc


def loaddataset0(path):
    data = io.loadmat(path)
    data1 = data['hsi']
    data2 = data['lidar']
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)
    trlabel = data['train']
    telabel = data['test']


    print(trlabel.shape)


    return data1, data2, trlabel, telabel
#4. Load dataset
def loaddataset(hpath, lpath, trpath, tepath):
    data1 = io.loadmat(hpath)
    data1 = data1['HSI']
    data2 = io.loadmat(lpath)
    data2 = data2['LiDAR']
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)
    trlabel = io.loadmat(trpath)
    trlabel = trlabel['TRLabel']
    telabel = io.loadmat(tepath)
    telabel = telabel['TSLabel']


    print(trlabel.shape)


    return data1, data2, trlabel, telabel


# 5.Data preprocessing 1- Standardization of hyperspectral and LiDAR data
def Datapreprocessing1(data1, data2):
    [m, n, l] = data1.shape
    for i in range(l):
        minimal = data1[:, :, i].min()
        maximal = data1[:, :, i].max()
        data1[:, :, i] = (data1[:, :, i] - minimal) / (maximal - minimal)
    minimal = data2.min()
    maximal = data2.max()
    data2 = (data2 - minimal) / (maximal - minimal)
    return data1, data2


# 6.Data preprocessing 2- PCA dimensionality reduction of hyperspectral data
def Datapreprocessing2(data1, nc):
    [m, n, l] = data1.shape
    PC = np.reshape(data1, (m * n, l))
    pca = PCA(n_components=nc, copy=True, whiten=False)
    PC = pca.fit_transform(PC)
    PC = np.reshape(PC, (m, n, nc))
    return PC


# 7.Data preprocessing 3- HSI and LiDAR data boundary interpolation
def Datapreprocessing3(PC, data2, patchsize1, patchsize2, nc):
    # HSI
    temp = PC[:, :, 0]
    pad_width1 = np.int(np.floor(patchsize1 / 2))
    temp2 = np.pad(temp, pad_width1, 'symmetric')
    [a, b] = temp2.shape
    x1 = np.empty((a, b, nc), dtype='float32')
    for i in range(nc):
        temp = PC[:, :, i]
        pad_width1 = np.int(np.floor(patchsize1 / 2))
        temp2 = np.pad(temp, pad_width1, 'symmetric')
        x1[:, :, i] = temp2
    # LiDAR
    pad_width2 = np.int(np.floor(patchsize2 / 2))
    temp2 = np.pad(data2, pad_width2, 'symmetric')  # 注意这里的temp2
    x2 = temp2
    return x1, x2, pad_width1, pad_width2


# 8. Data Preprocessing 4- Generate Dataset (Set as Universal)
def Datapreprocessing4(data, label, patchsize, pad_width, nc):
    [ind1, ind2] = np.where(label != 0)
    print('data:',data.shape)
    Num = len(ind1)
    print(Num)
    Patch = np.empty((Num, nc, patchsize, patchsize), dtype='float32')
    Label = np.empty(Num)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        if nc == 1:
            patch = data[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                    (ind4[i] - pad_width):(ind4[i] + pad_width + 1)]
        else:
            patch = data[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                    (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]

        patch = np.reshape(patch, (patchsize * patchsize, nc))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (nc, patchsize, patchsize))
        Patch[i, :, :, :] = patch
        patchlabel = label[ind1[i], ind2[i]]
        Label[i] = patchlabel
    return Patch, Label


# 9. Data Preprocessing 5- Building a Data Loader (Set as Universal)
def Datapreprocessing5(TrainPatch1, TrainLabel1, TrainPatch2, TrainLabel2, TestPatch1, TestLabel1, TestPatch2,
                       TestLabel2, batchsize):
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainLabel1 = torch.from_numpy(TrainLabel1) - 1
    TrainLabel1 = TrainLabel1.long()
    TestPatch1 = torch.from_numpy(TestPatch1)
    TestLabel1 = torch.from_numpy(TestLabel1) - 1
    TestLabel1 = TestLabel1.long()
    Classes = len(np.unique(TrainLabel1))
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel2 = torch.from_numpy(TrainLabel2) - 1
    TrainLabel2 = TrainLabel2.long()
    dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel2)
    train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=False)
    TestPatch2 = torch.from_numpy(TestPatch2)
    TestLabel2 = torch.from_numpy(TestLabel2) - 1
    TestLabel2 = TestLabel2.long()
    dataset = dataf.TensorDataset(TestPatch1, TestPatch2, TestLabel2)
    test_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=False)
    return train_loader, test_loader, Classes

# 10.Prepare labels and accuracy for the Houston dataset
def acc_reports(y_test, y_pred_test):
    target_names = ['1', '2', '3', '4', '5', '6', '7',
                    '8', '9', '10', '11',
                    '12', '13', '14', '15']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, round(oa * 100, 2), confusion, np.round(each_acc*100, 2), round(aa * 100, 2), round(kappa * 100,2)


# 11.Output accuracy and confusion matrix
def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# 12.Print results
def AngPrint(oa,aa,kappa):
    print('OA:', oa)
    print('AA:', aa)
    print('Kappa:', kappa)
    print("\n")

