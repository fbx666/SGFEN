from scipy import io
import torch
import random
import numpy as np
import os
from sklearn.decomposition import PCA
import torch.utils.data as dataf
from operator import truediv
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, confusion_matrix


'''
1. Dataset selection
'''


def selectdataset(name):
    if name == "HoustonN":
        print("New-Houston")
        return "./data/HSI.mat", "./data/LiDAR.mat", "./data/TRLabel.mat", "./data/TSLabel.mat"
    elif name == "Trento":
        print('Trento')
        return "data/HSI.mat", "data/LiDAR.mat", "data/TRLabel.mat", "data/TSLabel.mat"


# 2. Set random seeds
def setseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def makeseed(seed):  # Fix all random seeds so that every run yields the same result
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    # Enabling cudnn.benchmark makes cuDNN spend a little extra time before
    # training to search for the fastest convolution algorithm for every
    # convolution layer, which speeds up networks with a fixed architecture
    # (it is not suitable for NAS).
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.deterministic is a boolean flag that controls
    # cuDNN's deterministic mode. When set to True, cuDNN uses deterministic
    # algorithms so that every run gives the same result; when set to False,
    # non-deterministic algorithms are used, which run faster.


# 3. Set parameters
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


# 4. Load dataset
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


# 5. Data preprocessing 1 - Standardization of HSI and LiDAR
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


# 6. Data preprocessing 2 - PCA dimensionality reduction of the HSI
def Datapreprocessing2(data1, nc):
    [m, n, l] = data1.shape
    PC = np.reshape(data1, (m * n, l))
    pca = PCA(n_components=nc, copy=True, whiten=False)
    PC = pca.fit_transform(PC)
    PC = np.reshape(PC, (m, n, nc))
    return PC


# 7. Data preprocessing 3 - Boundary padding for HSI and LiDAR
def Datapreprocessing3(PC, data2, patchsize1, patchsize2, nc):
    # HSI: symmetric padding
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
    # LiDAR: symmetric padding
    pad_width2 = np.int(np.floor(patchsize2 / 2))
    temp2 = np.pad(data2, pad_width2, 'symmetric')
    x2 = temp2
    return x1, x2, pad_width1, pad_width2


# 8. Data preprocessing 4 - Generate patches
def Datapreprocessing4(data, label, patchsize, pad_width, nc):
    [ind1, ind2] = np.where(label != 0)
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


# 9. Data preprocessing 5 - Build data loaders
def Datapreprocessing5(TrainPatch1, TrainLabel1, TrainPatch2, TrainLabel2, TestPatch1, TestLabel1, TestPatch2,
                       TestLabel2, batchsize):
    # Convert patches and labels to PyTorch tensors
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


# 10. Accuracy metrics
def acc_reports(y_test, y_pred_test):
    target_names = ['1', '2', '3', '4', '5', '6']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return (classification, round(oa * 100, 2), confusion,
            np.round(each_acc * 100, 2), round(aa * 100, 2), round(kappa * 100, 2))


# 11. Per-class accuracy and average accuracy
def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


# 12. Print metrics
def AngPrint(oa, aa, kappa):
    print('OA:', oa)
    print('AA:', aa)
    print('Kappa:', kappa)
    print("\n")


# Add salt-and-pepper noise
def add_salt_and_pepper_noise(data, noise_ratio, salt_ratio=0.5, pepper_ratio=0.5):
    num_noise = int(data.shape[0] * noise_ratio)
    rand_idx = np.random.choice(data.shape[0], num_noise, replace=False)

    # Add salt-and-pepper noise
    for idx in rand_idx:
        # Get the data dimensions
        _, c, h, w = data.shape

        # Compute the center position
        center_h = h // 2
        center_w = w // 2

        # Generate a random mask (0 for pepper noise, 1 for salt noise)
        noise = np.random.randint(0, 2, size=(h, w))
        print(noise.shape)

        # Replace pixels with the corresponding noise values
        data[idx, :, center_h, center_w] = np.where(noise == 0, 0, data[idx, :, center_h, center_w])
        data[idx, :, center_h, center_w] = np.where(noise == 1, 1, data[idx, :, center_h, center_w])

    # Scale the noise values according to the salt/pepper ratios
    salt_value = np.max(data)
    pepper_value = np.min(data)
    noisy_data = np.copy(data)
    noisy_data[data == salt_value] = salt_ratio * salt_value
    noisy_data[data == pepper_value] = pepper_ratio * pepper_value

    return noisy_data


def addn2(data):
    num_points = 283

    # Randomly select center points
    center_indices = np.random.choice(np.arange(data.shape[0]), size=num_points, replace=False)
    for idx in center_indices:
        center = data[idx, :, 5, 5]  # The center pixel is located at (5, 5)
        mask = np.random.choice([0, 1], size=center.shape, p=[0.5, 0.5])  # 50% random mask
        noise = mask  # Half of the values are set to 0 and half to 1
        data[idx, :, 5, 5] += noise  # Add noise
    return data
