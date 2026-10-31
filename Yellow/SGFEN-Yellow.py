import torch.nn as nn
import time
from tqdm import tqdm
from model.SGFEN import SGFEN
from utils import *
import warnings


def count_parameters(model):
    """
    Count the number of trainable model parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


warnings.filterwarnings('ignore')  # Suppress possible warnings
t = 1  # Number of training runs
epoch = 100
roa = []
raa = []
rka = []
rea = []
for i in range(t):
    hpath, lpath, labelpath = selectdataset('cl')
    print(hpath)
    # 2. Set random seeds
    setseed(0)
    # 3. Set parameters
    patch = 23
    lr = 0.001
    band = 20
    patchsize1, patchsize2, batchsize, epoch, lr, NC = setfig(patch, patch, 64, epoch, lr, band)
    # 4. Load dataset
    data1, data2, trlabel, telabel = loaddataset(hpath, lpath, labelpath)
    # 5. Data preprocessing 1 - Standardization of HSI and LiDAR
    data1, data2 = Datapreprocessing1(data1, data2)
    # 6. Data preprocessing 2 - PCA dimensionality reduction of the HSI
    nc = NC
    if NC < 100:  # Below 100, dimensionality reduction is skipped
        nc = band
        data1 = Datapreprocessing2(data1, nc)
    print('data1.shape:', data1.shape)
    print('data1.shape:', data2.shape)
    # 7. Data preprocessing 3 - Boundary padding for HSI and LiDAR
    x1, x2, pad_width1, pad_width2 = Datapreprocessing3(data1, data2, patchsize1, patchsize2, nc)
    # 8. Data preprocessing 4 - Generate patches
    TrainPatch1, TrainLabel1 = Datapreprocessing4(x1, trlabel, patchsize1, pad_width1, nc)  # HSI training patches
    TestPatch1, TestLabel1 = Datapreprocessing4(x1, telabel, patchsize1, pad_width1, nc)  # HSI test patches
    TrainPatch2, TrainLabel2 = Datapreprocessing4(x2, trlabel, patchsize2, pad_width2, 1)  # LiDAR training patches
    TestPatch2, TestLabel2 = Datapreprocessing4(x2, telabel, patchsize2, pad_width2, 1)  # LiDAR test patches
    print('HSI Tr+Te:', TrainPatch1.shape, 'and', TestPatch1.shape)
    print('LiDAR Tr+Te:', TrainPatch2.shape, 'and', TestPatch2.shape)
    # 9. Data preprocessing 5 - Build data loaders
    train_loader, test_loader, Classes = Datapreprocessing5(
        TrainPatch1, TrainLabel1, TrainPatch2, TrainLabel2,
        TestPatch1, TestLabel1, TestPatch2, TestLabel2, batchsize)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SGFEN(input_channels=NC, input_channels2=1).to(device)
    print(f"Params: {count_parameters(model)}")

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    BestAcc = 0
    start = time.time()

    Q = range(1, epoch + 1)
    pbar = tqdm(Q, colour='green', ncols=100)
    for epoch in pbar:
        total_loss = 0
        for step, (b_x1, b_x2, b_y) in enumerate(train_loader):
            b_x1, b_x2, b_y = b_x1.to(device), b_x2.to(device), b_y.to(device)
            out = model(b_x1, b_x2)
            loss = loss_func(out, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        pbar.set_description(
            '[epoch: %d]   [average loss: %.4f]   [current loss: %.4f]'
            % (epoch + 1, total_loss / (epoch + 1), loss.item()))
    end = time.time()
    print(end - start)
    Train_time = end - start

    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    for b_x1, b_x2, b_y in test_loader:
        b_x1, b_x2 = b_x1.to(device), b_x2.to(device)
        outputs = model(b_x1, b_x2)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = b_y
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, b_y))

    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    # Print accuracy metrics
    print("-----------ITEM:{} (epoch={}):------------".format(i+1, epoch))
    AngPrint(oa, aa, kappa)
    roa.append(oa)
    raa.append(aa)
    rka.append(kappa)
    rea.append(each_acc)

    file_name = 'result/SGFEN.txt'
    with open(file_name, 'a+') as x_file:
        x_file.write("-----------ITEM:{}:------------".format(i+1))
        x_file.write('\n')
        x_file.write('{} OA (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} AA (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Kappa (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} EA (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))
        x_file.write('\n')


soa = round(sum(roa) / len(roa), 2)
saa = round(sum(raa) / len(raa), 2)
ska = round(sum(rka) / len(rka), 2)
sea = np.round(sum(rea) / len(rea), 2)
file_name = 'result/SGFEN.txt'
print(soa)
print(saa)
print(ska)
print(sea)
if t > 1:
    with open(file_name, 'a+') as x_file:
        x_file.write("-----------AVERAGE:------------")
        x_file.write('\n')
        x_file.write('{} AVOA (%)'.format(soa))
        x_file.write('\n')
        x_file.write('{} AVAA (%)'.format(saa))
        x_file.write('\n')
        x_file.write('{} AVKappa (%)'.format(ska))
        x_file.write('\n')
        x_file.write('{} AVEA (%)'.format(sea))
        x_file.write('\n')
        x_file.write("-----------END------------")
        x_file.write('\n')
