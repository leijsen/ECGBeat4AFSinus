import sys

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import argparse

from My_util import *
from Ecgdata_getN import Ecgdata
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset
from Net1d_model import Net1D
# from tensorboardX import SummaryWriter
# from torchsummary import summary
from transformers import logging
logging.set_verbosity_warning()
from sklearn.metrics import roc_auc_score, roc_curve, auc, recall_score, precision_score
from sklearn import metrics
np.seterr(divide='ignore', invalid='ignore')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

def _cal_acc(pred, real):
    C = np.zeros((2, 2))
    for i in range(len(pred)):
        C[pred[i], real[i]] += 1
    acc = np.sum(np.diag(C))/np.sum(C)
    return C, acc

def _cal_recall(C):
    tn = C[0, 0]
    fn = C[0, 1]
    fp = C[1, 0]
    tp = C[1, 1]
    recall = tp / (tp + fn)
    return recall

def _cal_precision(C):
    tn = C[0, 0]
    tp = C[1, 1]
    fn = C[0, 1]
    fp = C[1, 0]
    precision = tp / (tp + fp)
    return precision

def _cal_F1(C):
    rec = C[1, 1] / (C[1, 1] + C[0, 1])
    pre = C[1, 1] / (C[1, 0] + C[1, 1])
    if pre + rec == 0:
        F1 = 0
    else:
        F1 = 2 * pre * rec / (pre + rec)
    return F1

def roc_image(all_probabilities, all_labels, epoch, way, save_num):
    # Converts the saved result into a one-dimensional array
    # all_probabilities = np.concatenate(all_probabilities)
    # all_labels = np.concatenate(all_labels)

    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"/data/lj/PAF_net1d{save_num}/{way}_epoch{epoch}_ROC")
    plt.close()
    # plt.show()

def train_Mymodel(train_loader, valid_loader, dataset, Ecgdata):
    save_num = config.save_num
    folder_path = f"/data/lj/PAF_net1d{save_num}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # global model
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in (range(config.epoch)):
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

        train_processor_bar = tqdm(range(len(train_loader)), mininterval=5, leave=True, file=sys.stderr)
        # test_processor_bar = tqdm(range(len(valid_loader)))

        # train
        model.train()
        loss_list = []
        acc_list = []
        confusion_matrix = np.zeros((2, 2))
        train_prob_all = []
        train_label_all = []
        outputs_list = []

        for (data, target) in (train_loader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            softmax = nn.Softmax(-1)
            output = softmax(output)
            loss.backward()
            optimizer.step()
            softmax = nn.Softmax(-1)
            output = softmax(output)

            with torch.no_grad():
                train_prob_all.extend(np.round(output[:, 1].cpu().numpy(), 2))
                train_label_all.extend(target.cpu().numpy())
                _, pred = output.max(1)
                # print(pred)
                C, acc = _cal_acc(pred.cpu().numpy(), target.cpu().numpy())
                confusion_matrix += C

            loss_list.append(loss.item())
            acc_list.append(acc)

            train_processor_bar.update(1)
            if train_processor_bar.n % 100 == 0:
                train_processor_bar.set_description(f'loss: {loss.item():f}')
                train_processor_bar.refresh()



        train_loss = np.mean(loss_list)
        train_acc = np.mean(acc_list)
        train_confusion_matrix = confusion_matrix
        train_recall = _cal_recall(train_confusion_matrix)
        train_precision = _cal_precision(train_confusion_matrix)
        train_F1 = _cal_F1(train_confusion_matrix)


        # valid
        model.eval()

        loss_list = []
        acc_list = []
        test_prob_all = []
        test_label_all = []
        confusion_matrix = np.zeros((2, 2))

        with torch.no_grad():
            for data, target in tqdm(valid_loader, file=sys.stderr):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                softmax = nn.Softmax(-1)
                output = softmax(output)

                test_prob_all.extend(np.round(output[:, 1].cpu().numpy(), 2))
                test_label_all.extend(target.cpu().numpy())
                _, pred = output.max(1)

                C, acc = _cal_acc(pred.cpu().numpy(), target.cpu().numpy())
                confusion_matrix += C

                loss_list.append(loss.item())
                acc_list.append(acc)

                # test_processor_bar.set_description(f'loss: {loss.item():f}')
                # test_processor_bar.update(1)

                if epoch == config.epoch - 1:
                    # Iterate through each output and the corresponding label
                    for i in range(output.shape[0]):
                        out = output[i]
                        label = target[i]

                        # Predict probability, ground_truth of data and labels
                        outputs_list.append([data[i].tolist(), out, label])

        test_loss = np.mean(loss_list)
        test_acc = np.mean(acc_list)
        test_confusion_matrix = confusion_matrix
        test_recall = _cal_recall(test_confusion_matrix)
        test_precision = _cal_precision(test_confusion_matrix)
        test_F1 = _cal_F1(test_confusion_matrix)

        print()
        print('--' * 40)
        print('- Epoch: %d ' % epoch)
        print('- Train_loss: %.5f' % train_loss)
        print('- Train_accuracy: {:.5f} %'.format(train_acc * 100))
        print('- Test_loss: %.5f' % test_loss)
        print('- Test_accuracy: {:.5f} %'.format(test_acc * 100))
        print('- Train_recall: {:.5f} %'.format(train_recall * 100))
        print('- Test_recall: {:.5f} %'.format(test_recall * 100))
        print('- Train_precision: {:.5f} %'.format(train_precision * 100))
        print('- Test_precision: {:.5f} %'.format(test_precision * 100))
        print('- train_F1: {:.5f} %'.format(train_F1 * 100))
        print('- test_F1: {:.5f} %'.format(test_F1 * 100))
        try:
            print('- train_AUC: {:.4f} %'.format(roc_auc_score(train_label_all, train_prob_all) * 100))
        except ValueError:
            pass
        try:
            print('- test_AUC: {:.4f} %'.format(roc_auc_score(test_label_all, test_prob_all) * 100))
        except ValueError:
            pass
        print(f'- tn, fn, fp, tp: {int(test_confusion_matrix[0, 0])}, {int(test_confusion_matrix[0, 1])}, {int(test_confusion_matrix[1, 0])}, {int(test_confusion_matrix[1, 1])}')
        print('--' * 40)
        try:
            if epoch == config.epoch// 2 or epoch == config.epoch - 1:
                torch.save(model, f"{folder_path}/model.pth")

                parser = argparse.ArgumentParser()
                parser.add_argument("--variable", type=str)
                args = parser.parse_args()
                if len(sys.argv) > 1:
                    variable = args.variable
                else:
                    variable = ''
                test_set = list(map(int, variable.split(',')))

                file_name = f"{folder_path}/list_data{test_set[0]}.h5"
                with h5py.File(file_name, 'a') as hf:
                    curr = 0
                    for rowindex in Ecgdata.write_list:
                        orginal_data = hf[f"{rowindex}"][:][:, :-2]
                        rownum = orginal_data.shape[0]
                        prob_data = np.array(test_prob_all[curr:curr+rownum]).reshape(rownum, -1)
                        label_data = np.array(test_label_all[curr:curr+rownum]).reshape(rownum, -1)
                        curr += rownum

                        new_data = np.concatenate([orginal_data, prob_data, label_data], axis=1)

                        hf[f"{rowindex}"][:] = new_data

                roc_image(test_prob_all, test_label_all, epoch, 'test', save_num)
        except:
            pass



class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    model = Net1D(
        in_channels=1,
        base_filters=32,
        ratio=1.0,
        filter_list=[16, 32, 32, 40, 40, 64, 64],
        m_blocks_list=[2, 2, 2, 2, 2, 2, 2],
        kernel_size=8,
        stride=1,
        groups_width=4,
        verbose=False,
        n_classes=2
    )
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    def train_once():
        # global model
        # config.data_path = r'/data/lj/data'

        # data = Ecgdata(config.data_path)
        # X = data.data
        # Y = data.labels
        # train_data, valid_data, train_labels, valid_labels = train_test_split(X, Y, test_size=0.2)

        data = Ecgdata(config.data_path)
        train_data, valid_data, train_labels, valid_labels = data.train_data, data.valid_data, data.train_labels, data.valid_labels

        print(len(train_labels), len(train_data))
        print(len(valid_labels), len(valid_data))

        dataset = MyDataset(train_data, train_labels)
        dataset_test = MyDataset(valid_data, valid_labels)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)

        batch_x, batch_y = next(iter(train_loader))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        print('batch_x: ', batch_x)
        print('batch_y: ', batch_y)

        print('batch_x.shape: ', batch_x.shape)
        print('batch_y.shape: ', batch_y.shape)

        print('outputs: ', outputs)
        print('outputs.shape: ', outputs.shape)

        train_Mymodel(train_loader, valid_loader, dataset, data)
        # summary(model, input_size=batch_x.shape, batch_size=config.batch_size, device=device)


    def train_all():
        # global model
        # config.data_path = r'/data/lj/all_data'

        # data = Ecgdata(config.data_path)
        # X = data.data
        # Y = data.labels
        # train_data, valid_data, train_labels, valid_labels = train_test_split(X, Y, test_size=0.2)

        data = Ecgdata(config.data_path)
        train_data, valid_data, train_labels, valid_labels = data.train_data, data.valid_data, data.train_labels, data.valid_labels

        print(len(train_labels), len(train_data))
        print(len(valid_labels), len(valid_data))

        dataset = MyDataset(train_data, train_labels)
        dataset_test = MyDataset(valid_data, valid_labels)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)

        batch_x, batch_y = next(iter(train_loader))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        print('batch_x: ', batch_x)
        print('batch_y: ', batch_y)

        print('batch_x.shape: ', batch_x.shape)
        print('batch_y.shape: ', batch_y.shape)

        print('outputs: ', outputs)
        print('outputs.shape: ', outputs.shape)

        train_Mymodel(train_loader, valid_loader, dataset, data)

    # train_once()
    train_all()