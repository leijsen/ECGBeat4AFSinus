import os
import csv
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from mne.filter import filter_data, notch_filter
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix

def afDurationCreate(aux):
    seg = [] # 用于存储房颤片段的起始点和终止点
    for i in range(len(aux)):
        if aux[i] == '(AFIB' or aux[i] == 'AFL':
            if len(seg) == 0:
                seg.append(i)
            else:
                continue
        elif aux[i] == '(N':
            if len(seg) != 0:
                seg.append(i)
            else:
                seg.append(i)
                seg.append(i)
        if i == len(aux) - 1 and len(seg) == 1:
            seg.append(i)
        if len(seg) == 2:
            afDuration.append(seg)
            seg = []
    if len(afDuration) == 0:
        afDuration.append([0, 0])
def checkInAf(afDuration, x):
    # print(afDuration)
    # try:
    for temp in afDuration:
        s, e = temp[0], temp[1]
        if x >= s and x < e:
            return True
    return False
    # except:
        
        # print("checkInAf: ", target)
        # print(afDuration)
def checkStable(afDuration, x):
    # 前后10s内不在房颤片段内
    for i in range(x-paratime, x+paratime):
        if checkInAf(afDuration, i):
            return False
    return True
def checkAnan(R_symbol, x):
    # 查看R_symbol前10s或后10s内有A
    # flag = 0
    for i in range(x-paratime, x):
        if R_symbol[i] == 'A':
            return True
            # flag = 1
            # break
    # if flag == 0:
    #     return False
    for i in range(x, x+paratime):
        if R_symbol[i] == 'A':
            return True
    return False
def checkvnvn(R_symbol, x):
    # 查看前10s或后10s内有V
    # flag = 0
    for i in range(x-paratime, x):
        if R_symbol[i] == 'V':
            return True
    # if flag == 0:
    #     return False
    for i in range(x, x+paratime):
        if R_symbol[i] == 'V':
            return True
    return False
def checkBefore(afDuration, x):
    # 查看前10s没有房颤片段，后10s即将进入房颤片段
    for i in range(x-paratime, x):
        if checkInAf(afDuration, i):
            return False
    for i in range(x, x+paratime):
        if checkInAf(afDuration, i):
            return True
    return False
def checkAfter(afDuration, x):
    # 查看前10s有房颤片段，后10s没有房颤片段
    flag = 0
    for i in range(x-paratime, x):
        if checkInAf(afDuration, i):
            flag = 1
            break
    if flag == 0: 
        return False
    for i in range(x, x+paratime):
        if checkInAf(afDuration, i):
            return False
    return True


all_data_path = r"PAF\coding\lj\all_data"
# file_path = "./Net1d_newGroup/PAF_net1d_Group2/list_data.h5"
# file_path = "Net1d_NewGroup_randomindex2\PAF_net1d_randomindex2_Group1\list_data.h5"
# file_path = r"PAF\Net1d_newGroup\PAF_net1d_Group2_testbad/list_data.h5"
# file_path = "Net1d_Split\PAF_net1d_Split1_4\list_data.h5" good
# file_path = "Net1d_Split\PAF_net1d_Split1_4\list_data.h5"
# file_path = "Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-16\list_data16.h5" # 58
# file_path = "Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-17\list_data17.h5" # 67
# file_path = "Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-43\list_data43.h5" # 81(90)
# file_path = "Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-67\list_data67.h5" # 83
# file_path = "Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-103\list_data103.h5" # 75

file_paths = ["Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-16\list_data16.h5", "Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-17\list_data17.h5", "Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-43\list_data43.h5", "Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-67\list_data67.h5", "Net1d_Split1_denoise\PAF_net1d_SplitDenoise_FirstNum-103\list_data103.h5"]
for file_path in file_paths:
    file_path_label = file_path.split('\\')[1].split('-')[1]
    print("file_path_label: ", file_path_label)

    hf_keys = []
    data = {}
    with h5py.File(file_path, 'r') as hf:
        for key in hf.keys():
            hf_keys.append(key)
            data[key] = hf[key][:]
    # print("hf_keys: ", hf_keys)
    print("len(hf_keys): ", len(hf_keys))

    people_info = pd.read_csv(r"PAF\coding\data_distil\people_info.csv")
    key_value = people_info.set_index('Key').to_dict()['Value']
    # print(key_value)

    global paratime
    paratime = 10
    normalPro = []
    normallabel = []
    testStableNormalPro = []
    Stablelabel = []
    testAnanNormalPro = []
    Ananlabel = []
    testBeforeNormalPro = []
    Beforelabel = []
    testAfterNormalPro = []
    Afterlabel = []
    vnvnPro = []
    vnvnlabel = []
    for target in hf_keys:
        file_name = target
        record = wfdb.rdrecord(os.path.join(all_data_path, file_name))
        sig, fields = wfdb.rdsamp(os.path.join(all_data_path, file_name))
        label = key_value[int(target.split('_')[1])]

        annotation = wfdb.rdann(os.path.join(all_data_path, file_name), 'atr')
        R_location = annotation.sample
        R_symbol = annotation.symbol
        aux = annotation.aux_note
        p_signal = record.p_signal[:, 0]
        p_signal = p_signal.flatten()

        # 滤波操作
        sfreq = 200
        low_freq = 0.5
        high_freq = 50.0
        p_signal = filter_data(p_signal, sfreq, low_freq, high_freq, method='iir', verbose=False)

        global afDuration
        afDuration = []
        afDurationCreate(aux)

        try:
            # 获得概率值
            seg_data = data[target]
            # print(seg_data.shape)
            seg_pro = []
            for i in range(len(seg_data)):
                seg_pro.append(round(seg_data[i][-2], 2))
            
            # AF_location = []
            # for i in range(len(R_symbol))[10:-5]:
            #     if R_symbol[i] != 'N':
            #         AF_location.append(i+1)

            # print("len(R_symbol): ", len(R_symbol)-15)
            # print("len(seg_pro): ", len(seg_pro))
            # print("len(Af_location): ", len(AF_location))
            
            
            lengthSymbol = len(R_symbol)
            # 概率变化情况，因只统计了N，所以要将我们的start和end转换为N的start和end
            all_seg_pro = [0 for i in range(10)]
            index = 0 # 测试中没有将V，A的概率值加入，所以要用index来计数
            for i in range(lengthSymbol)[10:-5]:
                if label == 1 and R_symbol[i] == 'N':
                    # 判断i是否越界和判断N的类型
                    # stable
                    if i >= 10 and i+10 < lengthSymbol and not checkInAf(afDuration, i) and checkStable(afDuration, i):
                        testStableNormalPro.append(seg_pro[index])
                        Stablelabel.append(label)
                    # anan
                    if i >= 10 and i+10 < lengthSymbol and checkAnan(R_symbol, i):
                        testAnanNormalPro.append(seg_pro[index])
                        Ananlabel.append(label)
                    # vnvn
                    if i >= 10 and i+10 < lengthSymbol and checkvnvn(R_symbol, i):
                        vnvnPro.append(seg_pro[index])
                        vnvnlabel.append(label)
                    # before
                    if i >= 10 and i+10 < lengthSymbol and not checkInAf(afDuration, i) and checkBefore(afDuration, i):
                        testBeforeNormalPro.append(seg_pro[index])
                        Beforelabel.append(label)
                    # after
                    if i >= 10 and i+10 < lengthSymbol and not checkInAf(afDuration, i) and checkAfter(afDuration, i):
                        testAfterNormalPro.append(seg_pro[index])
                        Afterlabel.append(label)
                    index += 1
                elif label == 0 and R_symbol[i] == 'N':
                    normalPro.append(seg_pro[index])
                    normallabel.append(label)
                    index += 1
            # print("index: ", index)
        except:
            pass


    # SplitAucLabel = [Stablelabel, Ananlabel, Beforelabel]
    # SplitAucPro = [testStableNormalPro, testAnanNormalPro, testBeforeNormalPro]
    # SplitAucname = ['Stable', 'Anan', 'Before']
    # SplitAucLabel = [Stablelabel]
    # SplitAucPro = [testStableNormalPro]
    # SplitAucname = ['Stable']
    # SplitAucLabel = [Ananlabel]
    # SplitAucPro = [testAnanNormalPro]
    # SplitAucname = ['Anan']
    # SplitAucLabel = [Beforelabel]
    # SplitAucPro = [testBeforeNormalPro]
    # SplitAucname = ['Before']
    # SplitAucLabel = [Stablelabel, Ananlabel, Beforelabel, Afterlabel, vnvnlabel]
    # SplitAucPro = [testStableNormalPro, testAnanNormalPro, testBeforeNormalPro, testAfterNormalPro, vnvnPro]
    # SplitAucname = ['Stable', 'Anan', 'Before', 'After', 'vnvn']
    SplitAucLabel = [Stablelabel, Beforelabel, Afterlabel, Ananlabel, vnvnlabel]
    SplitAucPro = [testStableNormalPro, testBeforeNormalPro, testAfterNormalPro, testAnanNormalPro, vnvnPro]
    SplitAucname = ['Stable', 'Before', 'After', 'Anan', 'vnvn']

    for label, pro, name in zip(SplitAucLabel, SplitAucPro, SplitAucname):
        label += normallabel
        pro += normalPro
        all_labels = np.array(label)
        all_probabilities = np.array(pro)
        binary_predictions = [1 if prob >= 0.5 else 0 for prob in all_probabilities]
        # print("name: ", name)
        # print("real label 1 num, 0 num: ", label.count(1), " ", label.count(0))
        # print("predict 1 num, 0 num: ", binary_predictions.count(1), " ", binary_predictions.count(0))

        acc = accuracy_score(all_labels, binary_predictions)
        recall = recall_score(all_labels, binary_predictions)
        precision = precision_score(all_labels, binary_predictions)
        f1 = f1_score(all_labels, binary_predictions)
        auc = roc_auc_score(all_labels, all_probabilities)

        # 将结果写入CSV文件
        if not os.path.exists("./DenoiseResultSplitTypes"):
            os.makedirs("./DenoiseResultSplitTypes")
        with open(f'./DenoiseResultSplitTypes/{name}_metrics{paratime}s.csv', mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 如果文件不存在，写入表头
            if csvfile.tell() == 0:
                writer.writerow(['Index', 'Accuracy', 'Recall', 'Precision', 'F1', 
                                 'AUC',
                                 'NormalNum',
                                 'AbnormalNum', 
                                 'Num'])
                writer.writerow([file_path_label, 
                                f"{acc:.4f}", 
                                f"{recall:.4f}", 
                                f"{precision:.4f}", 
                                f"{f1:.4f}", 
                                f"{auc:.4f}", 
                                len(normalPro),
                                len(all_labels)-len(normalPro),
                                len(all_labels)])


        # 打印结果
        # print(f"{name} - 准确率: {acc:.2f}, 召回率: {recall:.2f}, 精确度: {precision:.2f}, F1值: {f1:.2f}, AUC值: {auc:.2f}")
all_dfs = []
for name in SplitAucname:
    folderPath = f"./DenoiseResultSplitTypes/{name}_metrics{paratime}s.csv"
    df = pd.read_csv(folderPath)
    avgValues = df.mean()
    avgDf = pd.DataFrame({"Index": ["Avg"],
                        "Accuracy": [f"{avgValues['Accuracy']:.2f}".zfill(5)],
                        "Recall": [f"{avgValues['Recall']:.2f}".zfill(5)],
                        "Precision": [f"{avgValues['Precision']:.2f}".zfill(5)],
                        "F1": [f"{avgValues['F1']:.2f}".zfill(5)],
                        "AUC": [f"{avgValues['AUC']:.2f}".zfill(5)],
                        "NormalNum": [f"{avgValues['NormalNum']:.0f}".zfill(5)],
                        "AbnormalNum": [f"{avgValues['AbnormalNum']:.0f}".zfill(5)],
                        "Num": [f"{avgValues['Num']:.0f}".zfill(5)]})
    resultDf = pd.concat([df, avgDf], ignore_index=True)
    resultDf.to_csv(folderPath, index=False)
    print("Saved to: ", folderPath)
    all_dfs.append(resultDf)
    all_dfs.append(pd.DataFrame({"name": [name]}))
merged_df = pd.concat(all_dfs, ignore_index=True)
# 保存结果到新的 CSV 文件
output_path = f"./DenoiseResultSplitTypes/merged_metrics{paratime}.csv"
merged_df.to_csv(output_path, index=False)
print(f"Merged metrics saved to {output_path}")
