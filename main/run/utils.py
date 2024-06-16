import logging
import os, sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import csv
import numpy as np
from PIL import Image as IMG
import matplotlib
import matplotlib.pyplot as plt
import shutil
import torchvision.datasets as dset
import torch.nn.functional as F
import pickle
import pdb
import time
import glob

from run.Args import Arguments
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix,\
                            average_precision_score, balanced_accuracy_score, f1_score,\
                            precision_score, recall_score, hamming_loss, RocCurveDisplay


def create_exp_dir(path, scripts_to_save=None):
    _create_Dir_(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def _create_Dir_(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return 1
    return 0

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6

def count_memory_in_MB(model, input:torch.Tensor):
    x = input.clone()
    x.requires_grad_(requires_grad=False)
    mods = list(model.modules())
    out_size = []
    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, torch.nn.Relu) and m.inplace:
            continue
        out = m(x)
        out_size.append(np.array(out.size()))
        x = out
    
    total_nums = 0
    for i in range(len(out_size)):
        s = out_size[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    return total_nums*4*2/1e6

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def Show_Histogram(input, title = 'Sample_histogram',save_path = ""):
    if type(input) == list:
        input = np.array(input)
    elif type(input) == torch.Tensor:
        input = input.numpy()
    elif type(input) == np.ndarray:
        pass
    else:
        raise Exception("wrong input type")
    input = input.reshape(-1)
    plt.hist(input)  

    if len(save_path)!=0:
        plt.savefig(os.path.join(save_path,title+'.png'))
        plt.clf()
    else:
        plt.show()

def Show_Samples(input, start_row = 0, title = 'Sample', bcdhw = False, save_path = ""):
    '''
    input: gray image of List[b, c, h, w], Tensor [h, w],[c, h, w],[b, c, h, w],[b, slice, c, h, w], [b, c, d, h, w]
    '''
    sample_rate = 1
    if type(input) == list:
        img_num = len(input)
        if len(input)>25:
            img_num = 25
            sample_rate = len(input)//25
            print("show 25 samples because too many (%d samples)"%img_num)
        from math import ceil
        col = ceil(img_num ** 0.5)
        row =  img_num // col

    elif type(input) == np.ndarray:
        Show_Samples(torch.tensor(input), bcdhw=bcdhw, title=title, save_path=save_path)
        return
    
    elif type(input) == torch.Tensor:
        print('Show samples in shape', input.shape)
        if input.dim() == 2 or input.dim() == 3:
            Show_Samples(input.unsqueeze(0), title=title, save_path=save_path)
            return
        elif input.dim() == 4:
            Show_Samples(input.cpu().tolist(), title=title, save_path =save_path)
            return
        elif input.dim() == 5:
            if bcdhw == True:
                Show_Samples(input[0].transpose(0,1), title=title, save_path =save_path)
                return
            for i, each in enumerate(input):
                Show_Samples(each.cpu().tolist(), start_row=i, title='batch%d'%i if type(title) == str else title[i], save_path =save_path)
            return
        else:
            print("input type for show_samples wrong, now pdb")
            import pdb; pdb.set_trace()


    elif type(input) == dict:
        for i, key in enumerate(input.keys()):
            Show_Samples(input[key], start_row = i, title = key, save_path =save_path)
        return
    
    else:
        print("input type for show_samples wrong, now pdb")
        import pdb; pdb.set_trace()

    Fig = plt.figure(title)
    figidx = 1
    for i, img in enumerate(input):
        if i%sample_rate != 0:
            continue
        plt.subplot(row, col, start_row*col+figidx)
        plt.imshow(img[0], 'gray')
        figidx+=1
        if figidx>row*col:
            break
    if len(save_path)!=0:
        plt.savefig(os.path.join(save_path,title+'.png'))
        plt.clf()
    else:
        plt.show()


class Aggremeter():
    def __init__(self) -> None:
        self.clear()

    def clear(self):
        self.loss_list = []
        self.pred_list = []
        self.label_list = []
        self.acc_list = []
        self.auc_list = []
        self.cal()

    def add_pred(self, pred:list, label:list):
        self.pred_list.extend(pred)
        self.label_list.extend(label)
    
    def add_metrix(self, acc:list, auc:list):
        self.acc_list.append(acc)
        self.auc_list.append(auc)
        self.cal()

    def add_loss(self, loss:float):
        self.loss_list.append(loss)
        self.cal()

    def cal(self):
        self.loss = np.round(np.mean(self.loss_list), 4)
        self.acc = np.round(np.mean(self.acc_list), 4)
        self.auc = np.round(np.mean(self.auc_list), 4)

def save_checkpoint(chkpath, fold, model, optimizer=None, scheduler=None, epoch=-1):
    chkpoint = {
        "fold": fold,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": (optimizer.state_dict if optimizer != None else 0),
        "scheduler": (scheduler.state_dict if scheduler != None else 0),
    }
    torch.save(chkpoint, chkpath)

def load_checkpoint(chkpath, current_fold, model, optimizer=None, scheduler=None):
    chekpoint = torch.load(chkpath)
    if current_fold != chekpoint["fold"]:
        return -1
    model.load_state_dict(chekpoint["model"])
    if optimizer != None:
        optimizer.load_state_dict(chekpoint["optimizer"])
    if scheduler != None:
        scheduler.load_state_dict(chekpoint["scheduler"])
    return chekpoint["epoch"]

def doctor_eval(file_path, write_output=False, save_path = None):
    import pandas as pd
    if write_output:
        assert save_path != None
    f = pd.read_csv(file_path, header=None)
    doc_num = f.shape[1]//13-1
    doc_pred = []
    doc_metrix = []
    id = f.iloc[:, 0].values
    Label = f.iloc[:, 1:13].values
    for i in range(doc_num):
        doc_pred.append(f.iloc[:, (i+1)*13+1:(i+2)*13].values)
        _, _, metric = evaluate_prediction(doc_pred[i], Label, metrix_output=True)
        # write metric:
        if write_output:
            write_metrix(metric, save_path, "doctor_%d"%i, overwrite=(i==0), savename="doc_metrix")
        doc_metrix.append(metric)
    return id, Label, doc_pred, doc_metrix

def evaluate_prediction(prediction:list, label:list, metrix_output = False, args=Arguments()):
    prediction = np.array(prediction)
    label = np.array(label)
    # prob_dict = {} # {taskname: prediction}
    auc_dict = {}
    acc_dict = {}
    metrix_dict = {}
    for i, task in enumerate(args.DiseaseList):
        preds = prediction[:,i]
        trues = label[:, i]
        auc_dict[task], acc_dict[task], metrix_dict[task] = summary(trues, preds)
        
    if metrix_output:
        one_pred = np.ones_like(prediction)*(prediction>=0.5)
        ACC = accuracy_score(label, one_pred)
        mAP = average_precision_score(label, prediction)
        CP = precision_score(label, one_pred, average="macro")
        OP = precision_score(label, one_pred, average="micro")
        CR = recall_score(label, one_pred, average="macro")
        OR = recall_score(label, one_pred, average="micro")
        CF1 = f1_score(label, one_pred, average="macro")
        OF1 = f1_score(label, one_pred, average="micro")
        Hamming = hamming_loss(label,one_pred)
        metrix_dict["overall"] = [ACC, mAP, CP, OP, CR, OR, CF1, OF1, Hamming]
        return auc_dict, acc_dict, metrix_dict
    else:
        return auc_dict, acc_dict

def summary(label, prob_pred):
    pred = np.ones_like(prob_pred)*(prob_pred>=0.5)
    acc = accuracy_score(label, pred)
    balance_acc = balanced_accuracy_score(label,pred)
    try:
        TN, FP, FN, TP = confusion_matrix(label, pred).ravel()
    except:
        TN, FP, FN, TP = [-1,-1,-1,-1]
    precision = precision_score(label, pred)
    recall = recall_score(label, pred)
    f1 = f1_score(label, pred)
    try:
        auc = roc_auc_score(label, prob_pred)
    except:
        auc = -1
    return auc, acc, [balance_acc, TN, FN, TP, FP, precision, recall, f1, auc, acc]

def write_metrix(metrix_dict, save_path, view, args=Arguments(), overwrite = False, savename="metrix"):
    per_class_metric_name = ['Task' , "balance_acc", "TN", "FN", "TP", "FP", "precision", "recall", "f1", "auc", "acc"]
    overall_metric_name = ["ACC", "mAP", "CP", "OP", "CR", "OR", "CF1", "OF1", "Hamming"]
    per_class_pad = 1
    overall_pad = 1
    if len(per_class_metric_name)>=len(overall_metric_name):
        overall_pad += len(per_class_metric_name)-len(overall_metric_name)
    else:
        per_class_pad += len(overall_metric_name)-len(per_class_metric_name)
    if overwrite:
        f = open(os.path.join(save_path, '%s.csv'%savename),'w+')
    else:
        f = open(os.path.join(save_path, '%s.csv'%savename),'a+')
    f_csv = csv.writer(f)
    multi = False
    if type(metrix_dict) == list:
        assert type(view) == list
        assert len(metrix_dict) == len(view)
        metrix_dict_list = metrix_dict
        view_list = view
        multi = True
    Buffer = []
    # per class
    if multi:
        per_class_header = []
        for v in view_list:
            per_class_header.extend([v, *per_class_metric_name, *([" "]*per_class_pad)])
    else:
        per_class_header = [view, *per_class_metric_name]
    Buffer.append(per_class_header)
    
    for task in args.DiseaseList:
        per_class_metric = []
        if multi:
            for m_dict in metrix_dict_list:
                per_class_metric.extend([" ", task, *m_dict[task], *([" "]*per_class_pad)])
        else:
            per_class_metric = [" ", task, *metrix_dict[task]]

        Buffer.append(per_class_metric)

    # overall
    overall_header = []
    if multi:
        for v in view_list:
            overall_header.extend([" ", *overall_metric_name, *([" "]*overall_pad)])
    else:
        overall_header = [" ", *overall_metric_name]
    Buffer.append(overall_header)

    overall_metric = []
    if multi:
        for m_dict in metrix_dict_list:
            overall_metric.extend([" ", *m_dict["overall"], *([" "]*overall_pad)])
    else:
        overall_metric = [" ", *metrix_dict["overall"]]
    Buffer.append(overall_metric)

    Buffer.append(["-------"]*15)
    f_csv.writerows(Buffer)
    f.close()


def load_doc_eval(filepath, kargs:Arguments):
    _, _, _, doc_metrix = doctor_eval(filepath)
    extra_dot = {}
    for task in kargs.DiseaseList:
        doc_dot = []
        dot_x = []
        dot_y = []
        for i in range(len(doc_metrix)):
            TN, FN, TP, FP = doc_metrix[i][task][1:5]
            tpr = TP/(TP+FN)
            fpr = FP/(FP+TN)
            dot_x.append(fpr)
            dot_y.append(tpr)
            doc_dot.append([fpr, tpr])
        extra_dot[task] = doc_dot
    return extra_dot

def save_pred(pred, label, save_path, kargs:Arguments):
    with open(os.path.join(save_path,"./pred_save.csv"), "w+") as f:
        outfile = csv.writer(f)
        for y, each in zip(label, kargs.DiseaseList):
            with open(os.path.join(kargs.ExpFolder, "%s_pred.pkl"%each), "rb") as infile:
                pred = pickle.load(infile)
                outfile.writerow([each, *pred.tolist()])
                outfile.writerow([each, *y.tolist()])

def plot_roc(fpr, tpr, auc, title_str = 'roc', extra_dot = [], save_path="", extra_dot2 = []):
    font = {'family' : 'arial',
        # 'weight' : 'bold',
        'size'   : 12}
    colormap = plt.get_cmap('Paired')(range(2,10))
    plt.rc('font', **font)
    dot_format = ["o", "x", "+", "d", "*", "^"]
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name=title_str)
    display.plot()
    if len(extra_dot) != 0:
        for i, doc_dot in enumerate(extra_dot):
            plt.plot([doc_dot[0]], [doc_dot[1]], dot_format[i], color=colormap[i], label="Junior Radiologist %d"%i if i<5 else "Senior Radiologist")
        plt.legend()
    if len(extra_dot2) != 0:
        for i, doc_dot in enumerate(extra_dot2):
            ox = extra_dot[i][0]
            oy = extra_dot[i][1]
            nx = doc_dot[0]
            ny = doc_dot[1]
            plt.plot([doc_dot[0]], [doc_dot[1]], dot_format[i], color=colormap[i])
            plt.arrow(ox, oy, nx-ox, ny-oy, head_width=0.02, length_includes_head=True, color="darkgray")
        plt.legend()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.savefig(os.path.join(save_path,'%s.png'%title_str),bbox_inches='tight')
    plt.clf()
    return

