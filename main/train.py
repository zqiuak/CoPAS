import shutil
import os
import time
import argparse
import logging
import sys
import pdb
import glob
import pickle
import csv

import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch import autocast
from torch.cuda.amp import GradScaler

from torch.utils.tensorboard import SummaryWriter


from dataloader import kneeDataSetSITK
from utils import Aggremeter, write_metrix, evaluate_prediction
import model
import warnings
warnings.simplefilter("ignore")

from Args import Arguments

Net = model.Multi_view_Knee

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--weight_path', type=str, default='')
    parser.add_argument('--prefix_name', type=str, default=Net.__name__)
    parser.add_argument('--a',type=str,default='')
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--gpu',type=str, default='2')
    parser.add_argument('--half', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parse_args = parser.parse_args()
    args = Arguments()
    args.transfer_from_argparse(parse_args)
    return args

def train_model(model, train_loader, epoch, optimizer, act_task, scalar, args:Arguments):
    model.train()

    agg_meter = Aggremeter()
    tbar = tqdm(train_loader)
    final = True
    for i, (images, label, _) in enumerate(tbar):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()


        if args.half:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits= model(images) # prediction= (final_pred, sag_pred, cor_pred, axi_pred)
                loss, loss_value = model.criterion(logits, label, act_task, final)
            scalar.scale(loss).backward()

            if (i+1) % args.iters_to_accumulate == 0 or (i+1) == len(train_loader):
                scalar.step(optimizer)
                scalar.update()

            logits = [x.type(torch.FloatTensor) for x in logits]

        else:
            logits= model(images) # prediction= (final_pred, sag_pred, cor_pred, axi_pred)
            loss, loss_value = model.criterion(logits, label, act_task, final)
            loss.backward()
            if (i+1) % args.iters_to_accumulate == 0 or (i+1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
        
        
        prediction = [torch.sigmoid(x).tolist() for x in logits]
        gt = label.tolist()
        agg_meter.add_pred(prediction[0], gt)
        agg_meter.add_loss(loss_value)

        if args.debug:
            break

    auc_dict, acc_dict, metrix_dict = evaluate_prediction(agg_meter.pred_list, agg_meter.label_list, metrix_output=True)
    write_metrix(metrix_dict, args.log_root_folder, "train_final", args)

    agg_meter.add_metrix(list(acc_dict.values()), list(auc_dict.values()))
    train_loss_epoch = agg_meter.loss
    train_auc_epoch = agg_meter.auc
    train_acc_epoch = agg_meter.acc
    return train_loss_epoch, train_auc_epoch, train_acc_epoch

def evaluate_model(model, val_loader, epoch, args:Arguments, mode = 'Val', save_path=None):
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    agg_meter = Aggremeter()
    sag_agg_meter = Aggremeter()
    cor_agg_meter = Aggremeter()
    axi_agg_meter = Aggremeter()

    tbar = tqdm(val_loader)
    for i, (images, label, id) in enumerate(tbar):
        
        
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()

        logits = model(images) # prediction= (final_pred, sag_pred, cor_pred, axi_pred)
        _, loss_value = model.criterion(logits, label)


        prediction = [torch.sigmoid(x).tolist() for x in logits]
        gt = label.tolist()
        agg_meter.add_pred(prediction[0], gt)
        agg_meter.add_loss(loss_value)
        sag_agg_meter.add_pred(prediction[1], gt)
        cor_agg_meter.add_pred(prediction[2], gt)
        axi_agg_meter.add_pred(prediction[3], gt)
        
        if args.debug:
            break

    _, _, sag_metrix_dict = evaluate_prediction(sag_agg_meter.pred_list, sag_agg_meter.label_list, metrix_output=True)
    _, _, cor_metrix_dict = evaluate_prediction(cor_agg_meter.pred_list, cor_agg_meter.label_list, metrix_output=True)
    _, _, axi_metrix_dict = evaluate_prediction(axi_agg_meter.pred_list, axi_agg_meter.label_list, metrix_output=True)
    final_auc, final_acc, fin_metrix_dict = evaluate_prediction(agg_meter.pred_list, agg_meter.label_list, metrix_output=True)

    agg_meter.add_metrix(list(final_acc.values()), list(final_auc.values()) )
    val_loss_epoch = agg_meter.loss
    val_auc_epoch = agg_meter.auc
    val_acc_epoch = agg_meter.acc


    with open(os.path.join(save_path, 'results%s.pk'%mode),'wb+') as outfile:
        pickle.dump(agg_meter, outfile)
    write_metrix([fin_metrix_dict, sag_metrix_dict, cor_metrix_dict, axi_metrix_dict], save_path, [mode + "fin", "sag", "cor", "axi"], args)

    return val_loss_epoch, val_auc_epoch, val_acc_epoch

def run(args:Arguments):
    net = Net(backbone='ResNet3D', encoder_layer=args.model_depth, pretrain=True, kargs=args)
    log_root_folder = args.ExpFolder+"/Exp-{}-{}-{}/".format(args.prefix_name,time.strftime("%m%d-%H%M%S"), args.a)

    setattr(args, "log_root_folder", log_root_folder)
    MODELSPATH = log_root_folder
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)
    os.makedirs(log_root_folder)

    script_save_folder = os.path.join(log_root_folder, 'scripts')
    os.mkdir(script_save_folder)
    for script in glob.glob("*.py"):
        dst_file = os.path.join(log_root_folder, 'scripts', os.path.basename(script))
        shutil.copyfile(script, dst_file)

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(log_root_folder, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", args)
    
    # net = torch.compile(net)

    augmentor = args.Augmentor

    dataset_name = "Internal"
    train_dataset = kneeDataSetSITK('train', dataset_name=dataset_name, transform=augmentor, aug_rate=2, use_cache=args.use_cache, args=args)

    validation_dataset = kneeDataSetSITK('val', transform=None, use_cache=args.use_cache, args=args)

    test_dataset = kneeDataSetSITK('test', transform=None, use_cache=args.use_cache, args=args)
    
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.SGD(params=net.parameters(), lr = args.lr, weight_decay=0.1)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9, verbose=True)

    best_val_loss = float('inf')
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()
    
    act_task = -1 # which class to train

    for epoch in range(num_epochs):

        t_start = time.time()
        
        train_dataset.balance_cls(act_task)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        scalar = GradScaler()
        train_loss, train_auc, train_acc = train_model(
            net, train_loader, epoch, optimizer, act_task, scalar, args)

        if args.data_balance:
            if act_task == 11:
                act_task = -1
            else:
                act_task = act_task+1

        if epoch % 20 == 0:
            scheduler.step()
        
        with torch.no_grad():
            val_loss, val_auc, val_acc = evaluate_model(
                net, validation_loader, epoch, args, save_path=log_root_folder)

            test_loss, test_auc, test_acc = evaluate_model(
                net, test_loader, epoch, args, save_path=log_root_folder, mode='Test')


        t_end = time.time()
        delta = t_end - t_start

        logging.info(
            "Epoch {10}\n train loss {0} | train auc {1} | train acc {2} |\n val loss {3} | val auc {4} | val acc {5} |\n test loss {6} | test auc {7} | test acc {8} elapsed time {9} s".format(
                train_loss, train_auc, train_acc, val_loss, val_auc, val_acc, test_loss, test_auc, test_acc, delta, epoch))

        iteration_change_loss += 1
        logging.info('-' * 50)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            net.save_or_load_encoder_para(path=log_root_folder)
            if bool(args.save_model):
                file_name = f'model_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_test_auc_{test_auc:0.4f}_epoch_{epoch + 1}_bestval.pth'
                try:
                    exported_model = net
                    torch.save(exported_model.state_dict(), f'{log_root_folder}/{file_name}')
                except:
                    pass

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            logging.info('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break

    t_end_training = time.time()
    logging.info(f'training took {t_end_training - t_start_training} s')


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)