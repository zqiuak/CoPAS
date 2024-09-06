import shutil
import os
import time
import logging
import sys
import numpy as np
import glob
from typing import Any

import torch
from pytorch_grad_cam import GradCAM, LayerCAM, GradCAMElementWise, GradCAMPlusPlus, EigenCAM, FullGrad, EigenGradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from tensorboardX import SummaryWriter

from run.Args import args
from run.utils import Show_Samples
from run.train import evaluate_model
from data.dataloader import kneeDataSetSITK
import matplotlib.pyplot as plt
import warnings
from pytorch_grad_cam import base_cam
warnings.simplefilter("ignore")


class Output_Target:
    def __init__(self, task:int) -> None:
        self.category = task
    
    def __call__(self, model_output, *args: Any, **kwds: Any) -> Any:
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

class ReshapeTransform():
    def __init__(self, slice_indx) -> None:
        self.slice_indx = slice_indx

    def __call__(self, input, *args: Any, **kwds: Any) -> Any:
        assert input.dim() == 5
        emb_dim = input.shape[2]
        indx = int(emb_dim*self.slice_indx//224)
        return input[:, :, indx, :, :]

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, input, branch = 0, img_indx = 0, slice_indx = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available:
            input = [x.cuda() for x in input]
        self.input = input
        self.model = model
        self.img_indx = img_indx
        self.slice_indx = slice_indx
        self.branch = branch 
        self.target_layer = model.axi_enc.model.layer4[-1]

    def forward(self, input_img):
        x = self.input[self.img_indx]
        x = torch.cat((x[:, :self.slice_indx], input_img, x[:, self.slice_indx+1:]), dim=1)
        images = [*self.input[:self.img_indx],x,*self.input[self.img_indx+1:]]
        images = torch.stack([image.unsqueeze(0) for image in images]) # 5, b, c, d, h, w
        output = self.model(images)
        output = [torch.sigmoid(x) for x in output]
        return output[self.branch]
    
    def old_forward(self, input_img):
        # for old codes that input [slice, 3, h, w]
        x = self.input[self.img_indx]
        x = torch.cat((x[:self.slice_indx], input_img, x[self.slice_indx+1:]), dim=0)
        images = [*self.input[:self.img_indx],x,*self.input[self.img_indx+1:]]
        images = torch.stack([image.unsqueeze(0) for image in images]) # 5, b, c, d, h, w
        output = self.model(images)
        output = [torch.sigmoid(x) for x in output]
        return output[self.branch]


def show_CAM(net, dataset):
    # cam args
    target_task = 1
    images, _, _ = dataset[0]
    output_branch = 3 # final_pred, sag_pred, cor_pred, axi_pred 
    input_image_indx = 2 # sag_img, cor_img, axi_img, t2_img, t1_img
    # slice_indx = 112
    for task in range(1):
        target_task = task
        for i in range(0, 220, 10):
            slice_indx = i
            model = ModelWrapper(model=net, input=images, branch=output_branch, img_indx = input_image_indx)
            target_layer = [model.target_layer]
            targets = [Output_Target(target_task)]
            reshape_func = ReshapeTransform(slice_indx)
            cam = EigenCAM(model=model, target_layers=target_layer, use_cuda=True, reshape_transform=reshape_func)
            

            input_image = images[input_image_indx][:, slice_indx:slice_indx+1] # c, d, h, w -> 1, 1, h, w
            # input_image = images[input_image_indx][slice_indx:slice_indx+1] # 1, 3, h, w
            cam_map = cam(input_tensor=input_image, targets=targets)
            # input_image = input_image[:, 0:1]
            grayscale_cam = cam_map[0, :]
            visualization = show_cam_on_image(np.stack([input_image[0][0].numpy()]*3, axis=-1), grayscale_cam, use_rgb=True)
            # img = visualization.transpose(2, 0, 1)
            print("output slice %d"%slice_indx)
            plt.imsave("task_%d_slice_%d.png"%(task, slice_indx), visualization)
            Show_Samples(input_image, title="task_%d_slice_%d_origin.png"%(task, slice_indx), save_path='./')
        # cam_map.shape()

def val_with_save(net, test_dataset):
    log_root_folder = args.log_folder
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

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, num_workers=16, shuffle=False)

    if args.half:
        net.half()
    net.eval()

    if torch.cuda.is_available():
        net = net.cuda()

    t_start = time.time()

    with torch.no_grad():
        test_loss, test_auc, test_acc = evaluate_model(net, test_loader, 1, args, mode="Test", save_path=log_root_folder)

    t_end = time.time()
    delta = t_end - t_start

    logging.info(
        "loss {0} | auc {1} | acc {2}elapsed time {3} s".format(
            test_loss, test_auc, test_acc, delta))

    logging.info('-' * 50)



