print("initiating")
import sys
a = ["--gpu", "0", "--epoch", "100", "--batch_size", "20", "--lr", "5e-7"]
# a.extend(["--debug"])
sys.argv.extend(a)



import os
import torch

from run.Args import args
from run.train import run
from model.model import Multi_view_Knee
from data.dataloader import test_ds_dict
from val_with_save import val_with_save

print("running")
if args.test:
    assert args.weight_path != "", "Please specify the weight path"
    args.half = False
    model = Multi_view_Knee()
    model_file = args.weight_path
    model = torch.load(model_file)
    model = model.float()
    test_dataset = test_ds_dict['Internal']
    # show_CAM(net=model, dataset=test_dataset)
    val_with_save(model)
else:
    run()