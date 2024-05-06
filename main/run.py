import os
import torch
from train import parse_arguments, run
from model import Multi_view_Knee
from dataloader import kneeDataSetSITK
from val_with_save import val_with_save

if __name__ == "__main__":
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.test:
        assert args.weight_path != "", "Please specify the weight path"
        args.half = False
        model = Multi_view_Knee()
        model_file = args.weight_path
        model = torch.load(model_file)
        model = model.float()
        test_dataset = kneeDataSetSITK(mode="test", use_cache=True, args=args)
        # show_CAM(net=model, dataset=test_dataset)
        val_with_save(args, model)
    else:
        run(args)