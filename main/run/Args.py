import os, argparse, time


from run.PathDict import dataroot, CodePath, ExpFolder, pretrain_folder, dataset_dict


class Arguments():
    def __init__(self):

        #path_args
        self.RootPath = dataroot
        self.CodePath = CodePath
        self.DocEvlPath =  ""
        self.ExpFolder = ExpFolder
        self.pretrain_folder = pretrain_folder
        if not os.path.exists(self.ExpFolder):
            os.makedirs(self.ExpFolder)
        
        # task_args
        self.DatasetNameList = [k for k in dataset_dict.keys()]
        self.DiseaseList = ['MENI', 'ACL', 'CART', 'PCL', 'MCL', 'LCL', 'EFFU', 'CONT', 'PLICA', 'CYST', 'IFP', 'PR']
        self.ViewList = ['Sag', 'Cor', 'Axi']
        self.SequenceList = ["sag PDW","cor PDW","axi PDW","sag T2WI","cor T1WI"]
        self.ClassNum = len(self.DiseaseList)

        # data_args        
        self.INPUT_DIM = 224 # resolution of model input
        self.MAX_PIXEL_VAL = 255
        self.MEAN = 58.09
        self.STDDEV = 49.73
        self.IMG_R = 576 # origin image resolution
        self.Spacing = (0.3, 0.3, 3.8)
        self.SliceNum = 24
        self.Patch_R = 448# patch resolution
        self.Center_Crop = True
        self.ClassDistr = [771, 563, 278, 319, 114, 148, 114, 703, 287, 488, 146, 305, 80] # [total, cls1, cls2...]
        self.cal_class_weight()
        self.Keep_slice = False

        # model args
        self.backbone = "ResNet3D"
        self.pretrain = False
        self.model_depth = 18
        self.pretrain_path = os.path.join(self.pretrain_folder, "resnet_%d_23dataset.pth"%self.model_depth)
        self.emb_dim = {"ResNet":512, "ResNet3D":512, "EffNet":1280}[self.backbone]
        self.emb_num = {"ResNet":24, "ResNet3D":28, "EffNet":24}[self.backbone]
        self.alpha = 0.1
        
        self.no_co_att = False
        self.no_cross_modal = False
        self.no_corr_mining = False   
        self.separate_final = False
        self.active_class = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.active_branch = [1, 1, 1]
        # self.active_class = [1]*12

        # train args
        self.gpu = '0'
        self.lr = 5e-5
        self.Augmentor = True
        self.augrate = 4
        self.use_cache = True
        self.num_workers = 7
        self.iters_to_accumulate = 1 # accumulate gradient for how many batches
        self.data_balance = False
        self.debug = False

        # analysis args
        self.show_patch_sample = False


        # else
        # self.attlist = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not attr.startswith("__")]
        self.transfer_from_argparse()

        self.log_folder = self.ExpFolder+"/Exp-{}-{}-{}/".format(self.prefix_name,time.strftime("%m%d-%H%M%S"), self.a)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        if self.debug:
            self.set_debug()


    def cal_class_weight(self):
        self.pos_weights = []
        total = self.ClassDistr[0]
        for pos_num in self.ClassDistr[1:]:
            self.pos_weights.append((total-pos_num)/pos_num)
        

    def transfer_from_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--weight_path', type=str, default='')
        parser.add_argument('--prefix_name', type=str, default=self.backbone)
        parser.add_argument('--a',type=str,default='')
        parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--lr', type=float, default=self.lr)
        parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
        parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
        parser.add_argument('--patience', type=int, default=500)
        parser.add_argument('--log_every', type=int, default=100)
        parser.add_argument('--gpu',type=str, default=self.gpu)
        parser.add_argument('--half', action='store_true', default=False)
        parser.add_argument('--debug', action='store_true', default=self.debug)

        # args = parser.parse_args()
        args, unknown = parser.parse_known_args() # compatable with jupyter notebook
        argslist = [attr for attr in dir(args) if not callable(getattr(args,attr)) and not attr.startswith("__")]
        for each in argslist:
            setattr(self, each, getattr(args, each))


    def set_debug(self):
            self.epochs = 1
            self.num_workers = 2
            self.prefix_name = "Debug"


args = Arguments()