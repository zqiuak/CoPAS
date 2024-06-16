from asyncio import transports
import pdb
import os
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from focal_loss.focal_loss import FocalLoss as FL

from run.utils import Show_Samples
from run.Args import args
from model.ResNet3D import generate_model as resnet3D
  
def ini_weights(module_list:list):
    for m in module_list:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Res_3D_Encoder(nn.Module):
    def __init__(self, kargs = args, **kwargs) -> None:
        super().__init__()
        layer = kargs.model_depth
        if layer == 50:
            self.model = resnet3D(kargs)
            self.feature_channel = 2048
        elif layer == 18:
            self.model = resnet3D(kargs)
            self.feature_channel = 512

    def forward(self, x, squeeze_to_vector = False, pool="max"):
        # input: b, c, d, h, w -> b, (d), c
        assert x.dim() == 5, 'Wrong input dimension'
        if pool == "avg":
            pool_func = F.adaptive_avg_pool3d
        else:
            pool_func = F.adaptive_max_pool3d
        x = self.model(x)
        if squeeze_to_vector:
            x = pool_func(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            x = x.transpose(1,2) # b, d, c, h, w
            x = pool_func(x, (self.feature_channel,1,1))
            x = torch.flatten(x, start_dim=2)
        return x

class Res_2D_Encoder(nn.Module):
    def __init__(self, kargs, *args,**kwargs) -> None:
        raise Exception("This module is deprecated")
        super().__init__()
        self.kargs = kargs
        if kargs.model_depth == 50:
            self.model = models.resnet50(weights="IMAGENET1K_V2")
            self.feature_channel = 2048
        elif kargs.model_depth == 18:
            self.model = models.resnet18(weights="IMAGENET1K_V1")
            self.feature_channel = 512
        
    def forward(self, input, squeeze_to_vector = False, pool="avg"):
        # x.shape = (1, 1, slice, h, w)
        assert input.dim() == 5, 'Wrong input dimension'
        assert input.shape[0]==1 and input.shape[1] == 1, "only support batchsize=1, but got shape"+str(input.shape) 
        x = torch.cat((input, input, input), dim=1).transpose(1, 2).squeeze(0) # slice, c, h, w
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.unsqueeze(0) # 1, slice, c, h, w
        if squeeze_to_vector:
            x = x.transpose(1,2) #1, c, d, h, w
            x = F.adaptive_max_pool3d(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            #-> 1, slice, c, 1
            x = F.adaptive_max_pool3d(x, (self.feature_channel,1,1))
            x = torch.flatten(x, start_dim=2)
        return x

class Eff_2D_Encoder(nn.Module):
    def __init__(self, kargs, *args, **kwargs) -> None:
        raise Exception("This module is deprecated")
        super().__init__(*args, **kwargs)
        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.feature_channel = 1280

    def forward(self, input, squeeze_to_vector = False, pool="avg"):
        # x.shape = (1, 1, slice, h, w)
        assert input.dim() == 5, 'Wrong input dimension'
        assert input.shape[0]==1 and input.shape[1] == 1, "only support batchsize=1, but got shape"+str(input.shape) 
        x = torch.cat((input, input, input), dim=1).transpose(1, 2).squeeze(0) # slice, c, h, w
        x = self.model.features(x)
        s, c, h, w = x.shape
        x = x.unsqueeze(0)
        if squeeze_to_vector:
            #->b, c, 1
            x = x.transpose(1,2)
            x = F.adaptive_max_pool3d(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            #-> b, slice, c
            x = F.adaptive_max_pool3d(x, (self.feature_channel,1,1))
            x = torch.flatten(x, start_dim=2)
        return x
        
class Co_Plane_Att(nn.Module):
    def __init__(self, embed_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb_dim = embed_dim
        self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        ini_weights(self.modules())

    def forward(self, main_f, co_f1, co_f2):
        # (batch, channel, d)
        res = main_f
        q = self.mq(main_f)
        k1 = self.mk1(co_f1).permute(0, 2, 1)
        k2 = self.mk2(co_f2).permute(0, 2, 1)
        v1 = self.mv1(co_f1)
        v2 = self.mv2(co_f2)
        att1 = torch.matmul(q, k1)/np.sqrt(self.emb_dim)
        att1 = torch.softmax(att1, -1)
        att2 = torch.matmul(q, k2)/np.sqrt(self.emb_dim)
        att2 = torch.softmax(att2, -1)
        out1 = torch.matmul(att1, v1)
        out2 = torch.matmul(att2, v2)
        self.attmap1 = att1.detach().cpu()
        self.attmap2 = att2.detach().cpu()
        f = self.norm(0.5*(out1+out2)+res)
        f = f.transpose(1, 2)
        f = F.adaptive_max_pool1d(f, 1)
        f = torch.flatten(f, start_dim=1)
        return f

class Cross_Modal_Att(nn.Module):
    def __init__(self, feature_channel, kargs = args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform_matrix = nn.Linear(2*feature_channel, feature_channel)
        self.norm = nn.BatchNorm1d(num_features=feature_channel)
        ini_weights([self.transform_matrix, self.norm])

    def forward(self, pdw_f, aux_f): #pdw feature and auxiliary modal 
        assert aux_f.dim() == pdw_f.dim()
        add_f = pdw_f+aux_f
        sub_f = torch.cat((pdw_f, aux_f), dim=1)
        att_f = self.transform_matrix(sub_f)
        att_f = torch.relu(att_f)
        att_f = torch.softmax(att_f, -1)
        f = add_f*att_f
        return f

class Branch_Classifier(nn.Module):
    def __init__(self, classnum, feature_channel, dropout_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classifiers = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(feature_channel, classnum))
        ini_weights(self.classifiers)

    def forward(self, f):
        if f.dim() == 3:
            f = f.squeeze(2)
        return self.classifiers(f)

class Multi_view_Knee(nn.Module):
    def __init__(self, backbone = 'ResNet', encoder_layer = 18, pretrain = True, parallel_device = True, kargs = args) -> None:
        super().__init__()
        self.kargs = kargs
        self.class_num = kargs.ClassNum
        self.para_device = parallel_device
        self.branch = kargs.active_branch
        # set parallel devices
        if torch.cuda.device_count() == 3 and self.para_device:
            self.device_list = ["cuda:%d"%x for x in range(torch.cuda.device_count())]
        else:
            self.device_list = ["cuda:0"]*3
        self.dropout_rate = 0.05
        self.backbone = kargs.backbone
        self.module_list = []
        self.__make_encoder__(self.backbone)
        self.__make_co_plane_att__()
        self.__make_cross_modal_att__()
        self.__make_classifier__()
        self.mining_conv = nn.Conv3d(1, 12, (12,12,12))
        ini_weights([self.mining_conv, self.multi_view_classifier])
        model_param = sum(np.prod(v.size()) for name, v in self.named_parameters()) / 1e6
        print('model param = %f MB'%model_param)

        if sum(self.branch) == 3:
            try:
                self.save_or_load_encoder_para("load", kargs.pretrain_folder)
                print('loading pretrained success')
            except:
                print('loading pretrained failed, using random init')

    def __make_encoder__(self, backbone_name):
        if backbone_name == 'ResNet':
            encoder_func = Res_2D_Encoder
        elif backbone_name == 'ResNet3D':
            encoder_func = Res_3D_Encoder
        elif backbone_name == "EffNet":
            encoder_func = Eff_2D_Encoder
        else:
            raise Exception('Wrong Backbone Name!')
        self.encoder_func = encoder_func
        if self.branch[0]:
            self.sag_enc = encoder_func(self.kargs).to(self.device_list[0])
            if not self.kargs.no_cross_modal:
                self.t2w_enc = encoder_func(self.kargs).to(self.device_list[0])
        if self.branch[1]:
            self.cor_enc = encoder_func(self.kargs).to(self.device_list[1])
            if not self.kargs.no_cross_modal:
                self.t1w_enc = encoder_func(self.kargs).to(self.device_list[0])
        if self.branch[2]:
            self.axi_enc = encoder_func(self.kargs).to(self.device_list[2])
        return

    def __make_co_plane_att__(self):
        if self.kargs.no_co_att:
            return
        emb_dim = self.kargs.emb_dim
        if self.branch[0]:
            self.sag_att = Co_Plane_Att(emb_dim).to(self.device_list[0])
        if self.branch[1]:
            self.cor_att = Co_Plane_Att(emb_dim).to(self.device_list[1])
        if self.branch[2]:
            self.axi_att = Co_Plane_Att(emb_dim).to(self.device_list[2])
        return
    
    def __make_cross_modal_att__(self):
        if self.kargs.no_cross_modal:
            return
        if self.branch[0]:
            self.sag_cross_att = Cross_Modal_Att(self.sag_enc.feature_channel, self.kargs).to(self.device_list[0])
        if self.branch[1]:
            self.cor_cross_att = Cross_Modal_Att(self.cor_enc.feature_channel, self.kargs).to(self.device_list[1])
        return
    
    def __make_classifier__(self):
        if self.branch[0]:
            self.sag_classifier = Branch_Classifier(self.class_num, self.sag_enc.feature_channel, self.dropout_rate).to(self.device_list[0])
        if self.branch[1]:
            self.cor_classifier = Branch_Classifier(self.class_num, self.cor_enc.feature_channel, self.dropout_rate).to(self.device_list[1])
        if self.branch[2]:
            self.axi_classifier = Branch_Classifier(self.class_num, self.axi_enc.feature_channel, self.dropout_rate).to(self.device_list[2])
        
        if self.kargs.no_corr_mining:
            self.multi_view_classifier = nn.Sequential(nn.Linear(36, 12)).to(self.device_list[2])
        else:
            self.multi_view_classifier = nn.Sequential(nn.Linear(12, 12)).to(self.device_list[2])
        return

    def __sag_branch__(self, input, device = None):
        sag_img, cor_img, axi_img, t2_img, _ = input

        if self.kargs.no_co_att:
            pdw_f = self.sag_enc(sag_img, squeeze_to_vector = True)
        else:
            sag_cor = cor_img.transpose(2,4)
            sag_axi = torch.rot90(axi_img.transpose(2,4), k=1, dims=[3,4]).detach()
            # Show sample
            if self.kargs.show_patch_sample:
                for i in range(0, self.kargs.INPUT_DIM, 10):
                    Show_Samples(sag_img[0][0][i], title="sag_img_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(sag_cor[0][0][i], title="sag_cor_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(sag_axi[0][0][i], title="sag_axi_%d"%i, bcdhw=True, save_path="./Pic")        
                    Show_Samples(t2_img[0][0][i], title="sag_t1_%d"%i, bcdhw=True, save_path="./Pic")
            if device != None:
                sag_img = sag_img.to(device, non_blocking=True)
                sag_cor = sag_cor.to(device, non_blocking=True)
                sag_axi = sag_axi.to(device, non_blocking=True)
                t2_img = t2_img.to(device, non_blocking=True)

            main_f = self.sag_enc(sag_img)
            with torch.no_grad():
                co_f1 = self.sag_enc(sag_cor)
                co_f2 = self.sag_enc(sag_axi)
            pdw_f = self.sag_att(main_f, co_f1, co_f2)

        if self.kargs.no_cross_modal:
            pred = self.sag_classifier(pdw_f)
        else:
            aux_f = self.t2w_enc(t2_img, squeeze_to_vector = True)
            cross_m_f = self.sag_cross_att(pdw_f, aux_f)
            pred = self.sag_classifier(cross_m_f)
        return pred
    
    def __cor_branch__(self, input, device = None):
        sag_img, cor_img, axi_img, _, t1_img = input
        
        if self.kargs.no_co_att:
            pdw_f = self.cor_enc(cor_img, squeeze_to_vector = True)
        else:
            cor_sag = sag_img.transpose(2,4)
            cor_axi = torch.rot90(axi_img.transpose(2,3), k=2, dims=[3,4])
            # Show sample
            if self.kargs.show_patch_sample:
                for i in range(0, self.kargs.INPUT_DIM, 10):
                    Show_Samples(cor_img[0][0][i], title="cor_img_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(cor_sag[0][0][i], title="cor_sag_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(cor_axi[0][0][i], title="cor_axi_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(t1_img[0][0][i], title="cor_t2_%d"%i, bcdhw=True, save_path="./Pic")
            if device != None:
                cor_img = cor_img.to(device, non_blocking=True)
                cor_sag = cor_sag.to(device, non_blocking=True)
                cor_axi = cor_axi.to(device, non_blocking=True)
                t1_img = t1_img.to(device, non_blocking=True)

            main_f = self.cor_enc(cor_img)
            with torch.no_grad():
                co_f1 = self.cor_enc(cor_sag)
                co_f2 = self.cor_enc(cor_axi)
            pdw_f = self.cor_att(main_f, co_f1, co_f2)

        if self.kargs.no_cross_modal:
            pred = self.cor_classifier(pdw_f)
        else:
            aux_f = self.t1w_enc(t1_img, squeeze_to_vector = True)
            cross_m_f = self.cor_cross_att(pdw_f, aux_f)
            pred = self.cor_classifier(cross_m_f)
        return pred
        
    def __axi_branch__(self, input, device = None):
        sag_img, cor_img, axi_img, _, _ = input

        if self.kargs.no_co_att:
            pdw_f = self.axi_enc(axi_img, squeeze_to_vector = True)
        else:
            axi_cor = cor_img.transpose(2, 3)
            axi_sag = sag_img.transpose(2, 4)

            if self.kargs.show_patch_sample:
                for i in range(0, self.kargs.INPUT_DIM, 10):
                    Show_Samples(axi_img[0][0][i], title="axi_img_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(axi_cor[0][0][i], title="axi_cor_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(axi_sag[0][0][i], title="axi_sag_%d"%i, bcdhw=True, save_path="./Pic")        

            if device != None:
                axi_img = axi_img.to(device, non_blocking=True)
                axi_cor = axi_cor.to(device, non_blocking=True)
                axi_sag = axi_sag.to(device, non_blocking=True)

            main_f = self.axi_enc(axi_img)
            with torch.no_grad():
                co_f1 = self.axi_enc(axi_cor)
                co_f2 = self.axi_enc(axi_sag)
            pdw_f = self.axi_att(main_f, co_f1, co_f2)

        pred = self.axi_classifier(pdw_f)
        return pred

    def __discovery__(self, sag_pred, cor_pred, axi_pred, device = None):
        with torch.no_grad():
            sag_pred = torch.sigmoid(sag_pred)
            cor_pred = torch.sigmoid(cor_pred)
            axi_pred = torch.sigmoid(axi_pred) 

        if self.kargs.separate_final:
            sag_pred = sag_pred.detach().clone()
            cor_pred = cor_pred.detach().clone()
            axi_pred = axi_pred.detach().clone()

        if self.kargs.no_corr_mining:
            # direct_fc
            pred_matrix = torch.cat((sag_pred, cor_pred, axi_pred), dim=1)
            return self.multi_view_classifier(pred_matrix)

        if device != None:
            sag_pred = sag_pred.to(device)
            cor_pred = cor_pred.to(device)
            axi_pred = axi_pred.to(device)

        union_prob = sag_pred*cor_pred*axi_pred
        sag_t = sag_pred.unsqueeze(2).unsqueeze(2)  # (b, 12, 1, 1)
        cor_t = cor_pred.unsqueeze(2).unsqueeze(1)  # (b, 1, 12, 1)
        axi_t = axi_pred.unsqueeze(1).unsqueeze(1)  # (b, 1, 1, 12)
        pred_matrix = (sag_t * cor_t * axi_t).unsqueeze(1)
        fin_att = torch.flatten(self.mining_conv(pred_matrix), start_dim=1)
        fin_pred = union_prob*fin_att
        return fin_pred

    def save_or_load_encoder_para(self, mode = "save", path = ""):
        if mode == "save":
            act_func = self.__save_para_
        elif mode == "load":
            act_func = self.__load_para_
        else:
            raise Exception("wrong mode name, should be save or load")
        
        if self.branch[0]:
            act_func(self.sag_enc, 'sag_enc', path = path)
            act_func(self.sag_classifier, "sag_cls", path=path)
            if not self.kargs.no_cross_modal:
                act_func(self.t2w_enc, 't2w_enc', path = path)
        if self.branch[1]:
            act_func(self.cor_enc, 'cor_enc', path = path)
            act_func(self.cor_classifier, "cor_cls", path=path)
            if not self.kargs.no_cross_modal:
                act_func(self.t1w_enc, 't1w_enc', path = path)
        if self.branch[2]:
            act_func(self.axi_enc, 'axi_enc', path = path)
            act_func(self.axi_classifier, "axi_cls", path=path)

    def att_map(self, input):
        if self.kargs.no_co_att:
            print("no co-plane attention enabled")
        attmap_dict = {}
        if self.branch[0]:
            attmap_dict["sag"] = [self.sag_att.attmap1, self.sag_att.attmap2]
        if self.branch[1]:
            attmap_dict["cor"] = [self.cor_att.attmap1, self.cor_att.attmap2]
        if self.branch[2]:
            attmap_dict["axi"] = [self.axi_att.attmap1, self.axi_att.attmap2]

    def __save_para_(self, model, name, path=""):
        if path != "":
            save_path = path
        else:
            print("save path not specified, not saving")
            return
        try:
            torch.save(model.state_dict(), os.path.join(save_path, "%s_%s_para.pkl"%(model.__class__.__name__,name)))
        except:
            print("save para failed")

    def __load_para_(self, model, name, path=""):
        if path == "":
            save_path = self.kargs.pretrain_folder
        else:
            save_path = path
        model.load_state_dict(torch.load(os.path.join(save_path, "%s_%s_para.pkl"%(model.__class__.__name__,name))))

    def forward(self, input):
        # input: [[bz, channel, slice, h, w], []..]
        if self.branch[0]:
            sag_pred = self.__sag_branch__(input)
            final_pred = sag_pred
        if self.branch[1]:
            cor_pred = self.__cor_branch__(input)
            final_pred = cor_pred
        if self.branch[2]:
            axi_pred = self.__axi_branch__(input)
            final_pred = axi_pred
        if sum(self.branch) == 1:
            return final_pred, final_pred, final_pred, final_pred
        if sum(self.branch) == 3:
            final_pred = self.__discovery__(sag_pred, cor_pred, axi_pred)
            return final_pred, sag_pred, cor_pred, axi_pred

    def criterion(self, pred, label, act_task = -1, final = False):
        task_weights = torch.ones((12))
        task_weights = task_weights.tolist()

        pos_weights = torch.tensor(self.kargs.pos_weights).cuda()

        final_pred, sag_pred, cor_pred, axi_pred = pred
        sag_loss = 0.0
        cor_loss = 0.0
        axi_loss = 0.0
        final_loss = 0.0
        lossfunc = F.binary_cross_entropy_with_logits
        lossfunc_final = Focal_Loss_with_logits

        for i in range(12):
            if i == act_task or act_task == -1:
                task_weight = task_weights[i]                
            else:
                task_weight = 0.01
            pos_wei = pos_weights[i]
            subject_label = label[:, i:i + 1]

            if self.branch[0]:
                sag_loss += task_weight* lossfunc(sag_pred[:, i:i + 1], subject_label, pos_weight=pos_wei)
            if self.branch[1]:
                cor_loss += task_weight* lossfunc(cor_pred[:, i:i + 1], subject_label, pos_weight=pos_wei)
            if self.branch[2]:
                axi_loss += task_weight* lossfunc(axi_pred[:, i:i + 1], subject_label, pos_weight=pos_wei)
            if final:
                final_loss += task_weight* lossfunc_final(final_pred[:, i:i + 1], subject_label, pos_weight=pos_wei)

        if sum(self.branch) == 1:
            loss = [sag_loss, cor_loss, axi_loss][self.branch.index(1)]
        elif final:
            loss= self.kargs.alpha*(sag_loss+cor_loss+axi_loss) + final_loss 
        else:
            loss = sag_loss+cor_loss+axi_loss

        return loss, loss.item()

class Pretrain_Encoder(nn.Module):
    def __init__(self, backbone = 'ResNet', encoder_layer = 18, pretrain = True, parallel_device = '1', kargs = args) -> None:
        super().__init__()
        self.kargs = kargs
        self.encoder = Res_3D_Encoder(kargs)
        self.classifier = Branch_Classifier(12, self.encoder.feature_channel, 0.05)
        plane = "sag"
        self.encoder_name = "%s_enc"%plane
        self.classifier_name = "%s_cls"%plane

    def forward(self, input):
        # input: [[bz, slice, channel, h, w], []..]
        sag_img, cor_img, axi_img, t2_img, t1_img = input
        x = sag_img
        f = self.encoder(x, squeeze_to_vector = True, pool="max")
        pred = self.classifier(f)
        return [pred]*4

    def save_or_load_encoder_para(self, mode = "save", path = ""):

        if mode == "load":
            self.load_state_dict(torch.load(os.path.join(path, "%s_%s_para.pkl"%(self.__class__.__name__,self.encoder_name))))
        elif mode == "save":
            if path != "":
                save_path = path
            else:
                print("save path not specified, not saving")
                return
            try:
                torch.save(self.encoder.state_dict(), os.path.join(save_path, "%s_%s_para.pkl"%(self.encoder.__class__.__name__,self.encoder_name)))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, "%s_%s_para.pkl"%(self.classifier.__class__.__name__,self.classifier_name)))
            except:
                print("save para failed")


    def criterion(self, pred, label, act_task = -1, final = False):
        lossfunc = F.binary_cross_entropy_with_logits
        final_pred, sag_pred, cor_pred, axi_pred = pred
        loss = lossfunc(final_pred, label)
        # pos_weights = torch.tensor(self.kargs.pos_weights).cuda()
        # for i in range(12):
        #     if i==act_task or act_task==-1:
        #         weight = 1.0
        #     else:
        #         weight = 0.1
        #     loss = loss + weight*lossfunc(final_pred[:,i:i+1], label[:,i:i+1], pos_weight = pos_weights[i])

        return loss, loss.item()

def Focal_Loss_with_logits( pred, label, pos_weight = None, gamma=2, reduction='mean'):
    sigpred = torch.sigmoid(pred)
    pred = torch.cat((1-sigpred, sigpred), dim=1)
    if pos_weight is not None:
        weight = torch.stack((1/(1+pos_weight), pos_weight/(1+pos_weight)))
    else:
        weight = None
    label = label.long()
    fl = FL(gamma=gamma, weights=weight, reduction="none",eps=5e-6)
    loss = fl(pred, label)
    if reduction == "mean":
        loss = loss.sum()/loss.shape[0]
    elif reduction == "sum":
        loss = loss.sum()
    return loss

