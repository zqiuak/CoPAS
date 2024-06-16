import pdb
import numpy as np
import os, csv
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import pdb
from tqdm import tqdm
import SimpleITK as sitk
import json
import math

from run.Args import args as Kargs
from run.PathDict import dataset_dict

INPUT_DIM = Kargs.INPUT_DIM
MAX_PIXEL_VAL = Kargs.MAX_PIXEL_VAL
MEAN = Kargs.MEAN
STDDEV = Kargs.STDDEV
IMG_R = Kargs.IMG_R
Patch_R = Kargs.Patch_R


def readLabels(labelPath):
    import pandas as pd
    sampleIDWithLabel = []
    labelData = pd.read_csv(labelPath)
    for i in range(len(labelData)):
        d = list(labelData.iloc[i, :])
        sampleIDWithLabel.append(d)
    return sampleIDWithLabel

class kneeDataSetSITK(data.Dataset):
    def __init__(self, mode="train", dataset_name = 'Internal', transform:bool = False, rebuild = False, aug_rate = 1, use_cache = True, args = Kargs):
        self.mode = mode
        assert dataset_name in args.DatasetNameList, 'invalid dataset name'
        self.dataset_name = dataset_name
        self.karg = args
        self.rebuild = rebuild
        self.className = args.DiseaseList
        self.centerCrop = args.Center_Crop
        self.transform = transform
        assert aug_rate>=1, 'aug_rate must larger than 0'
        self.aug_rate = aug_rate
        self.use_cache = use_cache
        self.cache_file = {}

        self.list_org = self.create_cls_dataset()
        self.org_len = len(self.list_org)
        self.aug_indx_map = list(range(self.org_len))
        self.list = self.list_org
        self.len = self.org_len

    def create_cls_dataset(self):
        mode = self.mode
        datadict = dataset_dict[self.dataset_name]
        try:
            self.label_path = datadict['%s_label'%mode]
            self.data_path = datadict['%s_path'%mode]
        except:
            raise Exception("There is no such dataset name: %s, mode: %s"%(self.dataset_name, mode))
        dataset_modals = datadict["modal"]
        self.modal_list = dataset_modals
        # read ROI d
        if not self.centerCrop:
            try:
                self.ROI_path = datadict['center_file']
                with open(self.ROI_path, 'r') as f:
                    self.centerROI = json.load(f)
            except:
                print("No ROI file found, use center crop")
                self.centerCrop = True
        # data cache
        self.cache_path = datadict['cache_path']
        if not self.rebuild:
            try:
                with open(os.path.join(self.cache_path, "datalist_cache_%s.pk"%mode), "rb") as infile:
                    self.miss_file, data_list = pickle.load(infile)
                return data_list
            except:
                print("No datalist cache found, rebuilding...")
        # build data list
        data_list = []
        self.miss_file = []
        labels = readLabels(self.label_path)
        tbar = tqdm(labels)
        for label in tbar:
            complete_flag = True
            id = str(label[0])
            pathlist = []
            for modal in self.karg.SequenceList:
                if modal in dataset_modals:
                    tmppath = os.path.join(self.data_path, id, modal)
                    if os.path.exists(tmppath) and len(os.listdir(tmppath))>0:
                        pathlist.append(tmppath)
                    else:
                        self.miss_file.append([id, label, modal])
                        complete_flag = False
                else:
                    pathlist.append('')
            try:
                class_labellist = [int(x) for x in label[1:]]
            except:
                complete_flag = False
                print("Error lable in %s"%id)
                continue
            if complete_flag:
                data_list.append([pathlist, *class_labellist, id])
        # data list cache
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        with open(os.path.join(self.cache_path, "datalist_cache_%s.pk"%self.mode), "wb+") as outfile:
                pickle.dump((self.miss_file, data_list), outfile)
        return data_list

    def dataset_status(self):
        print("Analysing dataset statistics")
        class_statis = [0]*len(self.className)
        missing_file = []
        f = open("./datastat.csv", "w+")
        outfile = csv.writer(f)
        Buffer=[]
        Buffer.append(["Name", "dx", "dy", "dz", "x", "y", "z", "maxi", "mini"])
        tbar = tqdm(self.list_org)
        for each in tbar:
            imgPaths, label, name = each[0], each[1:-1], each[-1]
            for imgPath in imgPaths:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(imgPath)
                reader.SetFileNames(dicom_names)
                sitk_img = reader.Execute() 
                dx, dy, dz = sitk_img.GetSpacing()
                x, y, z = sitk_img.GetSize()
                filter = sitk.MinimumMaximumImageFilter()
                filter.Execute(sitk_img)
                maxi = filter.GetMaximum()
                mini = filter.GetMinimum()
                Buffer.append([imgPath, dx, dy, dz, x, y, z, maxi, mini])
        outfile.writerows(Buffer)
        f.close()

    def load_vols(self, index):
        imgPaths, label, name = self.list_org[index][0], self.list_org[index][1:-1], self.list_org[index][-1]

        try:
            vols = []
            for imgPath in imgPaths:
                if imgPath != '':
                    if os.path.exists(imgPath):
                        vol = self.sitkReadZoom(imgPath)
                        # print(vol.shape)
                        vols.append(vol)
                    else:
                        raise Exception(imgPath + ' read img error')
                else:
                    vols.append(None)
        except:
            raise Exception(imgPath + ' read img error')
        vols_new = self.crop_vols(name, vols)
        data = (vols_new, label, name)
        self.cache_file[index] = data
        return data

    def sitkReadZoom(self, path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        sitk_img = reader.Execute()

        # unify spacing
        ndx, ndy, ndz = self.karg.Spacing
        dx, dy, dz = sitk_img.GetSpacing()
        x, y, z = sitk_img.GetSize()
        nx, ny, nz = [math.ceil(x*dx/ndx), math.ceil(y*dy/ndy), math.ceil(z*dz/ndz)]
        # print("Image %s, original size: %d, %d, %d, spacing %3f, %3f, %3f"%(path, x, y, z, dx, dy, dz))
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing((ndx, ndy, ndz))
        resample.SetSize((nx, ny, nz))
        resample.SetOutputOrigin(sitk_img.GetOrigin())
        resample.SetOutputDirection(sitk_img.GetDirection())
        newImage = resample.Execute(sitk_img)
        out_array = sitk.GetArrayFromImage(newImage)
        out_array = out_array/out_array.max()

        # CropResize
        SliceNum = self.karg.SliceNum
        xl, xr = (nx//2-Patch_R//2, nx//2+Patch_R//2)
        yl, yr = (ny//2-Patch_R//2, ny//2+Patch_R//2)
        zl, zr = (nz//2-SliceNum//2, nz//2+SliceNum//2)
        pad_x = abs(xl) if xl<0 else 0
        pad_y = abs(yl) if yl<0 else 0
        pad_z = abs(zl) if zl<0 else 0
        out_array = np.pad(out_array, ((pad_z,pad_z),(pad_x,pad_x),(pad_y,pad_y)), mode="constant")
        out_array = out_array[zl+pad_z:zr+pad_z, xl+pad_x:xr+pad_x, yl+pad_y:yr+pad_y]
        out_tensor = torch.FloatTensor(out_array)
        return out_tensor

    def whiting(self, img):
        img2 = (img - MEAN) / STDDEV
        img2 = ((img2 - np.min(img2)) / (np.max(img2) - np.min(img2)) * MAX_PIXEL_VAL)
        return img2.astype(np.uint8)

    def crop_vols(self, name, vols):
        # '''
        # ROI Crop
        if self.centerCrop:
            return vols
        try:
            roi = self.centerROI[f'segPDW_{name}']
            vols_new = []
            for i, vol in enumerate(vols):
                vol = torch.transpose(vol, [1, 2, 0])
                _cx, _cy, _cz, X, Y, Z = roi

                if i in [0, 3]:
                    _x, _y, _z = vol.shape
                    vol = vol[_cy - int(Patch_R/2):_cy + int(Patch_R/2), _cx -
                                int(Patch_R/2):_cx + int(Patch_R/2), :]
                if i in [1, 2]:
                    _y, _z, _x = vol.shape
                    _rcz, _rcy = (_cz - int(Patch_R/2),
                                    _cz + int(Patch_R/2)), (_cy - int(Patch_R/2),
                                                        _cy + int(Patch_R/2))
                    if _rcz[0] < 0:
                        _rcz = (_z // 2 - (Patch_R//2), _z // 2 + (Patch_R//2))
                    if _rcy[0] < 0:
                        _rcy = (_y // 2 - (Patch_R//2), _y // 2 + (Patch_R//2))
                    vol = vol[_rcy[0]:_rcy[1], _rcz[0]:_rcz[1], :]
                if i == 4:
                    _z, _x, _y = vol.shape
                    _rcz, _rcx = (_cz - int(Patch_R/2),
                                    _cz + int(Patch_R/2)), (_cx - int(Patch_R/2),
                                                        _cx + int(Patch_R/2))
                    if _rcz[0] < 0:
                        _rcz = (_z // 2 - (Patch_R//2), _z // 2 + (Patch_R//2))
                    if _rcx[0] < 0:
                        _rcx = (_x // 2 - (Patch_R//2), _x // 2 + (Patch_R//2))
                    vol = vol[_rcz[0]:_rcz[1], _rcx[0]:_rcx[1], :]
                if 0 in vol.shape:
                    print('shape error in ', name)
                    pdb.set_trace()
                vol = torch.transpose(vol, (2,0,1))
                vols_new.append(vol)
        except:
            print('roi error in %s, using centerCrop for this case'%name)
            self.centerCrop = True
            return
        
        return vols_new

    def cache_save(self, overwrite = False):
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        data = []
        print('Saving Cache for mode %s'%self.mode)
        tbar = tqdm(self.list_org)
        for i, _ in enumerate(tbar):
            cachefile = os.path.join(self.cache_path, 'cache_%s_%d.pk'%(self.mode,i))
            if os.path.exists(cachefile) and not overwrite:
                continue
            data = self.load_vols(i)
            with open(cachefile, 'wb+') as outfile:
                pickle.dump(data, outfile)

    def balance_cls(self, cls_indx):
        if cls_indx == -1:
            return
        pos_number = self.karg.ClassDistr[cls_indx+1]
        nag_number = self.org_len - pos_number
        pos_list = []
        nag_list = []
        for i, each in enumerate(self.list_org):
            class_label = each[cls_indx+1]
            if class_label:
                pos_list.append(i)
            else:
                nag_list.append(i)
        if pos_number > nag_number:
            aug_list = np.random.choice(nag_list, pos_number-nag_number)
        else:
            aug_list = np.random.choice(pos_list, nag_number-pos_number)
        self.aug_indx_map = list(range(self.org_len))
        self.aug_indx_map.extend(aug_list)
        self.len = len(self.aug_indx_map)

    def __len__(self):
        return self.len

    def __load_cache__(self, i = 0):
        assert self.use_cache == True
        if i in self.cache_file.keys():
            return self.cache_file[i]
        else:
            with open(os.path.join(self.cache_path, 'cache_%s_%d.pk'%(self.mode,i)), "rb") as infile:
                data = pickle.load(infile)
            self.cache_file[i] = data
            return data
    
    def __reshape_input__(self, vol):
        '''
        vol: (slice, h, w)
        '''
        vol = vol.unsqueeze(0).unsqueeze(0)  # vol->(1, 1, slice, h, w)
        r=self.karg.INPUT_DIM
        if self.karg.Keep_slice:
            vol = torch.nn.functional.interpolate(vol, (self.karg.SliceNum, r, r), mode='trilinear')
        else:
            vol = torch.nn.functional.interpolate(vol, (r, r, r), mode='trilinear')
        return vol[0]

    def __transform__(self, seed, vol):
        slice, h, w = vol.shape
        crop_top = int(seed[0]*20)
        crop_left = int(seed[1]*20)
        crop_height = int(h-crop_top-seed[2]*10)
        crop_width = int(w-crop_left-seed[2]*10)
        if seed[3]>0.5:
            vol = TF.resized_crop(vol, top=crop_top, left=crop_left, height=crop_height, width=crop_width, size=[h, w], antialias=True)
        if seed[4]>0.5:
            vol = TF.rotate(vol, angle=(seed[5]-0.5)*20, fill=0)
        if seed[6]>0.5:
            vol = TF.adjust_contrast(vol.unsqueeze(1), contrast_factor=seed[7]*0.5+0.75).squeeze(1) #->slice, 1, h, w->slice, h, w
        # if seed[8]>0.5:
            vol = TF.adjust_brightness(vol.unsqueeze(1), brightness_factor=seed[9]*0.5+0.75).squeeze(1)
        if seed[9]>0.5:
            vol = TF.affine(vol,angle=0, translate=[0, 0], shear=(seed[10]-0.5)*10, scale=1, fill=0)
        return vol

    def __getitem__(self, aug_index):
        index = self.aug_indx_map[aug_index]
        try:
            vols, label, name = self.cache_file[index]
        except:
            if self.use_cache:
                try: 
                    vols, label, name = self.__load_cache__(index)
                except:
                    print("load cache fail in %d"%index)
                    vols, label, name = self.load_vols(index)
            else:
                vols, label, name = self.load_vols(index)

        vols_new = []
        if self.transform:
            transform_seed = torch.rand(20).tolist()
        for i, vol in enumerate(vols):
            if vol==None:
                vols_new.append(torch.tensor([]))
                continue
            
            # transform
            if self.transform:
                vol = self.__transform__(transform_seed, vol)
                
            vol = self.__reshape_input__(vol) #->(1, slice, h, w)
            
            vols_new.append(vol)
        
        label_tensor = torch.FloatTensor(label)
        return vols_new, label_tensor, name
    



train_ds = kneeDataSetSITK('train', dataset_name='Internal', transform=Kargs.Augmentor, aug_rate=Kargs.augrate, use_cache=Kargs.use_cache, args=Kargs)
val_ds = kneeDataSetSITK('val', dataset_name='Internal', transform=False, use_cache=Kargs.use_cache, args=Kargs)
test_ds_dict = {dsname:kneeDataSetSITK('test', dataset_name=dsname, transform=False, use_cache=Kargs.use_cache, args=Kargs) for dsname in Kargs.DatasetNameList}