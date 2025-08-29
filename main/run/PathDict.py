'''

This file is used to define the path of the data and the cache file

Write dataset path in this format
DatasetName = {
    "train_label"   :   "",
    "train_path"    :   "",
    "val_label"     :   "",
    "val_path"      :   "",
    "test_label"    :   "",
    "test_path"     :   "",
    "cache_path"    :   "",
    "center_file"   :   "",
    "doctor_file"   :   "",
    "modal"         :   ["",]
}

RootPath        = ""
CodePath        = ""
DocEvlPath      = ""
ExpFolder       = ""
pretrain_folder = ""

----------------------------------

A Sample is provided here:

Internal = {
    "train_label"   :   os.path.join(dataroot, "train.csv"), 
    "train_path"    :   os.path.join(dataroot, "data/"), 
    "val_label"     :   os.path.join(dataroot, "valid.csv"),
    "val_path"      :   os.path.join(dataroot, "data/"),
    "test_label"    :   os.path.join(dataroot, "test.csv"),
    "test_path"     :   os.path.join(dataroot, "data/"),
    "cache_path"    :   os.path.join(cache_root, "Internal"),
    "center_file"   :   os.path.join(dataroot, "center.json"), 
    "doctor_file"   :   os.path.join(dataroot, "doc_eval.csv"), 
    "modal"         :   ["sag PDW","cor PDW","axi PDW","sag T2WI","cor T1WI"] 
}

---------------------------
The file structure of dataroot dir:

.
├── train.csv
├── valid.csv
├── test.csv
├── center.json
├── doc_eval.csv
├── data
│   ├── MR00001
│   │   ├── axi PDW
│   │   │  ├─ 001.dcm
│   │   │  ├─ 002.dcm
│   │   │  ├─ 003.dcm
│   │   │  ├─ ...
│   │   ├── cor PDW
│   │   │  ├─ 001.dcm
│   │   │  ├─ ...
│   │   ├── cor T1WI
│   │   │  ├─ 001.dcm
│   │   │  ├─ ...
│   │   ├── sag PDW
│   │   │  ├─ 001.dcm
│   │   │  ├─ ...
│   │   ├── sag T2WI
│   │   │  ├─ 001.dcm
│   │   │  ├─ ...
│   ├── MR00002
│   ├── MR00003
│   ├── ...
└── 





'''

