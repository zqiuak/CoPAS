# CoPAS: learning Co-Plane Attention across MRI Sequences for diagnosing twelve types of knee abnormalities: A multi-center retrospective study

[![Python](https://img.shields.io/badge/Python-3.8.0-blue)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://opensource.org/licenses/Apache-2.0)
<!-- [![DOI]()]()-->


## Introduction
This is the official repository of 

**Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities** 

by
*Zelin Qiu, Zhuoyao Xie, Yanwen Li, Huangjing Lin, Qiang Ye, Menghong Wang, Shisi Li, Yinghua Zhao, and Hao Chen*

## Installation Guide:
The code is based on Python 3.8.0

1. Download the repository
```bash
git clone https://github.com/zqiuak/CoPAS
```

2. Go to the `main` folder and install requested libarary.
```bash
cd main
pip install -r requirements.txt
```
Typically, it will take few minutes to complete the installation.


## Run
1. Fill the data path in ```DataDict.json```, the sample is given in the file.
2. Change parameters in ```Args.py``` to fit your data.
#### Run the following command for training:
```bash
python run.py
```

#### Run the following command for testing:
```bash
python run.py --test --weight_path PATH_TO_WEIGHT
```

#### Other useful command line arguments:
```--epochs```: Maximum number of epoches in training.<br>
```--batch_size```: Batch size.<br>
```--lr```: Initial learning rate.<br>
```--gpu```: GPU card number.<br>
```--augment```: ```bool```, use augmentation or not.<br>


We have prepared 50 sample data for test, click [here](https://drive.google.com/drive/folders/1b7H4zIppkeU2YGFSTMbIKFSDvhNtEMNY?usp=sharing) to download.

If you have any special requests, please send a email to Zelin Qiu (zqiuak@connect.ust.hk).

## License & Citation

This project is covered under the **Apache 2.0 License**.

If you find this work useful, please cite our paper:

```
@article{qiu2024learning,
  title={Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities},
  author={Qiu, Zelin and Xie, Zhuoyao and Lin, Huangjing and Li, Yanwen and Ye, Qiang and Wang, Menghong and Li, Shisi and Zhao, Yinghua and Chen, Hao},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={7637},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
