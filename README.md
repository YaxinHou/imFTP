# Deep Imbalance Learning via Fuzzy Transition and Prototypical Learning (imFTP, Information Sciences 2024)

This repository contains the Pytorch implementations of the paper submitted to Information Sciences 2024:

Yaxin Hou, Weiping Ding, Chongsheng Zhang. Deep Imbalance Learning via Fuzzy Transition and Prototypical Learning. Information Sciences 2024. &nbsp; &nbsp;[Paper]()

## Introduction
 
This work (imFTP) aims to 

![image](model/model.png)

Abstract: 

## Prerequisites

All codes are written by Python 3.8 with:

* Operating System: Windows 10
* torch              1.13.0
* torchaudio         0.13.0
* torchvision        0.14.0
* pandas             1.5.2
* scikit-learn       1.1.3
* imbalanced-learn   0.9.1
* numpy              1.23.5
* openpyxl           3.0.10

## Code structure description

```
   imFTP
   ├──data
   │   ├──original_data  
   │   └──spilted_data
   │
   ├──model
   │   ├──model.png
   │   └──model.py
   │
   ├──result
   │
   ├──trained_model
   │   
   ├──utils
   │   ├──log.py
   │   ├──dataset.py
   │   ├──split_data.py
   │   └──transformer.py 
   │
   ├──imFTP_TRAIN.py
   ├──imFTP_TEST.py
   └──README.MD
```   
  
## Train
  
  To train a classifier for class-imbalanced data:

   ```
   python imFTP_TRAIN.py --dataset mfcc
   ```
  
## Test
 
  To test the classifier with the trained model:

  ```
   python imFTP_TEST.py --dataset mfcc
   ```

## Our Trained models
Under the folder “trained_model”, we have uploaded our trained models for the mfcc dataset.

*  Classification model is at `./trained_model/`

## Citation

If you find our method useful, please consider citing our paper:

  ```
  @inproceedings{imFTP2023,
    title={Deep Imbalance Learning via Fuzzy Transition and Prototypical Learning},
    author={Yaxin Hou and Weiping Dingand Chongsheng Zhang},
    booktitle={Pattern Recognition},
    year={2024},
  }
  ```
