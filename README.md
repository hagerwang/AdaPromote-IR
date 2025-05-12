# AdaPromote-IR

Adaptive Learning to Perceive Degradation Semantic and Prompting for All-in-One Image Restoration.

This is a Pytorch implementation of [AdaPromote-IR](https://github.com/hagerwang/AdaPromote-IR/).
![image](https://github.com/hagerwang/AdaPromote-IR/blob/main/framework.png)  

## Prerequisites 

1. Install the repository

```
conda env create -f env.yml
```

2. Dataset preparation
```
    |--data   
    |    |--Train
    |    |    |--Deblur
    |    |    |    |--blur
    |    |    |    |    |--GOPR0372_07_00_000047.png
    |    |    |    |     ...
    |    |    |    |--sharp
    |    |    |    |    |--GOPR0372_07_00_000047.png
    |    |    |    |     ...
    |    |    |--Dehaze
    |    |    |    |--original
    |    |    |    |    |--0025.png
    |    |    |    |     ...
    |    |    |    |--synthetic
    |    |    |    |    |--part1
    |    |    |    |    |    |--0025_0.8_0.1.jpg
    |    |    |    |    |    ...
    |    |    |    |    |--part2
    |    |    |    |    |    |--3068_0.8_0.1.jpg
    |    |    |    |    |    ...    
    |    |    |    |    |--part3
    |    |    |    |    |    |--5425_0.8_0.1.jpg
    |    |    |    |    |    ...   
    |    |    |    |    |--part4
    |    |    |    |    |    |--6823_0.8_0.1.jpg
    |    |    |    |    |    ...
    |    |    |--Denoise
    |    |    |    |--00001.bmp
    |    |    |    ...
    |    |    |    |--5096.jpg
    |    |    |    ...
    |    |    |--Derain
    |    |    |    |--gt
    |    |    |    |    |--norain-1.png
    |    |    |    |     ...
    |    |    |    |--rainy
    |    |    |    |    |--rain-1.png
    |    |    |    |     ...
    |    |    |--Enhance
    |    |    |    |--gt
    |    |    |    |    |--2.png
    |    |    |    |     ...
    |    |    |    |--low
    |    |    |    |    |--2.png
    |    |    |    |     ...
    |    |--test
    |    |    |--deblur
    |    |    |    |--gopro
    |    |    |    |    |--input
    |    |    |    |    |   |--GOPR0384_11_00_000001.png
    |    |    |    |    |   ...
    |    |    |    |    |--target
    |    |    |    |    |   |--GOPR0384_11_00_000001.png
    |    |    |    |    |   ...
    |    |    |--dehaze
    |    |    |    |--input
    |    |    |    |    |--0001_0.8_0.2.jpg
    |    |    |    |    ...
    |    |    |    |--target
    |    |    |    |   |--0001.png
    |    |    |    |   ...
    |    |    |--denoise
    |    |    |    |--bsd68
    |    |    |    |    |--3096.jpg
    |    |    |    |    ...
    |    |    |    |--urban100
    |    |    |    |    |--img_001.png
    |    |    |    |    ...
    |    |    |    |--kodak24
    |    |    |    |    |--kodim01.png
    |    |    |    |    ...
    |    |    |--derain
    |    |    |    |--Rain100L
    |    |    |    |    |--input
    |    |    |    |    |   |--1.png
    |    |    |    |    |   ...
    |    |    |    |    |--target
    |    |    |    |    |   |--1.png
    |    |    |    |    |   ...
    |    |    |--enhance
    |    |    |    |--lol
    |    |    |    |    |--input
    |    |    |    |    |   |--1.png
    |    |    |    |    |   ...
    |    |    |    |    |--target
    |    |    |    |    |   |--1.png
    |    |    |    |    |   ...
```
Dataset links (NOTE: The images of each dataset can be obtained from their official website.)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: Train[ RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2), Test [SOTS-Outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

Deblur: [GoPro](https://drive.google.com/file/d/1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v/view?usp=sharing)

Enhance: [LOL-V1](https://daooshee.github.io/BMVC2018website/)

## Training

```
python train.py
```

## Testing

```
python test.py --mode {n}
```
n is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehazing, 3 for deblurring, 4 for enhancement, 5 for three-degradation all-in-one setting and 6 for five-degradation all-in-one setting.
