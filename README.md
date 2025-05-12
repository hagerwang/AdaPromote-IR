# AdaPromote-IR

Adaptive Learning to Perceive Degradation Semantic and Prompting for All-in-One Image Restoration.

This is a Pytorch implementation of [AdaPromote-IR](https://github.com/hagerwang/AdaPromote-IR/).
![image](https://github.com/hagerwang/AdaPromote-IR/blob/main/framework.png)  

NOTE: Full code will be available soon.

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
    |    |    |    |    |--GOPR0372_07_00_000048.png
    |    |    |    |     ...
    |    |    |    |--sharp
    |    |    |    |    |--GOPR0372_07_00_000047.png
    |    |    |    |    |--GOPR0372_07_00_000048.png
    |    |    |    |     ...
    |    |    |--Dehaze
    |    |    |    |--original
    |    |    |    |    |--0025.png
    |    |    |    |    |--0039.png
    |    |    |    |     ...
    |    |    |    |--synthetic
    |    |    |    |    |--part1
    |    |    |    |    |    |--0025_0.8_0.1.jpg
    |    |    |    |    |    |--0025_0.8_0.2.jpg
    |    |    |    |    |    ...
    |    |    |    |    |--part2
    |    |    |    |    |    |--3068_0.8_0.1.jpg
    |    |    |    |    |    |--3068_0.8_0.2.jpg
    |    |    |    |    |    ...    
    |    |    |    |    |--part3
    |    |    |    |    |    |--5425_0.8_0.1.jpg
    |    |    |    |    |    |--5425_0.8_0.2.jpg
    |    |    |    |    |    ...   
    |    |    |    |    |--part4
    |    |    |    |    |    |--6823_0.8_0.1.jpg
    |    |    |    |    |    |--6823_0.8_0.2.jpg
    |    |    |    |    |    ...
    |    |    |--Denoise
    |    |    |    |--00001.bmp
    |    |    |    |--00001.bmp
    |    |    |    ...
    |    |    |    |--5096.jpg
    |    |    |    |--6046.jpg
    |    |    |    ...
    |    |    |--Derain
    |    |    |    |--gt
    |    |    |    |    |--norain-1.png
    |    |    |    |    |--norain-2.png
    |    |    |    |     ...
    |    |    |    |--rainy
    |    |    |    |    |--rain-1.png
    |    |    |    |    |--rain-2.png
    |    |    |    |     ...
    |    |    |--Enhance
    |    |    |    |--gt
    |    |    |    |    |--2.png
    |    |    |    |    |--5.png
    |    |    |    |     ...
    |    |    |    |--low
    |    |    |    |    |--2.png
    |    |    |    |    |--5.png
    |    |    |    |     ...
    |    |--test
    |    |    |--deblur
    |    |    |    |--gopro
    |    |    |    |    |--input
    |    |    |    |    |   |--GOPR0384_11_00_000001.png
    |    |    |    |    |   |--GOPR0384_11_00_000002.png
    |    |    |    |    |   ...
    |    |    |    |    |--target
    |    |    |    |    |   |--GOPR0384_11_00_000001.png
    |    |    |    |    |   |--GOPR0384_11_00_000002.png
    |    |    |    |    |   ...
    |    |    |--dehaze
    |    |    |    |--input
    |    |    |    |    |--0001_0.8_0.2.jpg
    |    |    |    |    |--0002_0.8_0.08.jpg
    |    |    |    |    ...
    |    |    |    |--target
    |    |    |    |   |--0001.png
    |    |    |    |   |--0002.png
    |    |    |    |   ...
    |    |    |--denoise
    |    |    |    |--bsd68
    |    |    |    |    |--3096.jpg
    |    |    |    |    |--12084.jpg
    |    |    |    |    ...
    |    |    |    |--urban100
    |    |    |    |    |--img_001.png
    |    |    |    |    |--img_002.png
    |    |    |    |    ...
    |    |    |    |--kodak24
    |    |    |    |    |--kodim01.png
    |    |    |    |    |--kodim02.png
    |    |    |    |    ...
    |    |    |--derain
    |    |    |    |--Rain100L
    |    |    |    |    |--input
    |    |    |    |    |   |--1.png
    |    |    |    |    |   |--2.png
    |    |    |    |    |   ...
    |    |    |    |    |--target
    |    |    |    |    |   |--1.png
    |    |    |    |    |   |--2.png
    |    |    |    |    |   ...
    |    |    |--enhance
    |    |    |    |--lol
    |    |    |    |    |--input
    |    |    |    |    |   |--1.png
    |    |    |    |    |   |--22.png
    |    |    |    |    |   ...
    |    |    |    |    |--target
    |    |    |    |    |   |--1.png
    |    |    |    |    |   |--22.png
    |    |    |    |    |   ...
```


Dataset links (NOTE: The images of each dataset can be obtained from their official website.)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: Train[ RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2), Test [SOTS-Outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

Deblur: [GoPro](https://drive.google.com/file/d/1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v/view?usp=sharing)

Enhance: [LOL-V1](https://daooshee.github.io/BMVC2018website/)
