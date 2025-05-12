# AdaPromote-IR

Adaptive Learning to Perceive Degradation Semantic and Prompting for All-in-One Image Restoration.

This is a Pytorch implementation of [AdaPromote-IR](https://github.com/hagerwang/AdaPromote-IR/).
![image](https://github.com/hagerwang/AdaPromote-IR/blob/main/framework.png)  

NOTE: Full code will be available soon.

## Prerequisites 

1. Install the repository

```
conda create -n PBO python=3.8 -y
conda activate PBO
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
cd PBO
pip install -r requirement.txt
```

2. Dataset preparation

```
data
|_ TD500
|  |_ Evel
|  |_ Test
|  |_ Test1
|  |_ Train
|_ ctw1500
|  |_ test
|  |_ train
.......
```

Dataset links (NOTE: The images of each dataset can be obtained from their official website.)

[TD500](https://drive.google.com/file/d/1ByluLnyd8-Ltjo9AC-1m7omZnI-FA1u0/view?usp=sharing) 

[CTW1500](https://drive.google.com/file/d/1A2s3FonXq4dHhD64A2NCWc8NQWMH2NFR/view?usp=sharing) 

[Total-Text](https://drive.google.com/file/d/17_7T_-2Bu3KSSg2OkXeCxj97TBsjvueC/view?usp=sharing)
