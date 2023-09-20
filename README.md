# Semi-TransUNet

[TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf) is the state-of-the-art algorithm for medial image segmentation by combining the UNet algorithm with stacks of transformer layers. However, in practice, medical image data is hard to gather, since not only is the data source scarce, they also require the consent from patients to be used for research purposes. Therefore, this project aims to explore the possibility of semi-supervised learning with the method ["Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles"](https://arxiv.org/abs/1603.09246), which only use a small set of real-life data set to construct a model that can still perform the segmentation task with a high accuracy.

## Abstract
Medical image segmentation has been a challenging task for deep learning, and one recent advancement in this field is to use image transformers together with a deep convolutional network. TransUNet is the model that has promising results using this technique, that is based on the U-net architecture. However, since obtaining labelled data in the medical field is challenging, it is of public interest to test the possibility of converting TransUNet into a semi-supervised learning model, that do not require all sample data to have labels. This project consists of the following contributions. The first contribution is the evaluation of the performance of the image segmentation model based on image transformers, TransUNet. This test proved to be consistent with the original paper and shows its robustness to different testing data. Then, the second contribution is a research of current semi-supervised learning frameworks that are viable for TransUNet, and the choose one for implementation. In the end, 5 candidate frameworks were selected, and the chosen semi-supervised learning framework is Self-loop uncertainty for its low memory usage. Thirdly, the chosen framework is implemented onto TransUNet. The implementation is complete, however, due to time constraints and one technical issue, during testing, the implementation was not able to perform the correct training. Hence the test result of the implementation was not included. Future work on this project could solve the technical issue preventing the training and perform the planned training and evaluation of the model. Further tests can be conducted on revised model parameters to investigate their effect on the performance and the model’s overall robustness.

## Model Architecture
![Poster_Image](https://github.com/JXKun980/SemiTransUNet/assets/44962755/c9219ec0-5bbf-40f2-a53f-6c2b6a181f9e)

## Usage

The code file structure is as follows:
```bash
.
├── TransUNet
│   ├──datasets
│   │       └── dataset_*.py
│   ├──train.py
│   ├──test.py
│   └──...
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
│           └── *.npz
└── data
    └──Synapse
        ├── test_vol_h5
        │   ├── case0001.npy.h5
        │   └── *.npy.h5
        └── train_npz
            ├── case0005_slice000.npz
            └── *.npz
```

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details, if you would like to use the preprocessed data, please use it for research purposes and do not redistribute it.

### 3. Environment

Please prepare an environment with python=3.7, and then use the command
```
pip install -r requirements.txt
```
for the dependencies.

### 4. Train

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory, and both can reach similar performance.
```bash
python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Continue training a model from checkpoint at epoch 500
```bash
python train.py --dataset Synapse --vit_name R50-ViT-B_16 --continue_from_epoch 500
```


### 5. Test
- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

 
