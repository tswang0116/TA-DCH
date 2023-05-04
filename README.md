# TA-DCH
Source code for the paper "Targeted Adversarial Attack for Deep Cross-modal Hashing Retrieval".

## Requirements
* python == 3.7.10
* pytorch == 1.4.0
* torchvision == 0.2.1
* numpy == 1.19.2
* h5py == 3.4.0
* scipy == 1.7.1

## Datasets
We use five cross-modal datasets for experiments. Since IAPR-TC12 and MS-COCO do not have common text features, we use the pre-trained BERT model to extract 1024-dimension text features. All datasets are available by the following link:

* WIKIPEDIA: https://pan.baidu.com/s/1YWMLqL56TakLfcEIjIDf4g <br> Password: 3475
* IAPR-TC12: https://pan.baidu.com/s/1YjItEeUmYj2_oPQeToV_Sw <br> Password: dx2s
* FLICKR-25K: https://pan.baidu.com/s/1Ie9PDqC9mAmBdxqX0KJ0ng <br> Password: yjkd
* MS-COCO: https://pan.baidu.com/s/1ocZTVx1GFFdceoSYbIWkbQ <br> Password: 2a6l
* NUS-WIDE: https://pan.baidu.com/s/1Yvqt4Bdjsq1gPaJn2IqIEw <br> Password: doi1

## Train attacked hashing models
We carry out targeted adversarial attack for six cross-modal hashing methods, including three supervised methods (DCMH, CPAH, DADH) and three unsupervised methods (DJSRH, JDSH, DGCPN). All attacked hashing models can be obtained by the following link:

* Deep Cross-Modal Hashing (DCMH): https://github.com/WendellGul/DCMH
* Consistency-Preserving Adversarial Hashing (CPAH): https://github.com/comrados/cpah
* Deep Adversarial Discrete Hashing (DADH): https://github.com/Zjut-MultimediaPlus/DADH
* Deep Joint-Semantics Reconstructing Hashing (DJSRH): https://github.com/zs-zhong/DJSRH
* Joint-modal Distribution-based Similarity Hashing (JDSH): https://github.com/KaiserLew/JDSH
* Deep Graph-neighbor Coherence Preserving Network (DGCPN): https://github.com/Atmegal/DGCPN

## Train targeted attack model
The attack model can be trained by running `main.py`. Here is a training example:
```shell
python main.py --train --test --dataset WIKI --attacked_method DGCPN --bit 32 --batch_size 24 --learning_rate 1e-4 --n_epochs 50 --n_epochs_decay 100
```
The above command indicates that 32-bit DGCPN is attacked on WIKIPEDIA. During the training, the initial learning rate is set to 0.0001, the normal iteration and the iteration with decay are set to 50 and 100 respectively.

Noted that the weight of the image reconstruction loss needs to be adjusted when attacking different hashing models.

## Test targeted attack performance
Test commands are similar to training commands. To test the attack performance for the 32-bit DGCPN on WIKIPEDIA, run the following command:
```shell
python main.py --test --dataset WIKI --attacked_method DGCPN --bit 32
```

## Test transferable attack performance
The transferable attack performance can be tested directly by using trained targeted attack model. To test the transferable attack performance of the 32-bit DGCPN attack model for 128-bit DADH, use the following command:
```shell
python main.py --train --test --transfer_attack --dataset WIKI --attacked_method DGCPN --bit 32 --transfer_attacked_method DADH --transfer_bit 128
```

## An attack example for 32-bit DGCPN on WIKEPEDIA
We provide an example here. We use the proposed TA-DCH to attack the 32-bit DGCPN on the WIKIPEDIA dataset. You need to put the hashing model `DGCPN.pth` in a path like `./attacked_models/DGCPN_WIKI_32/DGCPN.pth`. The hashing model `DGCPN.pth` can be obtained by the following link:
* 32-bit DGCPN on WIKIPEDIA: https://pan.baidu.com/s/1bX7-lmpR01VerYs8JmvREQ <br> Password: xaa1

Alternatively, you can use the trained attack model for testing. The trained attack model for 32-bit DGCPN on WIKIPEDIA can be obtained by the following link:
* The trained attack model: https://pan.baidu.com/s/11-tImjfgQKwzpywPnb9LOg <br> Password: f56x

## Citation
Coming soon...
