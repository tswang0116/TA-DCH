# DCHTA
Source code for the paper "Targeted Adversarial Attack for Deep Cross-modal Hashing Retrieval".

## Requirements
* python == 3.7.10
* pytorch == 1.4.0
* torchvision == 0.2.1
* numpy == 1.19.2
* h5py == 3.4.0
* scipy == 1.7.1

## Datasets
We used five cross-modal datasets for experiments. Since IAPR-TC12 and MS-COCO do not have common text features, we use the pre-trained BERT model to extract 1024-dimension text features. All datasets are available by the following link:

* WIKIPEDIA:
* IAPR-TC12:
* FLICKR-25K:
* MS-COCO:
* NUS-WIDE:

## Train attacked hashing models
We carry out targeted adversarial attack for six cross-modal hashing methods, including three supervised methods (DCMH, CPAH, DADH) and three unsupervised methods (DJSRH, JDSH, DGCPN). All attacked hashing models can be obtained by the following code:

* Deep Cross-Modal Hashing (DCMH): https://github.com/WendellGul/DCMH
* Consistency-Preserving Adversarial Hashing (CPAH): https://github.com/comrados/cpah
* Deep Adversarial Discrete Hashing (DADH): https://github.com/Zjut-MultimediaPlus/DADH
* Deep Joint-Semantics Reconstructing Hashing (DJSRH): https://github.com/zs-zhong/DJSRH
* Joint-modal Distribution-based Similarity Hashing (JDSH): https://github.com/KaiserLew/JDSH
* Deep Graph-neighbor Coherence Preserving Network (DGCPN): https://github.com/Atmegal/DGCPN

## Train targeted attack model
The attack model can be trained by running `main.py`. Here is a training example:
```shell
python main.py --dataset WIKI --method DCMH --bit 16 --batch_size 24 --learning_rate 1e-4 --n_epochs 50 --n_epochs_decay 100
```
The above command indicates that 16-bit DCMH is attacked on WIKIPEDIA. During the training, the initial learning rate is set to 0.0001, the normal iteration and the iteration with decay are set to 50 and 100 respectively.

## Test targeted attack performance
Test commands are similar to training commands. To test the attack performance for the 16-bit DCMH on WIKIPEDIA, run the following command:
```shell
python main.py --train False --test True --dataset WIKI --method DCMH --bit 16
```

## Test transferable attack performance
The transferable attack performance can be tested directly by using trained targeted attack model. To test the transferable attack performance of the 16-bit DCMH attack model for 32-bit DADH, use the following command:
```shell
python main.py --train False --test False --transfer_attack True --dataset WIKI --method DCMH --bit 16 --transfer_attacked_method DADH --transfer_bit 32
```

## Citation
If you find this code useful, please cite our paper:<br>
Coming soon...
