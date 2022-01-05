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
We used five cross-modal datasets for experiments. Since IAPR-TC12 and MS-COCO do not have common text features, we use pre-trained BERT model to extract 1024-dimension text features. All datasets are available by the following link:

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
