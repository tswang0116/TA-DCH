import h5py
import torch


def load_dataset(path):
    Data = h5py.File(path)
    images = Data['IAll'][:]
    labels = Data['LAll'][:]
    tags = Data['TAll'][:]
    images = images.transpose(3,2,0,1)
    labels = labels.transpose(1,0)
    tags = tags.transpose(1,0)
    Data.close()
    return images, tags, labels


def split_dataset(images, tags, labels, query_size, training_size, database_size):
    X = {}
    X['query'] = images[0: query_size]
    X['train'] = images[query_size: training_size + query_size]
    X['retrieval'] = images[query_size: query_size + database_size]
    Y = {}
    Y['query'] = tags[0: query_size]
    Y['train'] = tags[query_size: training_size + query_size]
    Y['retrieval'] = tags[query_size: query_size + database_size]
    L = {}
    L['query'] = labels[0: query_size]
    L['train'] = labels[query_size: training_size + query_size]
    L['retrieval'] = labels[query_size: query_size + database_size]
    return X, Y, L


def allocate_dataset(X, Y, L):
    train_images = torch.from_numpy(X['train'])
    train_texts = torch.from_numpy(Y['train'])
    train_labels = torch.from_numpy(L['train'])
    database_images = torch.from_numpy(X['retrieval'])
    database_texts = torch.from_numpy(Y['retrieval'])
    database_labels = torch.from_numpy(L['retrieval'])
    test_images = torch.from_numpy(X['query'])
    test_texts = torch.from_numpy(Y['query'])
    test_labels = torch.from_numpy(L['query'])
    return train_images, train_texts, train_labels, database_images, database_texts, database_labels, test_images, test_texts, test_labels


class Dataset_Config(object):
    def __init__(self, dataset, dataset_path):
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.vgg_path = self.dataset_path + 'imagenet-vgg-f.mat'

        if self.dataset == 'WIKI':
            self.data_path = self.dataset_path + 'WIKIPEDIA.mat'
            self.tag_dim = 10
            self.query_size = 693
            self.training_size = 2173
            self.database_size = 2173
            self.num_label = 10

        if self.dataset == 'IAPR':
            self.data_path = self.dataset_path + 'IAPR-TC12.mat'
            self.tag_dim = 1024
            self.query_size = 2000
            self.training_size = 5000
            self.database_size = 18000
            self.num_label = 255

        if self.dataset == 'FLICKR':
            self.data_path = self.dataset_path + 'FLICKR-25K.mat'
            self.tag_dim = 1386
            self.query_size = 2000
            self.training_size = 5000
            self.database_size = 18015
            self.num_label = 24

        if self.dataset == 'COCO':
            self.data_path = self.dataset_path + 'MS-COCO.mat'
            self.tag_dim = 1024
            self.query_size = 2000
            self.training_size = 10000
            self.database_size = 121287
            self.num_label = 80

        if self.dataset == 'NUS':
            self.data_path = self.dataset_path + 'NUS-WIDE.mat'
            self.tag_dim = 1000
            self.query_size = 2100
            self.training_size = 10500
            self.database_size = 193734
            self.num_label = 21