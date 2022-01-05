import torch
import torch.nn as nn
import scipy.io as scio


class Attacked_Model(nn.Module):
    def __init__(self, method, dataset, bit, attacked_models_path, dataset_path):
        super(Attacked_Model, self).__init__()
        self.method = method
        self.dataset = dataset
        self.bit = bit
        vgg_path = dataset_path + 'imagenet-vgg-f.mat'
        if self.dataset == 'WIKI':
            tag_dim = 10
            num_label = 10
        if self.dataset == 'IAPR':
            tag_dim = 1024
            num_label = 255
        if self.dataset == 'FLICKR':
            tag_dim = 1386
            num_label = 24
        if self.dataset == 'COCO':
            tag_dim = 1024
            num_label = 80
        if self.dataset == 'NUS':
            tag_dim = 1000
            num_label = 21

        if self.method == 'DCMH':
            load_img_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/image_model.pth'
            load_txt_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/text_model.pth'
            from attacked_methods.DCMH.img_module import ImgModule
            from attacked_methods.DCMH.txt_module import TxtModule
            pretrain_model = scio.loadmat(vgg_path)
            self.image_hashing_model = ImgModule(self.bit, pretrain_model)
            self.text_hashing_model = TxtModule(tag_dim, self.bit)
            self.image_hashing_model.load(load_img_path)
            self.text_hashing_model.load(load_txt_path)
            self.image_hashing_model.cuda().eval()
            self.text_hashing_model.cuda().eval()
        if self.method == 'CPAH':
            CPAH_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/CPAH.pth'
            image_dim = 4096
            hidden_dim = 512
            from attacked_methods.CPAH.CNN_F import image_net
            from attacked_methods.CPAH.CPAH import CPAH
            pretrain_model = scio.loadmat(vgg_path)
            self.vgg = image_net(pretrain_model)
            self.model = CPAH(image_dim, tag_dim, hidden_dim, self.bit, num_label)
            self.model.load(CPAH_path)
            self.vgg.cuda().eval()
            self.model.cuda().eval()
        if self.method == 'DADH':
            DADH_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/DADH.pth'
            dropout = False
            image_dim = 4096
            hidden_dim = 8192
            from attacked_methods.DADH.DADH import GEN
            pretrain_model = scio.loadmat(vgg_path)
            self.generator = GEN(dropout, image_dim, tag_dim, hidden_dim, self.bit, pretrain_model=pretrain_model)
            self.generator.load(DADH_path)
            self.generator.cuda().eval()
        if self.method == 'DJSRH':
            DJSRH_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/DJSRH_WON.pth'
            from attacked_methods.DJSRH.DJSRH import ImgNet, TxtNet
            self.img_model = ImgNet(self.bit)
            self.txt_model = TxtNet(self.bit, tag_dim)
            models = torch.load(DJSRH_path, map_location=lambda storage, loc: storage.cuda())
            self.img_model.load_state_dict(models['ImgNet'])
            self.txt_model.load_state_dict(models['TxtNet'])
            self.img_model.cuda().eval()
            self.txt_model.cuda().eval()
        if self.method == 'JDSH':
            JDSH_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/JDSH_WON.pth'
            from attacked_methods.JDSH.JDSH import ImgNet, TxtNet
            self.img_model = ImgNet(self.bit)
            self.txt_model = TxtNet(self.bit, tag_dim)
            models = torch.load(JDSH_path, map_location='cpu')
            self.img_model.load_state_dict(models['ImgNet'])
            self.txt_model.load_state_dict(models['TxtNet'])
            self.img_model.cuda().eval()
            self.txt_model.cuda().eval()
        if self.method == 'DGCPN':
            DGCPN_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/DGCPN.pth'
            from attacked_methods.DGCPN.DGCPN import ImgNet, TxtNet, ImgModule
            pretrain_model = scio.loadmat(vgg_path)
            self.vgg = ImgModule(pretrain_model)
            self.img_model = ImgNet(self.bit)
            self.txt_model = TxtNet(self.bit, tag_dim)
            models = torch.load(DGCPN_path, map_location='cpu')
            self.img_model.load_state_dict(models['ImgNet'])
            self.txt_model.load_state_dict(models['TxtNet'])
            self.vgg.cuda().eval()
            self.img_model.cuda().eval()
            self.txt_model.cuda().eval()

    def generate_image_feature(self, data_images):
        num_data = data_images.size(0)
        B = torch.zeros(num_data, self.bit)
        if self.method == 'DCMH':
            for i in range(num_data):
                output = self.image_hashing_model(data_images[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.data
        if self.method == 'CPAH':
            for i in range(num_data):
                temp = self.vgg(data_images[i].type(torch.float).unsqueeze(0).cuda())
                output = self.model.generate_img_code(temp.squeeze().unsqueeze(0))
                B[i, :] = output.data
        if self.method == 'DADH':
            for i in range(num_data):
                output = self.generator.generate_img_code(data_images[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'DJSRH':
            for i in range(num_data):
                img = data_images[i].type(torch.float).cuda()/255
                _, __, output = self.img_model(img.unsqueeze(0))
                B[i, :] = output.cpu().data
        if self.method == 'JDSH':
            for i in range(num_data):
                img = data_images[i].type(torch.float).cuda()/255
                _, __, output = self.img_model(img.unsqueeze(0))
                B[i, :] = output.cpu().data
        if self.method == 'DGCPN':
            for i in range(num_data):
                temp = self.vgg(data_images[i].type(torch.float).unsqueeze(0).cuda())
                _, output = self.img_model(temp.unsqueeze(0))
                B[i, :] = output.cpu().data
        return B
    
    def generate_text_feature(self, data_texts):
        num_data = data_texts.size(0)
        B = torch.zeros(num_data, self.bit)
        if self.method == 'DCMH':
            for i in range(num_data):
                output = self.text_hashing_model(data_texts[i].type(torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(3).cuda())
                B[i, :] = output.data
        if self.method == 'CPAH':
            for i in range(num_data):
                output = self.model.generate_txt_code(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.data
        if self.method == 'DADH':
            for i in range(num_data):
                output = self.generator.generate_txt_code(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'DJSRH':
            for i in range(num_data):
                _, __, output = self.txt_model(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'JDSH':
            for i in range(num_data):
                output = self.txt_model(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'DGCPN':
            for i in range(num_data):
                _, output = self.txt_model(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        return B

    def generate_image_hashcode(self, data_images):
        num_data = data_images.size(0)
        B = torch.zeros(num_data, self.bit)
        if self.method == 'DCMH':
            for i in range(num_data):
                output = self.image_hashing_model(data_images[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.data
        if self.method == 'CPAH':
            for i in range(num_data):
                temp = self.vgg(data_images[i].type(torch.float).unsqueeze(0).cuda())
                output = self.model.generate_img_code(temp.squeeze().unsqueeze(0))
                B[i, :] = output.data
        if self.method == 'DADH':
            for i in range(num_data):
                output = self.generator.generate_img_code(data_images[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'DJSRH':
            for i in range(num_data):
                img = data_images[i].type(torch.float).cuda()/255
                _, __, output = self.img_model(img.unsqueeze(0))
                B[i, :] = output.cpu().data
        if self.method == 'JDSH':
            for i in range(num_data):
                img = data_images[i].type(torch.float).cuda()/255
                _, __, output = self.img_model(img.unsqueeze(0))
                B[i, :] = output.cpu().data
        if self.method == 'DGCPN':
            for i in range(num_data):
                temp = self.vgg(data_images[i].type(torch.float).unsqueeze(0).cuda())
                _, output = self.img_model(temp.unsqueeze(0))
                B[i, :] = output.cpu().data
        return torch.sign(B)

    def generate_text_hashcode(self, data_texts):
        num_data = data_texts.size(0)
        B = torch.zeros(num_data, self.bit)
        if self.method == 'DCMH':
            for i in range(num_data):
                output = self.text_hashing_model(data_texts[i].type(torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(3).cuda())
                B[i, :] = output.data
        if self.method == 'CPAH':
            for i in range(num_data):
                output = self.model.generate_txt_code(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.data
        if self.method == 'DADH':
            for i in range(num_data):
                output = self.generator.generate_txt_code(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'DJSRH':
            for i in range(num_data):
                _, __, output = self.txt_model(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'JDSH':
            for i in range(num_data):
                output = self.txt_model(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'DGCPN':
            for i in range(num_data):
                _, output = self.txt_model(data_texts[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        return torch.sign(B)

    def image_model(self, data_images):
        if self.method == 'DCMH':
            output = self.image_hashing_model(data_images)
        if self.method == 'CPAH':
            data_images = self.vgg(data_images)
            output = self.model.img_model(data_images)
        if self.method == 'DADH':
            output = self.generator.img_model(data_images)
        if self.method == 'DJSRH':
            data_images = data_images/255
            _, __, output = self.img_model(data_images)
        if self.method == 'JDSH':
            data_images = data_images/255
            _, __, output = self.img_model(data_images)
        if self.method == 'DGCPN':
            data_images = self.vgg(data_images)
            _, output = self.img_model(data_images)
        return output