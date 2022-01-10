import os
import numpy as np
import scipy.io as scio
from torchvision import transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PrototypeNet, Generator, Discriminator, GANLoss, get_scheduler
from utils import set_input_images, CalcSim, log_trick, CalcMap, mkdir_p
from attacked_model import Attacked_Model

class DCHTA(nn.Module):
    def __init__(self, args, Dcfg):
        super(DCHTA, self).__init__()
        self.bit = args.bit
        self.num_classes = Dcfg.num_label
        self.dim_text = Dcfg.tag_dim
        self.batch_size = args.batch_size
        self.model_name = '{}_{}_{}'.format(args.dataset, args.attacked_method, args.bit)
        self.args = args
        self._build_model(args, Dcfg)
        self._save_setting(args)
        if self.args.transfer_attack:
            self.transfer_bit = args.transfer_bit
            self.transfer_model = Attacked_Model(args.transfer_attacked_method, args.dataset, args.transfer_bit, args.attacked_models_path, args.dataset_path)
            self.transfer_model.eval()

    def _build_model(self, args, Dcfg):
        pretrain_model = scio.loadmat(Dcfg.vgg_path)
        self.prototypenet = nn.DataParallel(PrototypeNet(self.dim_text, self.bit, self.num_classes)).cuda()
        self.generator = nn.DataParallel(Generator()).cuda()
        self.discriminator = nn.DataParallel(Discriminator(self.num_classes)).cuda()
        self.criterionGAN = GANLoss('lsgan').cuda()
        self.attacked_model = Attacked_Model(args.attacked_method, args.dataset, args.bit, args.attacked_models_path, args.dataset_path)
        self.attacked_model.eval()

    def _save_setting(self, args):
        self.output_dir = os.path.join(args.output_path, args.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'Model')
        self.image_dir = os.path.join(self.output_dir, 'Image')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)

    def sample(self, image, sample_dir, name):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        image = image.cpu().detach()[0]
        image = transforms.ToPILImage()(image)
        image.convert(mode='RGB').save(os.path.join(sample_dir, name + '.png'), quality=100)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()
        self.args.lr = self.optimizers[0].param_groups[0]['lr']

    def save_prototypenet(self):
        torch.save(self.prototypenet.module.state_dict(),
            os.path.join(self.model_dir, 'prototypenet_{}.pth'.format(self.model_name)))

    def save_generator(self):
        torch.save(self.generator.module.state_dict(),
            os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name)))

    def load_generator(self):
        self.generator.module.load_state_dict(torch.load(os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name))))
        self.generator.eval()

    def load_prototypenet(self):
        self.prototypenet.module.load_state_dict(torch.load(os.path.join(self.model_dir, 'prototypenet_{}.pth'.format(self.model_name))))
        self.prototypenet.eval()

    def train_prototypenet(self, train_images, train_texts, train_labels):
        num_train = train_labels.size(0)
        optimizer_a = torch.optim.Adam(self.prototypenet.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        epochs = 50
        batch_size = 64
        steps = num_train // batch_size + 1
        lr_steps = epochs * steps
        scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()
        B = self.attacked_model.generate_image_hashcode(train_images).cuda()
        for epoch in range(epochs):
            index = np.random.permutation(num_train)
            for i in range(steps):
                end_index = min((i+1)*batch_size, num_train)
                num_index = end_index - i*batch_size
                ind = index[i*batch_size : end_index]
                batch_text = Variable(train_texts[ind]).type(torch.float).cuda()
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                optimizer_a.zero_grad()
                _, mixed_h, mixed_l = self.prototypenet(batch_label, batch_text)
                S = CalcSim(batch_label.cpu(), train_labels.type(torch.float))
                theta_m = mixed_h.mm(Variable(B).t()) / 2
                logloss_m = - ((Variable(S.cuda()) * theta_m - log_trick(theta_m)).sum() / (num_train * num_index))
                regterm_m = (torch.sign(mixed_h) - mixed_h).pow(2).sum() / num_index
                classifer_m = criterion_l2(mixed_l, batch_label)
                loss = classifer_m + 5 * logloss_m + 1e-3 * regterm_m
                loss.backward()
                optimizer_a.step()
                if i % self.args.print_freq == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, l_m:{:.5f}, r_m: {:.5f}, c_m: {:.7f}'
                        .format(epoch, i, scheduler_a.get_last_lr()[0], logloss_m, regterm_m, classifer_m))
                scheduler_a.step()
        self.save_prototypenet()

    def test_prototypenet(self, test_texts, test_labels, database_images, database_texts, database_labels):
        self.load_prototypenet()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        for i in range(num_test):
            _, mixed_h, __ = self.prototypenet(test_labels[i].cuda().float().unsqueeze(0), test_texts[i].cuda().float().unsqueeze(0))
            qB[i, :] = torch.sign(mixed_h.cpu().data)[0]
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        c2i_map = CalcMap(qB, IdB, test_labels, database_labels, 50)
        c2t_map = CalcMap(qB, TdB, test_labels, database_labels, 50)
        print('C2I_MAP: %3.5f, C2T_MAP: %3.5f' % (c2i_map, c2t_map))

    def train(self, train_images, train_texts, train_labels, database_images, database_texts, database_labels, test_texts, test_labels):
        # Stage I: Prototype Learning
        self.train_prototypenet(train_images, train_texts, train_labels)
        self.test_prototypenet(test_texts, test_labels, database_images, database_texts, database_labels)
        # Stage II: Adversarial Generation
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizers = [optimizer_g, optimizer_d]
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]
        num_train = train_labels.size(0)
        batch_size = self.batch_size
        total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
        criterion_l2 = torch.nn.MSELoss()
        B = self.attacked_model.generate_text_feature(train_texts)
        B = B.cuda()
        for epoch in range(self.args.epoch_count, total_epochs):
            print('\nTrain epoch: {}, learning rate: {:.7f}'.format(epoch, self.args.lr))
            index = np.random.permutation(num_train)
            for i in range(num_train // batch_size + 1):
                end_index = min((i+1)*batch_size, num_train)
                num_index = end_index - i*batch_size
                ind = index[i*batch_size : end_index]
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                batch_text = Variable(train_texts[ind]).type(torch.float).cuda()
                batch_image = Variable(train_images[ind]).type(torch.float).cuda()
                batch_image = set_input_images(batch_image/255)
                select_index = np.random.choice(range(train_labels.size(0)), size=num_index)
                batch_target_label = train_labels.index_select(0, torch.from_numpy(select_index)).type(torch.float).cuda()
                batch_target_text = train_texts.index_select(0, torch.from_numpy(select_index)).type(torch.float).cuda()
                label_feature, target_hashcode, _ = self.prototypenet(batch_target_label, batch_target_text)
                batch_fake_image = self.generator(batch_image, label_feature.detach())
                # update D
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    batch_image_d = self.discriminator(batch_image)
                    batch_fake_image_d = self.discriminator(batch_fake_image.detach())
                    real_d_loss = self.criterionGAN(batch_image_d, batch_label, True)
                    fake_d_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, False)
                    d_loss = (real_d_loss + fake_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()
                # update G
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()
                batch_fake_image_m = (batch_fake_image + 1) / 2 * 255
                predicted_target_hash = self.attacked_model.image_model(batch_fake_image_m)
                logloss = - torch.mean(predicted_target_hash * target_hashcode) + 1
                batch_fake_image_d = self.discriminator(batch_fake_image)
                fake_g_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, True)
                reconstruction_loss_l = criterion_l2(batch_fake_image, batch_image)
                # backpropagation
                g_loss = 5 * logloss + 1 * fake_g_loss + 150 * reconstruction_loss_l
                g_loss.backward()
                optimizer_g.step()
                if i % self.args.sample_freq == 0:
                    self.sample((batch_fake_image + 1) / 2, '{}/'.format(self.image_dir), str(epoch) + '_' + str(i) + '_fake')
                    self.sample((batch_image + 1) / 2, '{}/'.format(self.image_dir), str(epoch) + '_' + str(i) + '_real')
                if i % self.args.print_freq == 0:
                    print('step: {:3d} d_loss: {:.3f} g_loss: {:.3f} fake_g_loss: {:.3f} logloss: {:.3f} r_loss_l: {:.7f}'
                        .format(i, d_loss, g_loss, fake_g_loss, logloss, reconstruction_loss_l))
            self.update_learning_rate()
        self.save_generator()

    def test(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size = test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            label_feature, _, __ =  self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda()/255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = 255 * fake_image
            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, fake_image[0]).data
        print('generate target images end!')
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def transfer_attack(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.transfer_bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size = test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            label_feature, _, __ =  self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda()/255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = 255 * fake_image
            target_hashcode = self.transfer_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, fake_image[0]).data
        print('generate target images end!')
        TdB = self.transfer_model.generate_text_hashcode(database_texts)
        IdB = self.transfer_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))