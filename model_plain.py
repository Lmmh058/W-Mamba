from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
import argparse

from select_network import define_G
from model_base import ModelBase
import os

from torch.utils.tensorboard import SummaryWriter
from utils1.utils_model import test_mode
from utils1.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)

        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        # ------------------------------------
        # Define Tensorboard 
        # ------------------------------------
        #创建tensorboard路径 利用summarywriter记录训练过程中的数据
        tensorboard_path = os.path.join(self.opt['path']['tensorboard'], 'Tensorboard')
        print(tensorboard_path)
        os.makedirs(tensorboard_path, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_path)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.step_counter = 0
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'vif':
            from models.loss_vif import fusion_loss_vif
            self.G_lossfn = fusion_loss_vif().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.CosineAnnealingLR(self.G_optimizer,T_max=self.opt_train['epoch']))
    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed under/over data
    # ----------------------------------------
    def feed_data(self, data, need_GT=False, phase='test'):
        self.A = data['A'].to(self.device)
        self.B = data['B'].to(self.device)
        if need_GT:
            self.GT = data['GT'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, phase='test'):
        self.E = self.netG(self.A, self.B)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        print(f"[Debug] Step {current_step} - Optimizing...")

        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_lossfn_type = self.opt_train['G_lossfn_type']
        ## loss function
        if G_lossfn_type in ['vif']:
            total_loss, loss_text, loss_int= self.G_lossfn(self.A, self.B, self.E)
            G_loss = self.G_lossfn_weight * total_loss
            print(f"[Debug] Fusion Loss: {G_loss.item()} | Text: {loss_text.item()} | Int: {loss_int.item()}")
        else:
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.GT)
            print(f"[Debug] Fusion Loss (other): {G_loss.item()}")
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)
        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['Fusion_loss'] = G_loss.item()
        if G_lossfn_type in ['loe', 'mef', 'vif', 'mff', 'gt', 'nir', 'med']:
            self.log_dict['Text_loss'] = loss_text.item()
            self.log_dict['Int_loss'] = loss_int.item()

        self.writer.add_scalar('Loss/Fusion_loss', self.log_dict['Fusion_loss'], current_step)
        self.writer.add_scalar('Loss/Text_loss', self.log_dict['Text_loss'], current_step)
        self.writer.add_scalar('Loss/Int_loss', self.log_dict['Int_loss'], current_step)

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

        #记录训练过程中的图像
        if G_lossfn_type == 'vif':
            if self.step_counter % 100 == 0:
                self.writer.add_image('ir_image', self.A[0],global_step=self.step_counter)
                self.writer.add_image('vi_image', self.B[0],global_step=self.step_counter)
                self.writer.add_image('fused_image', self.E[0],global_step=self.step_counter)
            self.step_counter += 1

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict
