import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from tqdm import tqdm

from utils1 import utils_logger
from utils1 import utils_image as util
from utils1 import utils_option as option
from utils1.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset

import warnings
warnings.filterwarnings("ignore")

def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from model_plain import ModelPlain as M
    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def main(json_path='train.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    global train_size
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    if opt['dist']:
        init_dist('pytorch')

    opt['rank'], opt['world_size'] = get_dist_info()
    if opt['rank'] == 0:
        for key, path in opt['path'].items():
            print(path)
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')

    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E

    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG

    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))
    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            #定义数据集 在这里就已经将彩色的VI变成单通道灰度图了
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    model = define_Model(opt)
    model.init_train()

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    last_three_epochs = []
    min_loss = float('inf')
    best_epoch = None
    for epoch in range(opt['train']['epoch']):  # keep running
        for i, train_data in tqdm(enumerate(train_loader),total=len(train_loader)):
            print('current_step:{}'.format(current_step))
            current_step += 1
            model.feed_data(train_data, need_GT=False)
            model.optimize_parameters(current_step)

        model.update_learning_rate(epoch)

        if opt['rank'] == 0:
            # 获取日志信息
            logs = model.current_log()  # such as loss

            message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                      model.current_learning_rate())
            # 添加所有键值对
            for k, v in logs.items():  # merge log information into message
                message += '{:s}: {:.3e} '.format(k, v)
            logger.info(message)


        if opt['rank'] == 0:
            logs = model.current_log()
            current_loss = logs['Fusion_loss']

            if len(last_three_epochs) >= 5:
                oldest_epoch = last_three_epochs.pop(0)
                oldest_model_path = '{}/{}_WMamba.pth'.format(opt['path']['models'], oldest_epoch)
                os.remove(oldest_model_path)

            last_three_epochs.append(epoch)

            save_dir = opt['path']['models']
            model_save_path = '{}/{}_WMamba.pth'.format(save_dir, epoch)
            model.save_network(save_dir, model.netG, 'WMamba', epoch)
            logger.info('Saving the model for epoch {}. Save path is: {}'.format(epoch, model_save_path))

            if current_loss < min_loss:
                if best_epoch is not None:
                    previous_best_model_path = '{}/WMamba_best.pth'.format(save_dir)
                    if os.path.exists(previous_best_model_path):
                        os.remove(previous_best_model_path)
                min_loss = current_loss
                best_epoch = epoch
                model.save_network(save_dir, model.netG, 'WMamba', 'best')
                logs = model.current_log()  # such as loss
                message_best = '<Saving the best model for epoch {}. Save path is: {}> '.format(best_epoch, save_dir)
                for k, v in logs.items():  # merge log information into message
                    message_best += '{:s}: {:.3e} '.format(k, v)
                logger.info(message_best)

if __name__ == '__main__':
    main()
