import argparse

import cv2
import numpy as np
import os
import time
import sys
from network_WMamba import WMamba as net
from utils1 import utils_image as util
from data.dataloder import Dataset as D
from torch.utils.data import DataLoader
from cut_recon import *
from utils1.utils_image import channel_convert

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./Model/Infrared_Visible_Fusion/models/')
    parser.add_argument('--model_number', type=str,default='best')
    parser.add_argument('--dataset', type=str, default='test_image',
                        help='input test image name')
    parser.add_argument('--A_dir', type=str, default='infrared',
                        help='input test ir image name')
    parser.add_argument('--B_dir', type=str, default='visible',
                        help='input test vi image name')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    parser.add_argument('--crop_size', type=int, default=128, help='size of image block')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model_path = os.path.join(args.model_path, args.model_number + '_WMamba.pth')
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        print('Traget model path: {} not existing!!!'.format(model_path))
        sys.exit()
    model = define_model(args)
    model.eval()
    model = model.to(device)

    save_dir= setup(args)

    a_dir = f'./{args.dataset}/{args.A_dir}'
    b_dir = f'./{args.dataset}/{args.B_dir}'
    os.makedirs(save_dir, exist_ok=True)
    vi_channels = 3
    test_set = D(a_dir, b_dir, ir_channels=1,vi_channels=vi_channels)
    test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
    t = []
    for i, test_data in enumerate(test_loader):
        name=os.path.splitext(os.path.basename(test_set.paths_A[i]))[0]
        img_a = test_data['A'].to(device) #[1,1,480,640]
        img_b = test_data['B'].to(device)
        img_b_uv = test_data['img_B_uv']
        start = time.time()
        with torch.no_grad():
            #image block test
            crop_size = args.crop_size
            img_a_crop,pada_H,pada_W = crop_image_overlap(img_a,(crop_size,crop_size))
            img_b_crop,padb_H,padb_W = crop_image_overlap(img_b,(crop_size,crop_size))
            output_crop = torch.zeros_like(img_a_crop)
            for ii in range(img_a_crop.shape[0]):
                output_crop[ii:ii+1] = test_vim(img_a_crop[ii:ii+1], img_b_crop[ii:ii+1], model)
            output = recon_image_overlap(output_crop, img_a, (crop_size, crop_size), pada_H, pada_W)
            output = output.detach()[0].float().cpu()
        output = util.tensor2uint(output)

        if vi_channels == 3:
            output = np.expand_dims(output, axis=2)
            img_b_uv = img_b_uv[0].numpy().astype(np.uint8)
            if output.shape[:2] != img_b_uv.shape[:2]:
                raise ValueError("Y channel and UV channels must have the same height and width.")
            img_yuv = cv2.merge((output, img_b_uv[:, :, 0], img_b_uv[:, :, 1]))
            #print(img_yuv.shape)
            output = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB)
        end = time.time()
        save_name = os.path.join(save_dir, name+'.bmp')
        util.imsave(output, save_name)
        print("[{}/{}]  Saving fused image to : {}, Processing time is {:4f} s".format(i+1, len(test_loader), save_name, end - start))
        t.append(end - start)
    print("mean:%s, std: %s" % (np.mean(t), np.std(t)))
    
def define_model(args):
    model = net(in_chans=args.in_channel,embed_dim=192)
    param_key_g = 'params'
    model_path = os.path.join(args.model_path, args.model_number + '_WMamba.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model

def setup(args):   
    save_dir = f'./results/{args.dataset}'
    return save_dir

def test_vim(img_a, img_b, model):
    output = model(img_a, img_b)
    return output

if __name__ == '__main__':
    main()
