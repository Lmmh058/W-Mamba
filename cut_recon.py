import torch
import torch.nn.functional as F

# 定义裁剪函数
def crop_and_pad_image(image, crop_size):
    _, _, H, W = image.size()
    crop_H, crop_W = crop_size

    pad_H = max(0, crop_H - H)
    pad_W = max(0, crop_W - W)

    image = F.pad(image, (0, pad_W, 0, pad_H), mode='constant', value=0)

    return image[:, :, :crop_H, :crop_W],pad_H,pad_W

# # 测试图像的尺寸
# test_image = torch.randn(1, 1, 256, 256)  # 例如，测试图像尺寸为256x256

# # 定义裁剪尺寸
# crop_size = (192, 192)

def crop_image(test_image,crop_size):


    cropped_images = []
    for i in range(0, test_image.size(2), crop_size[0]):
        for j in range(0, test_image.size(3), crop_size[1]):
            cropped_image,pad_H,pad_W = crop_and_pad_image(test_image[:, :, i:i+crop_size[0], j:j+crop_size[1]], crop_size)
            cropped_images.append(cropped_image)

    # 将裁剪后的图像块叠加到batch_size维度
    cropped_images = torch.cat(cropped_images, dim=0)

    return cropped_images,pad_H,pad_W


def recon_image(output_images,test_image,crop_size,pad_H,pad_W):

    # 对输出图像进行重组，使其与原始测试图像尺寸相同
    temp_shape=test_image.shape
    reconstructed_image = torch.zeros(temp_shape[0],temp_shape[1],temp_shape[2]+pad_H,temp_shape[3]+pad_W)
    index = 0
    for i in range(0, test_image.size(2), crop_size[0]):
        for j in range(0, test_image.size(3), crop_size[1]):
            reconstructed_image[:, :, i:i + crop_size[0], j:j + crop_size[1]] = output_images[index]
            index += 1

    # 对重组的图像进行裁剪，确保尺寸与原始测试图像相同
    reconstructed_image = reconstructed_image[:, :, :test_image.size(2), :test_image.size(3)]

    return reconstructed_image

    # reconstructed_image 就是最终的输出，与原始测试图像尺寸相同



def crop_image_overlap(test_image,crop_size,overlap_pixels=64):

    cropped_images = []
    overlap_pixels = crop_size[0]//2
    # 对测试图像进行裁剪和填充 如果裁剪到图片边缘 crop and pad image函数会自动填充图片到cropsize大小 pad_H和pad_W是填充大小
    #overlap_pixels是重叠部分大小
    for i in range(0, test_image.size(2), crop_size[0] - overlap_pixels):
        for j in range(0, test_image.size(3), crop_size[1] - overlap_pixels):
            cropped_image,pad_H,pad_W = crop_and_pad_image(test_image[:, :, i:i+crop_size[0], j:j+crop_size[1]], crop_size)
            cropped_images.append(cropped_image)

    cropped_images = torch.cat(cropped_images, dim=0)

    return cropped_images,pad_H,pad_W

def recon_image_overlap_yuan(output_images,test_image,crop_size,pad_H,pad_W,overlap_pixels=64):

    # 对输出图像进行重组，使其与原始测试图像尺寸相同
    overlap_pixels = crop_size[0]//2
    temp_shape=test_image.shape
    output_images.cuda()
    reconstructed_image = torch.zeros(temp_shape[0],temp_shape[1],temp_shape[2]+pad_H,temp_shape[3]+pad_W).cuda()
    overlap_counts = torch.zeros(temp_shape[0],temp_shape[1],temp_shape[2]+pad_H,temp_shape[3]+pad_W).cuda()
    index = 0
    for i in range(0, test_image.size(2), crop_size[0] - overlap_pixels):
        for j in range(0, test_image.size(3), crop_size[1] - overlap_pixels):
            reconstructed_image[:, :, i:i+crop_size[0], j:j+crop_size[1]] += output_images[index]
            overlap_counts[:, :, i:i+crop_size[0], j:j+crop_size[1]] += 1
            index += 1

    # 计算重叠区域的平均值
    reconstructed_image /= overlap_counts

    reconstructed_image = reconstructed_image[:, :, :test_image.size(2), :test_image.size(3)]

    return reconstructed_image

def recon_image_overlap(output_images,test_image,crop_size,pad_H,pad_W,overlap_pixels=64):

    # 对输出图像进行重组，使其与原始测试图像尺寸相同
    overlap_pixels = crop_size[0]//2
    delete_size = crop_size[0]//8
    temp_shape=test_image.shape
    output_images.cuda()
    #原图像尺寸+膨胀尺寸
    reconstructed_image = torch.zeros(temp_shape[0],temp_shape[1],temp_shape[2]+pad_H,temp_shape[3]+pad_W).cuda()
    #用于显示图片的重叠区域
    overlap_counts = torch.zeros(temp_shape[0],temp_shape[1],temp_shape[2]+pad_H,temp_shape[3]+pad_W).cuda()
    index = 0
    last_i = list(range(0, test_image.size(2), crop_size[0] - overlap_pixels))[-1]
    last_j = list(range(0, test_image.size(3), crop_size[1] - overlap_pixels))[-1]
    for i in range(0, test_image.size(2), crop_size[0] - overlap_pixels):
        for j in range(0, test_image.size(3), crop_size[1] - overlap_pixels):
            if i == 0:
                if j == 0:
                    reconstructed_image[:, :, i:(i+crop_size[0]-delete_size), j:(j+crop_size[1]-delete_size)] += output_images[index,:,:-delete_size,:-delete_size]
                    overlap_counts[:, :, i:(i+crop_size[0]-delete_size), j:(j+crop_size[1]-delete_size)] += 1
                elif j == last_j:
                    reconstructed_image[:, :, i:(i+crop_size[0]-delete_size), (j+delete_size):(j+crop_size[1])] += output_images[index,:,:-delete_size,delete_size:]
                    overlap_counts[:, :, i:i + crop_size[0] - delete_size, j + delete_size:j + crop_size[1]] += 1
                else:
                    reconstructed_image[:, :, i:(i+crop_size[0]-delete_size), (j+delete_size):(j+crop_size[1]-delete_size)] += output_images[index,:,:-delete_size,delete_size:-delete_size]
                    overlap_counts[:, :, i:(i + crop_size[0]-delete_size), (j+delete_size):(j + crop_size[1]-delete_size)] += 1
            elif i == last_i:
                if j == 0:
                    reconstructed_image[:, :, (i+delete_size):(i + crop_size[0]), j:(j + crop_size[1]-delete_size)] += output_images[index,:,delete_size:,:-delete_size]
                    overlap_counts[:, :, (i+delete_size):(i + crop_size[0]), j:(j + crop_size[1]-delete_size)] += 1
                elif j == last_j:
                    reconstructed_image[:, :, (i+delete_size):(i + crop_size[0]), (j+delete_size):(j + crop_size[1])] += output_images[index,:,delete_size:,delete_size:]
                    overlap_counts[:, :, (i+delete_size):(i + crop_size[0]), (j+delete_size):(j + crop_size[1])] += 1
                else:
                    reconstructed_image[:, :, (i+delete_size):(i + crop_size[0]), (j+delete_size):(j + crop_size[1]-delete_size)] += output_images[index,:,delete_size:,delete_size:-delete_size]
                    overlap_counts[:, :, (i+delete_size):(i + crop_size[0]), (j+delete_size):(j + crop_size[1]-delete_size)] += 1
            else:
                if j == 0:
                    reconstructed_image[:, :, (i+delete_size):(i + crop_size[0]-delete_size), j:(j + crop_size[1]-delete_size)] += output_images[index,:,delete_size:-delete_size,:-delete_size]
                    overlap_counts[:, :, (i+delete_size):(i + crop_size[0]-delete_size), j:(j + crop_size[1]-delete_size)] += 1
                elif j == last_j:
                    reconstructed_image[:, :, (i+delete_size):(i + crop_size[0]-delete_size), (j+delete_size):(j + crop_size[1])] += output_images[index,:,delete_size:-delete_size,delete_size:]
                    overlap_counts[:, :, (i+delete_size):(i + crop_size[0]-delete_size), (j+delete_size):(j + crop_size[1])] += 1
                else:
                    reconstructed_image[:, :, (i+delete_size):(i + crop_size[0]-delete_size), (j+delete_size):(j + crop_size[1]-delete_size)] += output_images[index,:,delete_size:-delete_size,delete_size:-delete_size]
                    overlap_counts[:, :, (i+delete_size):(i + crop_size[0]-delete_size), (j+delete_size):(j + crop_size[1]-delete_size)] += 1
            index += 1

    # 计算重叠区域的平均值 有没有可能经过叠加最高叠加到255之后就不叠了？
    reconstructed_image /= overlap_counts
    #裁剪掉膨胀区域
    reconstructed_image = reconstructed_image[:, :, :test_image.size(2), :test_image.size(3)]

    return reconstructed_image

    # reconstructed_image 就是最终的输出，与原始测试图像尺寸相同

def reassemble_image(image_blocks, original_shape, stride, overlap, pad_H, pad_W):
    image_blocks.cuda()
    num_blocks, channels, block_height, block_width = image_blocks.size()
    _, _, original_height, original_width = original_shape
    
    # 计算水平和垂直方向的块数
    num_horizontal_blocks = (original_width - stride) // (stride - overlap) + 1
    num_vertical_blocks = (original_height - stride) // (stride - overlap) + 1
    
    # 初始化用于拼接的全零张量
    reconstructed_image = torch.zeros(1, 1, original_height+pad_H, original_width+pad_W).cuda()
    count_map = torch.zeros_like(reconstructed_image).cuda()
    
    # 循环遍历每个图像块并进行拼接
    for i in range(num_blocks):
        # 计算当前块的位置
        row = i // num_horizontal_blocks
        col = i % num_horizontal_blocks
        
        # 计算当前块在重构图像中的位置
        start_y = row * (stride - overlap)
        start_x = col * (stride - overlap)
        
        # 将当前图像块的像素加到重构图像中
        print(reconstructed_image[:, :, start_y:start_y+block_height, start_x:start_x+block_width].shape)
        print(image_blocks[i].shape)
        reconstructed_image[:, :, start_y:start_y+block_height, start_x:start_x+block_width] += image_blocks[i]
        count_map[:, :, start_y:start_y+block_height, start_x:start_x+block_width] += 1
    
    # 取平均值
    reconstructed_image /= count_map

    reconstructed_image = reconstructed_image[:, :, :original_height, original_width]
    
    return reconstructed_image

# 示例用法
# 假设 image_blocks 是尺寸为 [20, 1, 192, 192] 的张量，original_shape 是原始图像的尺寸 [1, 1, 640, 480]
# reconstructed_image = reassemble_image(image_blocks, original_shape)
