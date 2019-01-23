import glob
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as skio
from torch.utils import data
from torchvision import transforms
from PIL import Image

class Video(data.Dataset):
    def __init__(self, video_dir, interval=2, middle_interval=1,
                 preload_imgs=False, transform_fn=transforms.ToTensor()):
        self.video_dir = video_dir
        self.interval = interval
        self.middle_interval = middle_interval
        self.preload_imgs = preload_imgs
        self.transform_fn = transform_fn
        self.img_paths = sorted(glob.glob(self.video_dir + '/*'))
        if self.preload_imgs:
            self.imgs = map(lambda x: Image.fromarray(skio.imread(x)), self.img_paths)
    def __len__(self):
        return len(self.img_paths) - self.interval
    def __getitem__(self, idx):
        if self.preload_imgs:
            img_open_fn = lambda i: self.imgs[i]
        else:
            img_open_fn = lambda i: Image.open(self.img_paths[i])
        img, img2 = img_open_fn(idx), img_open_fn(idx+self.interval)
        middle_img = img_open_fn(idx+self.middle_interval)
        img, img2, middle_img = map(self.transform_fn, [img, img2, middle_img])
        return img, img2, middle_img

class VideoDataset:
    def __init__(self, videos, batch_size, shuffle=False, num_workers=0):
        self.videos = videos
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = data.ConcatDataset(videos)
        self.data_loader = data.DataLoader(self.dataset, batch_size, shuffle,
                                           pin_memory=True, num_workers=num_workers)

"""
class VideoDataset:
    def __init__(self, videos):
        self.videos = videos
        self.video_lens = map(len, self.videos)
        # self.transforms = transforms.Compose(
        #         [transforms.Resize((384, 1280)),
        #          transforms.ToTensor(),
        #          transforms.Normalize([0.5, 0.5, 0.5],
        #                               [0.5, 0.5, 0.5])])
        self.transforms = transforms.ToTensor()
    def sample(self):
        video_idx = np.random.randint(len(self.videos))
        video = self.videos[video_idx]
        frame_idx = np.random.randint(len(video) - interval)
        frame1, frame2 = video[frame_idx], video[frame_idx + interval]
        middle_frame = video[frame_idx + middle_interval]
        frame1 = self.transforms(frame1)
        frame2 = self.transforms(frame2)
        middle_frame = self.transforms(middle_frame)
        return frame1, frame2, middle_frame
    def loop(self, interval=2, middle_interval=1):
        for i, video in enumerate(self.videos):
            for j in range(interval, len(video), interval):
                print(j-interval+middle_interval)
                frame1, frame2 = video[j-interval], video[j]
                middle_frame = video[j-interval+middle_interval]
                frame1 = self.transforms(frame1)
                frame2 = self.transforms(frame2)
                middle_frame = self.transforms(middle_frame)
                yield frame1, frame2, middle_frame
"""

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_op):
        super(DownConv, self).__init__()
        if downsample_op == 'max_pool':
            self.conv1 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            )
        elif downsample_op == 'strided_conv':
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_op):
        super(UpConv, self).__init__()
        if upsample_op == 'bilinear':
            self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            assert in_channels == 2 * out_channels
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        elif upsample_op == 'conv_transpose':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                nn.ReLU()
            )
            self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class Net(nn.Module):
    def __init__(self, device, num_layers, start_channels,
                 upsample_op='bilinear', downsample_op='strided_conv'):
        super(Net, self).__init__()
        self.conv_in = nn.Conv2d(3, start_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(2*start_channels, 3, 3, 1, 1)
        self.down_convs = [nn.Sequential(self.conv_in, nn.ReLU())]
        self.up_convs = []
        for i in range(num_layers):
            conv_layer = DownConv(start_channels*(2**i),
                                  start_channels*(2**(i+1)), downsample_op)
            self.down_convs.append(conv_layer)
            self.add_module('down_conv%d' % i, conv_layer)
        for i in range(num_layers, 0, -1):
            conv_layer = UpConv(start_channels*(2**(i+1)),
                                start_channels*(2**i), upsample_op)
            self.up_convs.append(conv_layer)
            self.add_module('up_conv%d' % (num_layers - i), conv_layer)
        self.up_convs += [self.conv_out]

        # self.deconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        # self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.deconv2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.deconv3 = nn.ConvTranspose2d(64, 32, 2, 2)
        # self.deconv4 = nn.Conv2d(32, 32, 3, 1, 1)
        # self.deconv4_2 = nn.Conv2d(32, 32, 3, 1, 1)
        # self.deconv5 = nn.ConvTranspose2d(32, 16, 2, 2)
        # self.deconv6 = nn.Conv2d(16, 16, 3, 1, 1)
        # self.deconv6_2 = nn.Conv2d(16, 3, 3, 1, 1)
        # self.deconvs = [self.deconv1, self.deconv2, self.deconv2_2, self.deconv3,
        #                 self.deconv4, self.deconv4_2, self.deconv5, self.deconv6, self.deconv6_2]
        # self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear',
        #                              align_corners=True)
        # self.deconv1 = nn.Conv2d(128, 64, 3, 1, 1)
        # self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear',
        #                              align_corners=True)
        # self.deconv3 = nn.Conv2d(64, 32, 3, 1, 1)
        # self.deconv4 = nn.Conv2d(32, 32, 3, 1, 1)
        # self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear',
        #                              align_corners=True)
        # self.deconv5 = nn.Conv2d(32, 16, 3, 1, 1)
        # self.deconv6 = nn.Conv2d(16, 3, 3, 1, 1)
        # self.deconvs = [[self.deconv1, self.deconv2], [self.deconv3,
        #                 self.deconv4], [self.deconv5, self.deconv6]]
        # self.upsamplings = [self.upsample1, self.upsample2, self.upsample3]

    def forward(self, x):
        frame1, frame2 = x
        frame1 = self.apply_down_convs(frame1)
        frame2 = self.apply_down_convs(frame2)
        x = torch.cat([frame1, frame2], dim=1)
        x = self.apply_up_convs(x)
        return x

    def apply_down_convs(self, x):
        for conv in self.down_convs:
            x = conv(x)
        return x

    def apply_up_convs(self, x):
        for conv in self.up_convs:
            x = conv(x)
        x = F.sigmoid(x)
        return x

def preprocess_subdirs(video_dir, height, width, output_dir, batch_size=64):
    import tensorflow as tf
    # Tensorflow resize.
    inputs = tf.placeholder(tf.uint8, [None, None, None, 3])
    resized_imgs = tf.image.resize_images(inputs, (height, width)) / 255.
    sess = tf.Session()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    subdirs = sorted(glob.glob(video_dir + '/*'))
    for subdir in subdirs:
        print(subdir)
        full_output_dir = os.path.join(output_dir, subdir.split('/')[-1])
        print(full_output_dir)
        if not os.path.exists(full_output_dir):
            os.mkdir(full_output_dir)
        img_paths = sorted(glob.glob(subdir + '/*'))
        imgs = map(skio.imread, img_paths)
        for i in range(0, len(imgs), batch_size): 
            print(i)
            start, end = i, min(i + batch_size, len(imgs))
            resized = sess.run(resized_imgs, {inputs: imgs[start:end]})
            for j, img in enumerate(resized):
                skio.imsave('%s/%05d.png' % (full_output_dir, i + j), img)

if __name__ == '__main__':
    mode = sys.argv[1] 
    if mode == 'preprocess_subdirs':
        video_dir = sys.argv[2]
        height, width = map(int, sys.argv[3].split(','))
        output_dir = sys.argv[4]
        preprocess_subdirs(video_dir, height, width, output_dir)
