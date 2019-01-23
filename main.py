import argparse
import glob
import numpy as np
import os
import skimage.io as skio
import skvideo.io
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from util import Video
from util import VideoDataset
from util import Net
from skimage import img_as_ubyte

def train(args, model, device, optimizer, video_dataset):
    model.train()
    i = 0
    while True:
        for frame1, frame2, middle_frames in video_dataset.data_loader:
            if i >= args.num_iters:
                return
            frame1, frame2, middle_frames = frame1.to(device), frame2.to(device), middle_frames.to(device)
            optimizer.zero_grad()
            output = model.forward((frame1, frame2))
            loss = F.mse_loss(output, middle_frames)
            loss.backward()
            optimizer.step()
            print('Iteration %d, Loss: %f' % (i, loss.item()))
            i += 1

def evaluate(args, model, device, video_dataset):
    model.eval()
    if args.test_video:
        vwriter = skvideo.io.FFmpegWriter(args.test_video, outputdict={'-pix_fmt': 'yuv420p'})
    mse_losses = []
    data_loader = video_dataset.data_loader
    data_iter = iter(data_loader)
    with torch.no_grad():
        for i in range(len(data_loader)):
            frame1, frame2, middle_frame = next(data_iter)
            frame1, frame2, middle_frame = frame1.to(device), frame2.to(device), middle_frame.to(device)
            output = model((frame1, frame2))
            output1 = model((frame1, output))
            output2 = model((output, frame2))
            loss = F.mse_loss(output, middle_frame)
            mse_losses.append(loss)
            if args.test_video:
                output = output[0].permute(1, 2, 0).cpu().numpy()
                vwriter.writeFrame(img_as_ubyte(output))
            print('Frame number %d, Loss: %f' % (i, loss))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Directory containing video directories.')
    parser.add_argument('--interval', default=2, type=int,
                        help='Interval between frames.')
    parser.add_argument('--middle_interval', default=1, type=int,
                        help='Prediction target in between.')
    parser.add_argument('--num_iters', default=100000, type=int,
                        help="Number of training iterations.")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size.")
    parser.add_argument('--model_path', default='video_upsampling.pt',
                        help="Path to video upsampling model.")
    parser.add_argument('--mode', help="Train or eval.")
    parser.add_argument('--test_video', help="Test video output path.")
    parser.add_argument('--preload_imgs', dest='preload_imgs', action='store_true',
                        help="Whether to load images into memory ahead of time.")
    parser.add_argument('--num_workers', default=0, type=int,
                        help="Number of workers for DataLoader")
    parser.add_argument('--nn_start_channels', default=16, type=int,
                        help="Number of channels in starting layer.")
    parser.add_argument('--nn_num_layers', default=4, type=int,
                        help="Number of up/down conv layers.")
    parser.add_argument('--upsample_op', default='bilinear', help="Upsampling op.")
    parser.add_argument('--downsample_op', default='strided_conv',
                        help="Downsampling op.")
    parser.set_defaults(preload_imgs=False)
    args = parser.parse_args()

    device = torch.device("cuda")

    video_paths = sorted(glob.glob(args.video_dir + '/*'))
    videos = map(lambda x: Video(x, args.interval, args.middle_interval,
                                 args.preload_imgs), video_paths)

    model = Net(device, args.nn_num_layers,
                args.nn_start_channels,
                args.upsample_op, args.downsampling_op).to(device)

    if args.mode == 'train':
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        video_dataset = VideoDataset(videos, args.batch_size,
                                     shuffle=True, num_workers=args.num_workers)
        train(args, model, device, optimizer, video_dataset)
        torch.save(model.state_dict(), args.model_path)
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model_path))
        video_dataset = VideoDataset(videos, batch_size=1,
                                     shuffle=False, num_workers=args.num_workers)
        evaluate(args, model, device, video_dataset)
    
if __name__ == '__main__':
    main()
