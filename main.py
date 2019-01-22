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
    for i in range(args.num_iters):
        samples = [video_dataset.sample(args.interval, args.middle_interval)
                   for _ in range(args.batch_size)]
        frame1 = torch.stack([x[0] for x in samples], dim=0).to(device)
        frame2 = torch.stack([x[1] for x in samples], dim=0).to(device)
        middle_frames = torch.stack([x[2] for x in samples], dim=0).to(device)
        optimizer.zero_grad()
        output = model.forward((frame1, frame2))
        loss = F.mse_loss(output, middle_frames)
        loss.backward()
        optimizer.step()
        print('Iteration %d, Loss: %f' % (i, loss.item()))

def evaluate(args, model, device, video_dataset):
    model.eval()
    if args.test_video:
        vwriter = skvideo.io.FFmpegWriter(args.test_video, outputdict={'-pix_fmt': 'yuv420p'})
    mse_losses = []
    with torch.no_grad():
        for i, (frame1, frame2, middle_frame) in \
                enumerate(video_dataset.loop(args.interval,
                                             args.middle_interval)):
            frame1 = frame1[None, :, :, :].to(device)
            frame2 = frame2[None, :, :, :].to(device)
            middle_frame = middle_frame[None, :, :, :].to(device)
            output = model((frame1, frame2))
            output1 = model((frame1, output))
            output2 = model((output, frame2))
            loss = F.mse_loss(output, middle_frame)
            mse_losses.append(loss)
            if args.test_video:
                output = output[0].permute(1, 2, 0)
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
    parser.set_defaults(preload_imgs=False)
    args = parser.parse_args()

    device = torch.device("cuda")

    video_paths = sorted(glob.glob(args.video_dir + '/*'))
    videos = map(lambda x: Video(x, args.preload_imgs), video_paths)
    video_dataset = VideoDataset(videos)

    model = Net().to(device)

    if args.mode == 'train':
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(args, model, device, optimizer, video_dataset)
        torch.save(model.state_dict(), args.model_path) 
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model_path))
        evaluate(args, model, device, video_dataset)
    
if __name__ == '__main__':
    main()
