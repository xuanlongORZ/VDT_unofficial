import torch
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, video_path, frames_per_clip, transform=None, mask_ratio=0.6):
        self.video_path = video_path
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.mask_ratio = mask_ratio
        self.video_path = video_path
        self.frame_path = sorted(os.listdir(video_path))
        self.frames = []
        for frame_path_single in self.frame_path:
            temp1 = os.path.join(self.video_path, frame_path_single)
            frame_items_1 = sorted(os.listdir(temp1))
            for frame_item_1 in frame_items_1:
                temp2 = os.path.join(temp1, frame_item_1)
                frame_items_2 = sorted(os.listdir(temp2))
                for frame_item_2 in frame_items_2:
                    temp3 = os.path.join(temp2, frame_item_2)
                    self.frames.append(temp3)

    def _extract_frames(self, idx):
        frame_name = self.frames[idx]
        cap = cv2.VideoCapture(frame_name)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[np.newaxis, :, :, :]
            frames.append(frame)
        frames = np.concatenate(frames, axis=0)
        cap.release()
        start_frame = len(frames) - self.frames_per_clip
        start_frame = np.random.randint(start_frame)
        frames = frames[start_frame:(start_frame+self.frames_per_clip), :, :, :]
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frames = self._extract_frames(idx)

        if self.transform:
            clip = [self.transform(frame) for frame in frames]
        clip = torch.stack(clip)
        # print('clip.shape', clip.shape)

        return clip



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])