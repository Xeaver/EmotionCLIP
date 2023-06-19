from __future__ import annotations
import os.path as osp
from typing import Optional
import math

import numpy as np
from torch.utils.data import Dataset
import torch


# Scale the bboxes accordingly, since the original bboxes are extracted from the scaled images 
def get_correct_bbox(
    boxes: list[list[float]], # [w1, h1, w2, h2, score]
    img_shape: tuple[int, int], # (w, h), returned by Image.size
    yolo_shape: tuple[int, int] = (640, 640) # (w, h)
) -> list[list[float]]:
    # compute the scale ratio
    r = min(yolo_shape[0] / img_shape[0], yolo_shape[1] / img_shape[1])
    # map the human box onto the frame
    new_boxes = [
        [
            max(math.floor(box[0]/r), 0),
            max(math.floor(box[1]/r), 0),
            min(math.ceil(box[2]/r), img_shape[0]),
            min(math.ceil(box[3]/r), img_shape[1]),
            box[4]
        ]
        for box in boxes
    ]
    return new_boxes

# list of boxes -> a single mask
# boxes with score less than threshold are ignored
def bbox_to_mask(
    boxes: list[list[float]], # [w1, h1, w2, h2, score], may be empty list
    target_shape: tuple[int, int], # (w, h), returned by Image.size
    binary: bool = True, # whether to return binary mask or float mask
    binary_threshold: Optional[float] = 0.5
) -> torch.Tensor:
    mask = torch.zeros(target_shape[1], target_shape[0])
    if len(boxes) == 0:
        return mask
    
    if binary:
        for box in boxes:
            if box[4] > binary_threshold:
                mask[box[1]:box[3], box[0]:box[2]] = 1
    else:
        mask = mask.unsqueeze(0).expand(len(boxes), *mask.shape)
        for i, box in enumerate(boxes):
            mask[i, box[1]:box[3], box[0]:box[2]] = box[4]
        mask = mask.max(dim=0)[0]
    return mask


# for debugging purposes
class DummyDataset(Dataset):
    def __init__(
        self,
        data_formats: dict[str, tuple[tuple, torch.dtype] | type] | list[tuple[tuple, torch.dtype] | type],
        length: int = 10000,
    ):
        self.length = length
        # return a dict
        if isinstance(data_formats, dict):
            self.dummy_data = {}
            for k, f in data_formats.items():
                if isinstance(f, tuple):
                    self.dummy_data[k] = torch.zeros(f[0], dtype=f[1])
                elif f is str:
                    self.dummy_data[k] = "dummy"
                elif f in [int, float]:
                    self.dummy_data[k] = 0
                else:
                    raise NotImplementedError(f'Unknown data format: {f}')
        # return a list
        elif isinstance(data_formats, list):
            self.dummy_data = []
            for f in data_formats:
                if isinstance(f, tuple):
                    shape, dtype = f
                    self.dummy_data.append(torch.zeros(shape, dtype=dtype))
                elif f is str:
                    self.dummy_data.append('dummy')
                elif f in [int, float]:
                    self.dummy_data.append(f())
                else:
                    raise NotImplementedError(f'Unknown data format: {f}')
            if len(self.dummy_data) == 1:
                self.dummy_data = self.dummy_data[0]
            else:
                self.dummy_data = tuple(self.dummy_data)
    
    def __len__(self):
        return self.length

    def __getitem__(self, _):
        return self.dummy_data