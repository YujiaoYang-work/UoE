"""
This file is from https://github.com/mlpen/Nystromformer
"""

import torch
import torch.nn as nn
import math
from torch.utils.data.dataset import Dataset
import sys
import os
import random
import json
import pickle
import numpy as np
import pdb
from torchvision import transforms

class LRADataset(Dataset):
    def __init__(self, file_path, endless, transform=False):

        self.endless = endless
        with open(file_path, "rb") as f:
            self.examples = pickle.load(f)
            random.shuffle(self.examples)
            self.curr_idx = 0

        self.transform = transform
        self.transform_method = transforms.Compose([
            transforms.RandomCrop(
                32, padding=4, padding_mode="symmetric"
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        print(f"Loaded {file_path}... size={len(self.examples)}", flush = True)

    def __len__(self):
        if self.endless:
            return 1000000000
        else:
            return len(self.examples)

    def create_inst(self, inst):
        output = {}
        if self.transform:
            image = inst["input_ids_0"].reshape(32, 32, 1)
            to_pil_image = transforms.ToPILImage()
            image = to_pil_image(image)
            image = self.transform_method(image)
            image_tensor = image.reshape(-1)
            output["input_ids_0"] = torch.tensor(image_tensor, dtype = torch.long)
        else:
            output["input_ids_0"] = torch.tensor(inst["input_ids_0"], dtype = torch.long)

        output["mask_0"] = (output["input_ids_0"] != 0).float()

        if "input_ids_1" in inst:
            output["input_ids_1"] = torch.tensor(inst["input_ids_1"], dtype = torch.long)
            output["mask_1"] = (output["input_ids_1"] != 0).float()
        output["label"] = torch.tensor(inst["label"], dtype = torch.long)
        return output
    
    def __getitem__(self, i):
        if not self.endless:
            return self.create_inst(self.examples[i])
        
        if self.curr_idx >= len(self.examples):
            random.shuffle(self.examples)
            self.curr_idx = 0
        inst = self.examples[self.curr_idx]
        self.curr_idx += 1
        
        return self.create_inst(inst)
