print("###################################################################################################", flush=True)
print("########################################### LATE FUSION ###########################################", flush=True)
print("###################################################################################################", flush=True)

"""
This script orchestrates the training and evaluation of an ensemble deep learning model for image classification tasks. 
The script is divided into several sections, including imports, settings, utility functions, helper functions, 
training/testing functions, and an orchestration section that coordinates the training flow and evaluations.
"""

################################################################################################
########################################### IMPORTS ############################################
################################################################################################

import os
import shutil
import copy
import random
from glob import glob
import sys
import itertools
import zipfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.init as init

import torchvision.models as models
from torchvision.models.swin_transformer import Swin_V2_T_Weights, Swin_V2_B_Weights
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as TF

from transformers import set_seed, AutoImageProcessor, AutoModelForImageClassification, get_cosine_schedule_with_warmup

from PIL import Image, ImageOps, ImageEnhance

from tqdm import tqdm

import numpy as np

import pytorch_lightning as pl

import inspect

import time

import pickle


################################################################################################
########################################### SETTINGS ###########################################
################################################################################################

"""
This section sets up environment variables, seeds for reproducibility, device configuration for GPU acceleration, 
and hyperparameter definitions for the training process. It also includes options for data selection, data augmentations and freezing schedules.
"""

os.environ["TRANSFORMERS_CACHE"] = "/mnt/data-target-val/transformers_cache"

pl.seed_everything(42, workers=True)

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda:0")

best_accuracy = 0

def get_hyperparameters_ch(model):
    num_epochs = 20
    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-3
    momentum = 0.9
    weight_decay = 0.0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = None
    return num_epochs, loss_function, optimizer, scheduler

def get_hyperparameters_ft(model):
    num_epochs = 20
    loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-5
    weight_decay = 0.01
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_epochs*0.1, num_training_steps=20)
    return num_epochs, loss_function, optimizer, scheduler


ch_data_options = "DepthAnything_Metric" # Options: "Rendered_Depth_Normalized", "ZoeDepth_Normalized", "Rendered_Depth_Metric", "ZoeDepth_Metric", "DepthAnything_Normalized", "DepthAnything_Metric"

ft_data_options = ["DepthAnything_Metric"] # Options: "ZoeDepth_Normalized", "ZoeDepth_Metric", "DepthAnything_Normalized", "DepthAnything_Metric", 'Rendered_Depth_Normalized', 'Rendered_Depth_Metric'

ch_model_options = ["/mnt/models/swinv2_late_fusion_model_ch_customtranforms_metric_depth.pt"] # Options: None, "/mnt/models/swinv2_late_fusion_model_ch_autoimageprocessor_normalized_depth.pt", "/mnt/models/swinv2_late_fusion_model_ch_customtranforms_normalized_depth.pt", "/mnt/models/swinv2_late_fusion_model_ch_customtranforms_metric_depth.pt"

transform_options = [CustomTransforms().synchronized_transform_train_ft_metric] # Options: CustomTransforms().synchronized_transform_train_ft, CustomTransforms().synchronized_transform_train_ft_metric

unfreeze_layers_options = [ 

    [['classifier', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'unfreeze stage 0',],],

    [['classifier', 'global average pooling', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'unfreeze stage 0', 'patch partition and linear embedding'],],

    [['classifier', 'unfreeze stage 3',],
    ['classifier', 'unfreeze stage 3', 'unfreeze stage 2',],
    ['classifier', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1',],
    ['classifier', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'unfreeze stage 0',],],

    [['classifier', 'global average pooling', 'unfreeze stage 3', 'patch partition and linear embedding'],
    ['classifier', 'global average pooling', 'unfreeze stage 3', 'unfreeze stage 2', 'patch partition and linear embedding'],
    ['classifier', 'global average pooling', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'patch partition and linear embedding'],
    ['classifier', 'global average pooling', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'unfreeze stage 0', 'patch partition and linear embedding'],],


]

# Options: [['classifier', 'global average pooling', 'unfreeze stage 3', 'unfreeze stage 2', 'unfreeze stage 1', 'unfreeze stage 0', 'patch partition and linear embedding']],

batch_size_options = [32]

################################################################################################
####################################### UTILITY FUNCTIONS ######################################
################################################################################################

"""
This section comprises various utility functions designed to streamline the data handling and model configuration:
- CustomTransforms: A class that defines various image transformations and augmentations.
- CustomImageDataset: A dataset class for loading and preprocessing image data.
- EnsembleModel: A class that defines the ensemble model architecture.
"""


class CustomTransforms:
    def __init__(self):
        self.size = 192
        self.mean_rgb = [0.5, 0.5, 0.5]
        self.std_rgb = [0.5, 0.5, 0.5]
        self.mean_depth = [0.5, 0.5, 0.5]
        self.std_depth = [0.5, 0.5, 0.5]
        #self.mean_rgb = [0.485, 0.456, 0.406]
        #self.std_rgb = [0.229, 0.224, 0.225]
        #self.mean_depth = [0.485, 0.456, 0.406]
        #self.std_depth = [0.229, 0.224, 0.225]
        self.augmentation_list = [
            self.identity,
            self.autocontrast,
            self.equalize,
            self.rotate,
            self.solarize,
            self.color,
            self.contrast,
            self.brightness,
            self.sharpness
        ]

    def int_parameter(self, level, maxval):
        return int(level * maxval / 10)
    
    def float_parameter(self, level, maxval):
        return float(level * maxval / 10)
    
    def identity(self, img, severity=1):
        return img
    
    def autocontrast(self, img, severity=1):
        return ImageOps.autocontrast(img)
    
    def equalize(self, img, severity=1):
        return ImageOps.equalize(img)
    
    def rotate(self, img, severity=1):
        degrees = self.int_parameter(severity, 30)
        if random.random() < 0.5:
            degrees = -degrees
        return img.rotate(degrees, resample=Image.BILINEAR)
    
    def solarize(self, img, severity=1):
        threshold = 256 - self.int_parameter(severity, 128)
        return ImageOps.solarize(img, threshold)
    
    def color(self, img, severity=1):
        factor = self.float_parameter(severity, 1.8) + 0.1
        return ImageEnhance.Color(img).enhance(factor)
    
    def contrast(self, img, severity=1):
        factor = self.float_parameter(severity, 1.8) + 0.1
        return ImageEnhance.Contrast(img).enhance(factor)
    
    def brightness(self, img, severity=1):
        factor = self.float_parameter(severity, 1.8) + 0.1
        return ImageEnhance.Brightness(img).enhance(factor)
    
    def sharpness(self, img, severity=1):
        factor = self.float_parameter(severity, 1.8) + 0.1
        return ImageEnhance.Sharpness(img).enhance(factor)
    
    def synchronized_augmix(self, rgb_image, depth_image, severity=3, width=3, depth=-1, alpha=1.0):
        ws = np.float32(np.random.dirichlet([alpha] * width))
        m = np.float32(np.random.beta(alpha, alpha))

        rgb_mix = np.zeros_like(np.array(rgb_image), dtype=np.float32)
        depth_mix = np.zeros_like(np.array(depth_image), dtype=np.float32)
    
        for i in range(width):
            rgb_image_aug = rgb_image.copy()
            depth_image_aug = depth_image.copy()

            depth_chain = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(depth_chain):
                op = np.random.choice(self.augmentation_list)
                rgb_image_aug = op(rgb_image_aug, severity)
                depth_image_aug = op(depth_image_aug, severity)
    
            rgb_mix += ws[i] * np.array(rgb_image_aug, dtype=np.float32)
            depth_mix += ws[i] * np.array(depth_image_aug, dtype=np.float32)
    
        rgb_mixed = (1 - m) * np.array(rgb_image, dtype=np.float32) + m * rgb_mix
        depth_mixed = (1 - m) * np.array(depth_image, dtype=np.float32) + m * depth_mix

        return Image.fromarray(np.uint8(rgb_mixed)), Image.fromarray(np.uint8(depth_mixed))

    def random_resized_crop(self, rgb_image, depth_image):
        i, j, h, w = transforms.RandomResizedCrop.get_params(rgb_image, scale=(0.7, 1.0), ratio=(1.0, 1.0))
        rgb_image = TF.resized_crop(rgb_image, i, j, h, w, size=(self.size, self.size), antialias=True)
        depth_image = TF.resized_crop(depth_image, i, j, h, w, size=(self.size, self.size), antialias=True)
        return rgb_image, depth_image 

    def random_horizontal_flip(self, rgb_image, depth_image):
        if random.random() > 0.5:
            rgb_image = TF.hflip(rgb_image)
            depth_image = TF.hflip(depth_image)
        return rgb_image, depth_image

    def to_tensor_and_normalize(self, rgb_image, depth_image):
        rgb_image = TF.to_tensor(rgb_image)
        depth_image = TF.to_tensor(depth_image)
        rgb_image = TF.normalize(rgb_image, mean=self.mean_rgb, std=self.std_rgb)
        depth_image = TF.normalize(depth_image, mean=self.mean_depth, std=self.std_depth)
        return rgb_image, depth_image

    def normalize_tensors(self, rgb_tensor, depth_tensor):
        rgb_tensor = TF.normalize(rgb_tensor, mean=self.mean_rgb, std=self.std_rgb)
        # no need to normalize metric depth data
        #depth_tensor = TF.normalize(depth_tensor, mean=self.mean_depth, std=self.std_depth)
        return rgb_tensor, depth_tensor

    def random_greyscale(self, rgb_image, depth_image):
        if random.random() > 0.1:
            rgb_image = TF.to_grayscale(rgb_image, num_output_channels=3)
            depth_image = TF.to_grayscale(depth_image, num_output_channels=3)
        return rgb_image, depth_image

    def random_greyscale_tensors(self, rgb_tensor, depth_tensor):
        if random.random() > 0.1:
            rgb_tensor = TF.rgb_to_grayscale(rgb_tensor, num_output_channels=3)
            depth_tensor = TF.rgb_to_grayscale(depth_tensor, num_output_channels=3)
        return rgb_tensor, depth_tensor

    def synchronized_transform_train_ch(self, rgb_image, depth_image):
        rgb_image, depth_image = self.random_resized_crop(rgb_image, depth_image)
        rgb_image, depth_image = self.random_horizontal_flip(rgb_image, depth_image)
        # ColorJitter (only for RGB)
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        rgb_image = color_jitter(rgb_image)
        rgb_image, depth_image = self.random_greyscale(rgb_image, depth_image)
        return self.to_tensor_and_normalize(rgb_image, depth_image)

    def synchronized_transform_train_ft(self, rgb_image, depth_image):
        rgb_image, depth_image = self.random_resized_crop(rgb_image, depth_image)
        rgb_image, depth_image = self.random_horizontal_flip(rgb_image, depth_image)
        rgb_image, depth_image = self.synchronized_augmix(rgb_image, depth_image)
        return self.to_tensor_and_normalize(rgb_image, depth_image)

    def synchronized_transform_val(self, rgb_image, depth_image):
        rgb_image = TF.resize(rgb_image, size=(self.size, self.size), antialias=True)
        depth_image = TF.resize(depth_image, size=(self.size, self.size), antialias=True)
        return self.to_tensor_and_normalize(rgb_image, depth_image)

    def synchronized_transform_train_ch_metric(self, rgb_image, depth_tensor):
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        rgb_image = color_jitter(rgb_image)
        rgb_tensor = TF.to_tensor(rgb_image)
        rgb_tensor, depth_tensor = self.random_resized_crop(rgb_tensor, depth_tensor)
        rgb_tensor, depth_tensor = self.random_horizontal_flip(rgb_tensor, depth_tensor)
        rgb_tensor, depth_tensor = self.random_greyscale_tensors(rgb_tensor, depth_tensor)
        return self.normalize_tensors(rgb_tensor, depth_tensor)

    def synchronized_transform_train_ft_metric(self, rgb_image, depth_tensor):
        rgb_tensor = TF.to_tensor(rgb_image)
        rgb_tensor, depth_tensor = self.random_resized_crop(rgb_tensor, depth_tensor)
        rgb_tensor, depth_tensor = self.random_horizontal_flip(rgb_tensor, depth_tensor)
        # Skipping synchronized_augmix as it's complex to adapt for tensors and most of it only applies to rgb data not depth data (colorization etc.)
        return self.normalize_tensors(rgb_tensor, depth_tensor)

    def synchronized_transform_val_metric(self, rgb_image, depth_tensor):
        rgb_tensor = TF.to_tensor(rgb_image)
        rgb_tensor = TF.resize(rgb_tensor, size=(self.size, self.size), antialias=True)
        depth_tensor = TF.resize(depth_tensor, size=(self.size, self.size), antialias=True)
        return self.normalize_tensors(rgb_tensor, depth_tensor)


class CustomImageDataset(Dataset):
    def __init__(self, rgb_folder_path, depth_folder_path, processor=None, transform=None):
        self.rgb_folder_path = rgb_folder_path
        self.depth_folder_path = depth_folder_path
        self.processor = processor
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.label_mapping = {}
        self.reverse_label_mapping = {}

        class_dirs = sorted([d for d in os.listdir(self.rgb_folder_path) if os.path.isdir(os.path.join(self.rgb_folder_path, d))])
        for label, class_dir in enumerate(class_dirs):
            self.label_mapping[class_dir] = label
            self.reverse_label_mapping[label] = class_dir

            rgb_class_path = os.path.join(self.rgb_folder_path, class_dir)
            depth_class_path = os.path.join(self.depth_folder_path, class_dir)
            rgb_filenames = sorted(os.listdir(rgb_class_path))
            depth_filenames = sorted(os.listdir(depth_class_path))

            valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.pt')
            for rgb_filename, depth_filename in zip(os.listdir(rgb_class_path), os.listdir(depth_class_path)):
                if rgb_filename.lower().endswith(valid_extensions) and depth_filename.lower().endswith(valid_extensions):
                   rgb_image_path = os.path.join(rgb_class_path, rgb_filename)
                   depth_image_path = os.path.join(depth_class_path, depth_filename)
                   self.image_paths.append((rgb_image_path, depth_image_path))
                   self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rgb_image_path, depth_image_path = self.image_paths[idx]
        rgb_inputs = Image.open(rgb_image_path).convert('RGB')
        label = self.labels[idx]
        if "/metric-data/" in self.depth_folder_path:
            depth_inputs = torch.load(depth_image_path)
            depth_inputs = depth_inputs.unsqueeze(0)
            depth_inputs = depth_inputs.repeat(3, 1, 1)
            rgb_inputs, depth_inputs = self.transform(rgb_inputs, depth_inputs)
        elif "/rendered-metric-depth-data/" in self.depth_folder_path:
            depth_inputs = torch.load(depth_image_path)
            depth_inputs = depth_inputs.permute(2, 0, 1)
            rgb_inputs, depth_inputs = self.transform(rgb_inputs, depth_inputs)
        elif "/depth-anything-data/DepthAnything/metric_depth/" in self.depth_folder_path:
            depth_inputs = torch.load(depth_image_path)
            depth_inputs = depth_inputs.repeat(3, 1, 1)
            rgb_inputs, depth_inputs = self.transform(rgb_inputs, depth_inputs)
        else: 
            depth_inputs_prelim = Image.open(depth_image_path).convert('L')
            depth_inputs = Image.new('RGB', depth_inputs_prelim.size)
            depth_inputs.paste(depth_inputs_prelim, (0, 0, depth_inputs_prelim.width, depth_inputs_prelim.height))
            if self.transform:
                rgb_inputs, depth_inputs = self.transform(rgb_inputs, depth_inputs)
            if self.processor:
                rgb_inputs = self.processor(images=rgb_inputs, return_tensors="pt", do_normalize=True, do_rescale=True, do_resize=True)
                depth_inputs = self.processor(images=depth_inputs, return_tensors="pt", do_normalize=True, do_rescale=True, do_resize=True)
                rgb_inputs = {'pixel_values': rgb_inputs['pixel_values'][0], 'labels': torch.tensor(label)}
                depth_inputs = {'pixel_values': depth_inputs['pixel_values'][0], 'labels': torch.tensor(label)}
                return rgb_inputs, depth_inputs
        rgb_inputs = {'pixel_values': rgb_inputs, 'labels': torch.tensor(label)}
        depth_inputs = {'pixel_values': depth_inputs, 'labels': torch.tensor(label)}
        return rgb_inputs, depth_inputs

    def get_class_name(self, label):
        return self.reverse_label_mapping[label]


class EnsembleModel(nn.Module):
    def __init__(self, model_rgb, model_depth, model_rgbd_classifier):
        super(EnsembleModel, self).__init__()

        self.model_rgb = model_rgb
        self.model_depth = model_depth
        self.model_rgbd_classifier = model_rgbd_classifier
        self.model_rgbd_classifier.classifier = nn.Linear(in_features=1024*2, out_features=102, bias=True)
        self.model_rgbd_classifier = list(self.model_rgbd_classifier.children())[-1]
        self.hook_output_rgb = None
        self.hook_output_depth = None
        self.model_rgb.swinv2.pooler.register_forward_hook(self.hook_fn_rgb)
        self.model_depth.swinv2.pooler.register_forward_hook(self.hook_fn_depth)

    def hook_fn_rgb(self, module, input, output):
        self.hook_output_rgb = output

    def hook_fn_depth(self, module, input, output):
        self.hook_output_depth = output

    def forward(self, pixel_values_rgb, pixel_values_depth):
        _ = self.model_rgb(pixel_values=pixel_values_rgb)
        _ = self.model_depth(pixel_values=pixel_values_depth)
        x1 = self.hook_output_rgb.squeeze(-1)
        x2 = self.hook_output_depth.squeeze(-1)
        x = torch.cat((x1, x2), dim=1)
        out = self.model_rgbd_classifier(x)
        return out

################################################################################################
####################################### HELPER FUNCTIONS #######################################
################################################################################################

"""
This section contains a collection of helper functions for various tasks such as dataset preparation, DataLoader instantiation, 
model layer freezing/unfreezing, calculation of evaluation metrics, printing training configurations 
and results, and saving/loading model states.

Key Functions:
- generate_datasets: Prepares train, validation, and test datasets based on specified configurations.
- generate_data_loaders: Creates DataLoader instances for the datasets to enable batch processing.
- calc_balanced_acc and calc_f1_macro: Compute evaluation metrics such as balanced accuracy and F1 macro score.
- freeze_all_layers and unfreeze_classifier (among others): Manipulate the model's layers' training state.
- print_configurations_CH and print_configurations_FT: Display the current configuration settings for training.
- print_hyperparameters: Prints the hyperparameters used for the current training session.
- print_train_test_results: Outputs the training and testing results to the console.
- save_ensemble_model and load_ensemble_model: Handle saving and loading of the trained model states.
"""

def generate_datasets(data, train_scheme, transform_train):
    rgb_folder_base = '/mnt/data-target-val/data_rgb/'

    if data == "ZoeDepth_Metric":
        depth_folder_base_train = depth_folder_base_val = depth_folder_base_test = '/mnt/metric-data/'
        processor = None
        if train_scheme == "CH":
            transform_train = CustomTransforms().synchronized_transform_train_ch_metric
        elif train_scheme == "FT":
            transform_train = CustomTransforms().synchronized_transform_train_ft_metric
        transform_val = CustomTransforms().synchronized_transform_val_metric
    elif data == "Rendered_Depth_Metric":
        depth_folder_base_train = '/mnt/rendered-metric-depth-data/'
        depth_folder_base_val = '/mnt/metric-data/'
        depth_folder_base_test = '/mnt/metric-data/'
        processor = None
        if train_scheme == "CH":
            transform_train = CustomTransforms().synchronized_transform_train_ch_metric
        elif train_scheme == "FT":
            transform_train = CustomTransforms().synchronized_transform_train_ft_metric
        transform_val = CustomTransforms().synchronized_transform_val_metric
    elif data == "DepthAnything_Metric":
        depth_folder_base_train = depth_folder_base_val = depth_folder_base_test = '/mnt/depth-anything-data/DepthAnything/metric_depth/'
        processor = None
        if train_scheme == "CH":
            print("TRAIN SCHEME IS CH")
            transform_train = CustomTransforms().synchronized_transform_train_ch_metric
        elif train_scheme == "FT":
            print("TRAIN SCHEME IS FT")
            transform_train = CustomTransforms().synchronized_transform_train_ft_metric
        transform_val = CustomTransforms().synchronized_transform_val_metric
    elif data == "Rendered_Depth_Normalized":
        depth_folder_base_train = depth_folder_base_val = depth_folder_base_test = '/mnt/rendered-depth-data/data_depth/' 
        processor = None if transform_train else AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
        transform_val = CustomTransforms().synchronized_transform_val if transform_train else None
    elif data == "ZoeDepth_Normalized":
        depth_folder_base_train = depth_folder_base_val = depth_folder_base_test = '/mnt/data-target-val/data_depth/'
        processor = None if transform_train else AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
        transform_val = CustomTransforms().synchronized_transform_val if transform_train else None
    elif data == "DepthAnything_Normalized":
        depth_folder_base_train = depth_folder_base_val = depth_folder_base_test = '/mnt/depth-anything-data/DepthAnything/normalized_depth/'
        processor = None if transform_train else AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
        transform_val = CustomTransforms().synchronized_transform_val if transform_train else None
    else: 
        print("Error: Selected Data is Unknown. Try ZoeDepth_Metric, Rendered_Depth_Metric, Rendered_Depth_Normalized or ZoeDepth_Normalized")
    
    train_dataset = CustomImageDataset(rgb_folder_path=rgb_folder_base + 'train', depth_folder_path=depth_folder_base_train + 'train', processor=processor, transform=transform_train)
    val_dataset = CustomImageDataset(rgb_folder_path=rgb_folder_base + 'val', depth_folder_path=depth_folder_base_val + 'val', processor=processor, transform=transform_val)
    ##val_dataset = CustomImageDataset(rgb_folder_path=rgb_folder_base + 'test', depth_folder_path=depth_folder_base + 'test', processor=processor, transform=transform_val)
    test_dataset = CustomImageDataset(rgb_folder_path=rgb_folder_base + 'test', depth_folder_path=depth_folder_base_test + 'test', processor=processor, transform=transform_val)
	
    return train_dataset, val_dataset, test_dataset

def generate_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def calc_balanced_acc(predictions, all_labels):
    num_classes = len(torch.unique(all_labels))
    acc_per_class = []
    for cls in range(num_classes):
        cls_mask = (all_labels == cls)
        cls_correct = torch.sum(predictions[cls_mask] == all_labels[cls_mask]).item()
        cls_total = torch.sum(cls_mask).item()
        if cls_total != 0:
            cls_acc = cls_correct / cls_total
            acc_per_class.append(cls_acc)
    balanced_acc = sum(acc_per_class) / len(acc_per_class) if acc_per_class else 0
    return 100 * balanced_acc

def calc_f1_macro(predictions, all_labels):
    num_classes = len(torch.unique(all_labels))
    f1_scores = []
    for i in range(num_classes):
        tp = ((predictions == i) & (all_labels == i)).sum().item()
        fp = ((predictions == i) & (all_labels != i)).sum().item()
        fn = ((predictions != i) & (all_labels == i)).sum().item()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_scores.append(f1)
    f1_macro = sum(f1_scores) / len(f1_scores)
    return 100 * f1_macro

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze_classifier(model):
    for param in model.model_rgbd_classifier.parameters():
        param.requires_grad = True
    return model

def unfreeze_global_average_pooling_layer(model):
    for param in model.model_rgb.swinv2.layernorm.parameters():
        param.requires_grad = True
    for param in model.model_depth.swinv2.layernorm.parameters():
        param.requires_grad = True
    return model

def unfreeze_stage_3(model):
    stage_rgb = model.model_rgb.swinv2.encoder.layers[3]
    for param in stage_rgb.parameters():
        param.requires_grad = True
    stage_depth = model.model_depth.swinv2.encoder.layers[3]
    for param in stage_depth.parameters():
        param.requires_grad = True    
    return model

def unfreeze_stage_2(model):
    stage_rgb = model.model_rgb.swinv2.encoder.layers[2]
    for param in stage_rgb.parameters():
        param.requires_grad = True
    stage_depth = model.model_depth.swinv2.encoder.layers[2]
    for param in stage_depth.parameters():
        param.requires_grad = True    
    return model

def unfreeze_stage_1(model):
    stage_rgb = model.model_rgb.swinv2.encoder.layers[1]
    for param in stage_rgb.parameters():
        param.requires_grad = True
    stage_depth = model.model_depth.swinv2.encoder.layers[1]
    for param in stage_depth.parameters():
        param.requires_grad = True    
    return model

def unfreeze_stage_0(model):
    stage_rgb = model.model_rgb.swinv2.encoder.layers[0]
    for param in stage_rgb.parameters():
        param.requires_grad = True
    stage_depth = model.model_depth.swinv2.encoder.layers[0]
    for param in stage_depth.parameters():
        param.requires_grad = True    
    return model

def unfreeze_patch_partition_and_linear_embedding_layers(model):
    for param in model.model_rgb.swinv2.embeddings.parameters():
        param.requires_grad = True
    for param in model.model_depth.swinv2.embeddings.parameters():
        param.requires_grad = True
    return model

unfreeze_layer_config_functions = {
	'classifier': unfreeze_classifier,
    'global average pooling': unfreeze_global_average_pooling_layer,
    'unfreeze stage 3': unfreeze_stage_3,
    'unfreeze stage 2': unfreeze_stage_2,
    'unfreeze stage 1': unfreeze_stage_1,
    'unfreeze stage 0': unfreeze_stage_0,
    'patch partition and linear embedding': unfreeze_patch_partition_and_linear_embedding_layers, 
    }

def freeze_specified_layers(model, unfreeze_layers_options):
    for config_name in unfreeze_layers_options:
        if config_name in unfreeze_layer_config_functions:
            unfreeze_function = unfreeze_layer_config_functions[config_name]
            model = unfreeze_function(model)
        else:
            raise ValueError(f"Configuration '{config_name}' is not defined.")
    return model

def print_configurations_CH(data, processor, transform_train, transform_val, batch_size):
    print("*"*30, flush=True)
    print("Classification Head Training Configurations", flush=True)
    print("Depth Data: " + str(data), flush=True)
    print("Data Augmentations: " + ("AutoImageProcessor" if processor else "CustomTransforms_CH"), flush=True)
    print("Batch Size: " +str(batch_size), flush=True)
    print("*"*30, flush=True)

def print_configurations_FT(data, ch_model_path, transform_train, unfreeze_layers, batch_size):
    print("*"*30, flush=True)
    print("Full Tuning Configurations:", flush=True)
    print("Depth Data: " + str(data), flush=True)
    training_scheme = "CH-FT" if ch_model_path else "FT"
    print("Training Scheme: " + str(training_scheme), flush=True)
    ch_base_augmentations = "AutoImageProcessor" if ch_model_path and "autoimageprocessor" in ch_model_path else "CustomTransforms_CH"
    print("CH Base Data Augmentations: " + str(ch_base_augmentations), flush=True)
    ft_augmentations = "CustomTransforms_FT" if transform_train else "AutoImageProcessor"
    print("Full Tuning Augmentations: " + str(ft_augmentations), flush=True)
    print("Freezing Schedule:", flush=True)
    print(unfreeze_layers, flush=True)
    print("Batch Size: " +str(batch_size), flush=True)
    print("*"*30, flush=True)

def print_hyperparameters(num_epochs, loss_function, optimizer, scheduler):
    print("*"*30, flush=True)
    print("Hyperparameters:", flush=True)
    print("Number of Epochs: " + str(num_epochs), flush=True)
    print("Loss Function: " + str(loss_function.__class__.__name__), flush=True)
    if optimizer:
        print("Optimizer: " + str(optimizer.__class__.__name__), flush=True)
        print("  Learning Rate: " + str(optimizer.defaults.get('lr', 'Not set')), flush=True)
        if 'momentum' in optimizer.defaults:
            print("  Momentum: " + str(optimizer.defaults['momentum']), flush=True)
        if 'weight_decay' in optimizer.defaults:
            print("  Weight Decay: " + str(optimizer.defaults['weight_decay']), flush=True)
    else:
        print("Optimizer: None", flush=True)
    print("Scheduler: " + str(scheduler.__class__.__name__) if scheduler else 'Scheduler: None', flush=True)
    print("*"*30, flush=True)

def print_train_test_results(val_loss_array, val_acc_array, test_accuracy, test_balanced_accuracy, test_f1_macro, run):
    min_loss = min(val_loss_array)
    min_loss_epoch = val_loss_array.index(min_loss)
    min_loss_accuracy = val_acc_array[min_loss_epoch]
    print(f"\nResults ({run}):", flush=True)
    print("\tMin val loss {:.4f} was achieved during epoch #{}".format(min_loss, min_loss_epoch + 1), flush=True)
    print("\tVal Accuracy during min val loss is {:.4f}".format(min_loss_accuracy), flush=True)
    print(f"Test Accuracy: {test_accuracy:.2f}%", flush=True)
    print(f"Test Balanced Accuracy: {test_balanced_accuracy:.2f}%", flush=True)
    print(f"Test F1-Macro: {test_f1_macro:.2f}%", flush=True)

def print_phase_info(phase, layers_unfrozen):
    print("*"*30, flush=True)
    print(f"Phase: {phase+1}\nUnfrozen Layers: {layers_unfrozen}", flush=True)
    print("*"*30, flush=True)

def save_ensemble_model(ensemble_model, save_path):
    model_save_dict = {
        'model_rgb_state_dict': ensemble_model.model_rgb.state_dict(),
        'model_depth_state_dict': ensemble_model.model_depth.state_dict(),
        'model_rgbd_classifier_state_dict': ensemble_model.model_rgbd_classifier.state_dict(),
    }
    torch.save(model_save_dict, save_path)

def save_test_results(all_outputs, all_labels, test_accuracy, balanced_acc, f1_macro):
    data_to_save = {
        'all_outputs': all_outputs.cpu().numpy(),
        'all_labels': all_labels.cpu().numpy(),
        'accuracy': test_accuracy,
        'balanced_acc': balanced_acc,
        'f1_macro': f1_macro,
    }
    with open('/mnt/models/best_model_test_results.pkl', 'wb') as file:
        pickle.dump(data_to_save, file)

def save_training_progress_data(train_loss_array, train_acc_array, val_loss_array, val_acc_array, training_run_title):
    loss_data = {'train_loss': train_loss_array, 'train_acc': train_acc_array, 'val_loss': val_loss_array, 'val_acc': val_acc_array}
    with open('/mnt/models/training_progress_' + str(training_run_title) + '.pkl', 'wb') as f:
        pickle.dump(loss_data, f)

def load_ensemble_model(save_path=None):
    base_model_swin_rgb = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)
    base_model_swin_depth = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)
    classifier_model_swin_rgbd = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)
    ensemble_model = EnsembleModel(base_model_swin_rgb, base_model_swin_depth, classifier_model_swin_rgbd)
    if save_path:
        model_save_dict = torch.load(save_path)
        ensemble_model.model_rgb.load_state_dict(model_save_dict['model_rgb_state_dict'])
        ensemble_model.model_depth.load_state_dict(model_save_dict['model_depth_state_dict'])
        ensemble_model.model_rgbd_classifier.load_state_dict(model_save_dict['model_rgbd_classifier_state_dict'])

    ensemble_model = ensemble_model.to(DEVICE)
    return ensemble_model

def dataset_ready(path, flag_file_name, timeout=6000):
    start_time = time.time()
    flag_file_path = os.path.join(path, flag_file_name)
    while time.time() - start_time < timeout:
        if os.path.exists(flag_file_path):
            return True
        time.sleep(60)  # Wait for 60 seconds before checking again
    return False

def count_image_files(directory):
    image_file_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt') or file.endswith('.png'):
                image_file_count += 1
    return image_file_count        

################################################################################################
################################## TRAINING/TESTING FUNCTIONS ##################################
################################################################################################

"""
Defines functions for training and evaluating models:
- training: Function to train the model on training data and evaluate on validation data.
- test_model: Function to evaluate the model on test data.
"""

def training(model, model_name, num_epochs, loss_function, optimizer, scheduler, train_dataloader, val_dataloader, initial_val_loss=np.inf):

    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
    lowest_val_loss = initial_val_loss
    best_model = None

    for epoch in tqdm(range(num_epochs), file=sys.stdout, dynamic_ncols=True):

        if scheduler:
            print('Epoch: {} | Learning rate: {}'.format(epoch + 1, scheduler.get_last_lr()), flush=True)
        else:
            print('Epoch: {}'.format(epoch + 1), flush=True)

        for phase in ['train', 'val']:

            epoch_loss = 0
            epoch_correct_items = 0
            epoch_items = 0

            if phase == 'train':
                model.train()
                with torch.enable_grad():
                    for (batch_rgb, batch_depth) in train_dataloader:
                        inputs_rgb = batch_rgb['pixel_values'].to(DEVICE)
                        inputs_depth = batch_depth['pixel_values'].to(DEVICE)
                        targets = batch_rgb['labels'].to(DEVICE)

                        optimizer.zero_grad()
                        outputs = model(pixel_values_rgb=inputs_rgb, pixel_values_depth=inputs_depth)
                        loss = loss_function(outputs, targets)

                        loss.backward()
                        optimizer.step()

                        if scheduler:
                            scheduler.step()

                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets).float().sum()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items.item()
                        epoch_items += len(targets)

                train_loss_array.append(epoch_loss / epoch_items)
                train_acc_array.append(epoch_correct_items / epoch_items)

            elif phase == 'val':
                model.eval()
                with torch.no_grad():
                    for (batch_rgb, batch_depth) in val_dataloader:
                        inputs_rgb = batch_rgb['pixel_values'].to(DEVICE)
                        inputs_depth = batch_depth['pixel_values'].to(DEVICE)
                        targets = batch_rgb['labels'].to(DEVICE)

                        outputs = model(pixel_values_rgb=inputs_rgb, pixel_values_depth=inputs_depth)
                        loss = loss_function(outputs, targets)

                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets).float().sum()

                        epoch_loss += loss.item()
                        epoch_correct_items += correct_items.item()
                        epoch_items += len(targets)

                val_loss_array.append(epoch_loss / epoch_items)
                val_acc_array.append(epoch_correct_items / epoch_items)

                if epoch_loss / epoch_items < lowest_val_loss:
                    lowest_val_loss = epoch_loss / epoch_items
                    print("\t| New lowest val loss for {}: {}".format(model_name, lowest_val_loss), flush=True)
                    best_model_run = copy.deepcopy(model)

                print("\t| Val accuracy for {}: {:.4f}".format(model_name, epoch_correct_items / epoch_items), flush=True)

    return best_model_run, train_loss_array, train_acc_array, val_loss_array, val_acc_array


def test_model(model, test_dataloader):
    model.eval()
    all_outputs = []
    all_labels = []
    global best_accuracy

    with torch.no_grad():
        for (batch_rgb, batch_depth) in test_dataloader:
            inputs_rgb = batch_rgb['pixel_values'].to(DEVICE)
            inputs_depth = batch_depth['pixel_values'].to(DEVICE)
            targets = batch_rgb['labels'].to(DEVICE)

            outputs = model(pixel_values_rgb=inputs_rgb, pixel_values_depth=inputs_depth)

            all_outputs.append(outputs)
            all_labels.append(targets)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    _, predictions = all_outputs.max(1)
    correct = (predictions == all_labels).sum().item()
    total = all_labels.size(0)
    accuracy = 100 * correct / total

    balanced_acc = calc_balanced_acc(predictions, all_labels)
    f1_macro =  calc_f1_macro(predictions, all_labels)

    if accuracy > best_accuracy:
        best_accuracy = accuracy 
        best_model_overall = copy.deepcopy(model)

        save_ensemble_model(best_model_overall, "/mnt/models/best_model.pt")
        save_test_results(all_outputs, all_labels, best_accuracy, balanced_acc, f1_macro)

    return all_outputs, all_labels, accuracy, balanced_acc, f1_macro


################################################################################################
###################################### PIPELINE FUNCTIONS ######################################
################################################################################################

"""
Consists of higher-level functions that orchestrate the training and evaluation process:
- run_ch_training: Function to run training for the classification head.
- run_ch_ft_training_with_config: Function to run full training with specific configurations.
"""

def run_ch_training(data="ZodeDepth_Normalized", processor=None, transform_train=None, transform_val=None, batch_size=32):
    
    pl.seed_everything(42, workers=True)

    print_configurations_CH(data, processor, transform_train, transform_val, batch_size)
    training_run_title = "CH_AutoImageProcessor" if processor else "CH_CustomTransforms"

    train_dataset, val_dataset, test_dataset = generate_datasets(data, "CH", transform_train)
    train_loader, val_loader, test_loader = generate_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

    base_model_swin_rgb = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)
    base_model_swin_depth = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)
    classifier_model_swin_rgbd = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k", num_labels=102, ignore_mismatched_sizes=True)
    swinv2_late_fusion_model_ch_training = EnsembleModel(base_model_swin_rgb, base_model_swin_depth, classifier_model_swin_rgbd)

    swinv2_late_fusion_model_ch_training = freeze_all_layers(swinv2_late_fusion_model_ch_training)
    swinv2_late_fusion_model_ch_training = unfreeze_classifier(swinv2_late_fusion_model_ch_training)
    swinv2_late_fusion_model_ch_training = swinv2_late_fusion_model_ch_training.to(DEVICE)

    num_epochs_ch, loss_function_ch, optimizer_ch, scheduler_ch = get_hyperparameters_ch(swinv2_late_fusion_model_ch_training)

    print_hyperparameters(num_epochs_ch, loss_function_ch, optimizer_ch, scheduler_ch)

    swinv2_late_fusion_model_ch_training_results = training(model=swinv2_late_fusion_model_ch_training,
	                                     model_name='swinv2_late_fusion_model_ch',
	                                     num_epochs=num_epochs_ch,
	                                     loss_function = loss_function_ch,
	                                     optimizer = optimizer_ch,
	                                     scheduler = scheduler_ch,
	                                     train_dataloader=train_loader,
	                                     val_dataloader=val_loader)

    swinv2_late_fusion_model_ch, train_loss_array, train_acc_array, val_loss_array, val_acc_array = swinv2_late_fusion_model_ch_training_results
    save_training_progress_data(train_loss_array, train_acc_array, val_loss_array, val_acc_array, training_run_title)
    all_outputs, all_labels, test_accuracy, balanced_acc, f1_macro = test_model(swinv2_late_fusion_model_ch, test_loader)
    print_train_test_results(val_loss_array, val_acc_array, test_accuracy, balanced_acc, f1_macro, "CH")

    return swinv2_late_fusion_model_ch

def run_ch_ft_training_with_config(config):

    pl.seed_everything(42, workers=True)

    data, ch_model_path, transform_train, unfreeze_layers, batch_size = config
    print_configurations_FT(data, ch_model_path, transform_train, unfreeze_layers, batch_size)

    ft_model_training = load_ensemble_model(ch_model_path) if ch_model_path else load_ensemble_model()

    train_dataset, val_dataset, test_dataset = generate_datasets(data, "FT", transform_train)
    train_loader, val_loader, test_loader = generate_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

    for phase, layers_unfrozen in enumerate(unfreeze_layers):
        print_phase_info(phase, layers_unfrozen)

        ft_model_training = freeze_all_layers(ft_model_training)
        ft_model_training = freeze_specified_layers(ft_model_training, layers_unfrozen)
        ft_model_training = ft_model_training.to(DEVICE)

        num_epochs_ft, loss_function_ft, optimizer_ft, scheduler_ft = get_hyperparameters_ft(ft_model_training)
        print_hyperparameters(num_epochs_ft, loss_function_ft, optimizer_ft, scheduler_ft)


        swinv2_late_fusion_model_ch_ft_training_results = training(model=ft_model_training,
	                                     model_name='swinv2_late_fusion_model_ch_ft',
	                                     num_epochs=num_epochs_ft,
	                                     loss_function = loss_function_ft,
	                                     optimizer = optimizer_ft,
	                                     scheduler = scheduler_ft,
	                                     train_dataloader=train_loader,
	                                     val_dataloader=val_loader,)

        ft_model_training, train_loss_array, train_acc_array, val_loss_array, val_acc_array = swinv2_late_fusion_model_ch_ft_training_results
        save_training_progress_data(train_loss_array, train_acc_array, val_loss_array, val_acc_array, str(phase))
        all_outputs, all_labels, test_accuracy, balanced_acc, f1_macro = test_model(ft_model_training, test_loader)
        print_train_test_results(val_loss_array, val_acc_array, test_accuracy, balanced_acc, f1_macro, f"CH-FT ({phase+1})")

    return None

###################################################################################################
####################################### ORCHESTRATION CODE ########################################
###################################################################################################

"""
The final section of the script that executes the training and evaluation pipeline. 
It checks data availability, performs classification head training, conducts a grid search for full tuning, and finally, waits for model download post-training.
It utilizes the previously defined settings and functions to systematically approach the training process, evaluate performance, and save the best models.
"""

############################################# Data (Sanity) Check ##########################################

models_directory = "/mnt/models"
if os.path.exists(models_directory):
    for filename in os.listdir(models_directory):
        file_path = os.path.join(models_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

directories_to_check = ['/mnt/metric-data/val', '/mnt/metric-data/test', '/mnt/metric-data/train',
    '/mnt/rendered-metric-depth-data/train',
    '/mnt/depth-anything-data/DepthAnything/metric_depth/val', '/mnt/depth-anything-data/DepthAnything/metric_depth/test', '/mnt/depth-anything-data/DepthAnything/metric_depth/train',
    '/mnt/depth-anything-data/DepthAnything/normalized_depth/val', '/mnt/depth-anything-data/DepthAnything/normalized_depth/test', '/mnt/depth-anything-data/DepthAnything/normalized_depth/train',
    '/mnt/rendered-depth-data/data_depth/val', '/mnt/rendered-depth-data/data_depth/test', '/mnt/rendered-depth-data/data_depth/train',
    '/mnt/data-target-val/data_depth/val', '/mnt/data-target-val/data_depth/test', '/mnt/data-target-val/data_depth/train',
    '/mnt/data-target-val/data_rgb/val', '/mnt/data-target-val/data_rgb/test', '/mnt/data-target-val/data_rgb/train',
    ]

for directory in directories_to_check:
    count = count_image_files(directory)
    print(f"Number of images in {directory}: {count}")


#################################### Classification Head Training ################################


custom_transforms = CustomTransforms()
transform_train = custom_transforms.synchronized_transform_train_ch_metric if ch_data_options == "ZoeDepth_Metric" or ch_data_options == "Rendered_Depth_Metric" or ch_data_options == 'DepthAnything_Metric' else custom_transforms.synchronized_transform_train_ch
transform_val = custom_transforms.synchronized_transform_val_metric if ch_data_options == "ZoeDepth_Metric" or ch_data_options == "Rendered_Depth_Metric" or ch_data_options == 'DepthAnything_Metric' else custom_transforms.synchronized_transform_val
swinv2_late_fusion_model_ch_synchronized_tranforms = run_ch_training(data=ch_data_options,
                                                                    processor=None,
                                                                    transform_train=transform_train,
                                                                    transform_val=transform_val,
                                                                    batch_size=32)
model_name = "CustomTransforms Metric Data" if ch_data_options == "ZoeDepth_Metric" or ch_data_options == "Rendered_Depth_Metric" or ch_data_options == 'DepthAnything_Metric' else "CustomTransforms Normalized Data"
swinv2_late_fusion_model_ch_synchronized_tranforms.model_name = model_name
model_path = f"/mnt/models/swinv2_late_fusion_model_ch_customtranforms_{'metric' if ch_data_options == 'ZoeDepth_Metric' or ch_data_options == 'Rendered_Depth_Metric' or ch_data_options == 'DepthAnything_Metric' else 'normalized'}_depth.pt"
save_ensemble_model(swinv2_late_fusion_model_ch_synchronized_tranforms, model_path)

if ch_data_options not in ("ZoeDepth_Metric", "Rendered_Depth_Metric", "DepthAnything_Metric"):
    swinv2_late_fusion_model_ch_huggingface_processor = run_ch_training(data=ch_data_options,
                                                                    processor=AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k"),
                                                                    transform_train=None,
                                                                    transform_val=None,
                                                                    batch_size=32)
    swinv2_late_fusion_model_ch_huggingface_processor.model_name = "AutoImageProcessor"
    save_ensemble_model(swinv2_late_fusion_model_ch_huggingface_processor, "/mnt/models/swinv2_late_fusion_model_ch_autoimageprocessor_normalized_depth.pt")


################################## Full Tuning (GridSearch) ###################################

all_configs = list(itertools.product(ft_data_options, ch_model_options, transform_options, unfreeze_layers_options, batch_size_options))

for config in all_configs:
    run_ch_ft_training_with_config(config)

print("###################################################################################################", flush=True)
print("###################################### TRAINING COMPLETED #########################################", flush=True)
print("###################################################################################################", flush=True)
print("################################## WAITING FOR MODEL DOWNLOAD #####################################", flush=True)
print("###################################################################################################", flush=True)

time.sleep(86400)  # Wait for 24 hours before completing the pod

shutil.rmtree("/mnt/data-target-val/transformers_cache", ignore_errors=True)

print("Cache deleted", flush=True)

print("Pod Completed. Terminating Pod.", flush=True)

exit(0)
