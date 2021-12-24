import os
import sys
import yaml
import torch 
import shutil
import torchvision
import numpy as np
from typing import Union
from datetime import datetime
import matplotlib.pyplot as plt


def get_configuration_file_and_get_project_path() -> Union[dict, str]:
    python_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configuration_path = python_path + '/config/config.yml'
    if  not os.path.isfile(configuration_path):
        print('config.yml  file not found!!! \nPlease place the specified name file in ' + str(configuration_path))
        sys.exit(2)
    with open(configuration_path, 'r', encoding='utf-8') as configuration_yml:
        return yaml.safe_load(configuration_yml), python_path


def log_training_files_and_get_log_path(configuration_dict: dict, python_path: str) -> str:
    logs_path = os.path.dirname(python_path) + '/logs'    
    if not os.path.isdir(logs_path): os.mkdir(logs_path)
    model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + configuration_dict['model_name'] 
    save_train_path = logs_path + '/' + model_name
    if not os.path.isdir(save_train_path): os.mkdir(save_train_path)
    shutil.copytree(python_path, save_train_path + '/' + 'python_files_copy')
    return save_train_path


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(loader: torch, model: torch, device: str = 'cuda', mode: bool = True, iterations: int = 20):
    with torch.no_grad():
        data = next(iter(loader))[0].to(device)
        input_tensor = data.cpu()
        if mode == 0:
            transformed_input_tensor = model.spatial_transformer_net(data).cpu()
        elif mode == 1:
            transformed_input_tensor = model.sequential_spatial_transformer_net(data).cpu()
        else:
            transformed_input_tensor, _, _, _ = model.reinforced_spatial_transformer_net(data)
            transformed_input_tensor = transformed_input_tensor.cpu()
        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))
        _, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

