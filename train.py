import torch
import torch.nn.functional as F
import numpy as np
from utils.processing import BoundingBox
import cv2


def train(model, train_loader, optimizer, criterion, epoch, device, log_interval=175):
    """ Function to train the model
    Args:
        model (nn.model object): Model to be trained
        train_loader (utils.dataloader object): Dataloader for training data
        optimizer (nn.optim object): Optimizer to be used
        criterion (nn.loss object): loss object to calculate MSE loss
        epoch (int): The current epoch
        device (torch.device object): device to load data on
        log_interval (int): interval at which to print batch metrics [Default: 175]
    
    Return:
        train_loss (double): Training loss over one epoch        
    """
    model.train()
    train_loss = 0.0
    prev_frame = None
    for batchIdx, (data) in enumerate(train_loader):
        data['image'], data['bbox'] = data['image'].to(
            device), data['bbox'].to(device)
        b, _, h, w = data['image'].shape
        c = 1
        for idx in range(b):
            bounding_box = BoundingBox(device)
            heat_map, _ = bounding_box.pre_process(
                data['bbox'][idx], (c, h, w), (c, int(h/4), int(w/4)))
            if idx == 0:
                ground_truth = heat_map[None, ...]
            else:
                ground_truth = torch.cat((ground_truth, heat_map[None, ...]))
        optimizer.zero_grad()
        output, prev_frame = model(data['image'], prev_frame)
        loss = criterion(ground_truth, output)
        train_loss += loss
        loss.backward()
        optimizer.step()
        if batchIdx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch+1, batchIdx * train_loader.batch_size, len(
                train_loader.dataset), 100. * batchIdx / len(train_loader), loss.item()/b))
    train_loss /= len(train_loader.dataset)
    return train_loss.item()


def __freeze_SweatyNet__(model, requires_grad=False):
    """
    Function to freeze/unfreeze weights of the SweatyNet part of the model
    Args:
        model (nn.model object): Model for which weights are to be frozen or unfrozen
        requires_grad (boolean): parameter to set for weights. If False, weights are frozen [Default: False]
    """
    for idx, (child) in enumerate(model.children()):
        if idx < 11:
            for param in child.parameters():
                param.requires_grad = requires_grad
