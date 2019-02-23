import torch
import torch.nn.functional as F
import numpy as np
from utils.processing import BoundingBox


def train(model, train_loader, optimizer, criterion, epoch, device, log_interval=125):
    model.train()
    train_loss = 0.0
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
        output = model(data['image'])
        # _, output = model(data['image'])
        loss = torch.sqrt(criterion(ground_truth*100, output))
        train_loss += loss
        loss.backward()
        optimizer.step()
        if batchIdx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch+1, batchIdx * train_loader.batch_size, len(
                train_loader.dataset), 100. * batchIdx / len(train_loader), loss.item()/b))
    train_loss /= len(train_loader.dataset)
    return train_loss.item()


def validate(model, val_loader, optimizer, criterion, device):
    model.eval()
    confusion_matrix = np.zeros((2, 2))
    val_loss = 0.0
    with torch.no_grad():
        for batchIdx, (data) in enumerate(val_loader):
            data['image'], data['bbox'] = data['image'].to(
                device), data['bbox'].to(device)
            b, _, h, w = data['image'].shape
            c = 1
            bounding_box = BoundingBox(device)
            for idx in range(b):
                heat_map, bbox = bounding_box.pre_process(
                    data['bbox'][idx], (c, h, w), (c, int(h/4), int(w/4)))
                if idx == 0:
                    ground_truth = heat_map[None, ...]
                    gt_bbox = bbox[None, ...]
                else:
                    ground_truth = torch.cat(
                        (ground_truth, heat_map[None, ...]))
                    gt_bbox = torch.cat((gt_bbox, bbox[None, ...]))
            output = model(data['image'])
            # snet_output, output = model(data['image'])
            val_loss += torch.sqrt(criterion(ground_truth*100, output))

            for idx in range(b):
                bounding_box = BoundingBox(device)
                conf_mat, _ = bounding_box.post_process(
                    output[idx], gt_bbox[idx])
                confusion_matrix += conf_mat
                if batchIdx == 0:
                    orig_image = data['image'][idx][None, ...].clone().to(
                        device)
                    orig_image = F.interpolate(orig_image, size=(
                        [output.shape[2], output.shape[3]]), mode='bilinear', align_corners=True)
                    gt_mask = torch.cat((ground_truth[idx], torch.ones(2, output.shape[2], output.shape[3]).to(device)))
                    # pred = (output[idx]/torch.max(output[idx]))[None, ...]
                    # snet_pred = torch.clamp(snet_output[idx], 0, 1)
                    # snet_pred = torch.cat((snet_pred, torch.ones(2, output.shape[2], output.shape[3]).to(device)))
                    pred = torch.clamp(output[idx], 0, 1)
                    pred = torch.cat((pred, torch.ones(2, output.shape[2], output.shape[3]).to(device)))
                    grid_row = torch.cat(
                        (orig_image, gt_mask[None, ...], pred[None, ...]))
                    # grid_row = torch.cat(
                    #     (orig_image, gt_mask[None, ...], snet_pred[None, ...], pred[None, ...]))
                    if idx == 0:
                        grid = grid_row
                    else:
                        grid = torch.cat((grid, grid_row))

    val_loss /= len(val_loader.dataset)
    return val_loss.item(), confusion_matrix, grid
