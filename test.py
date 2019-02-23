import torch
import torch.nn.functional as F
import numpy as np
from utils.processing import BoundingBox
import imageio
import cv2


def test_model(model, test_loader, optimizer, criterion, device, output_vid='output.mp4'):
    model.eval()
    confusion_matrix = np.zeros((2, 2))
    test_loss = 0.0
    writer = imageio.get_writer(output_vid, fps=6)
    with torch.no_grad():
        for batchIdx, (data) in enumerate(test_loader):
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
            test_loss += torch.sqrt(criterion(ground_truth*100, output))

            for idx in range(b):
                bounding_box = BoundingBox(device)
                conf_mat, detected_bbox = bounding_box.post_process(
                    output[idx], data['bbox'][idx])                
                confusion_matrix += conf_mat
                orig_image = data['image'][idx]
                if batchIdx == 0:
                    orig = F.interpolate(orig_image.clone().to(device)[None, ...], size=(
                        [output.shape[2], output.shape[3]]), mode='bilinear', align_corners=True)
                    gt_mask = torch.cat((ground_truth[idx], torch.ones(
                        2, output.shape[2], output.shape[3]).to(device)))
                    # pred = (output[idx]/torch.max(output[idx]))[None, ...]
                    # snet_pred = torch.clamp(snet_output[idx], 0, 1)
                    # snet_pred = torch.cat((snet_pred, torch.ones(2, output.shape[2], output.shape[3]).to(device)))
                    pred = torch.clamp(output[idx], 0, 1)
                    pred = torch.cat(
                        (pred, torch.ones(2, output.shape[2], output.shape[3]).to(device)))
                    grid_row = torch.cat(
                        (orig, gt_mask[None, ...], pred[None, ...]))
                    # grid_row = torch.cat(
                    #     (orig_image, gt_mask[None, ...], snet_pred[None, ...], pred[None, ...]))
                    if idx == 0:
                        grid = grid_row
                    else:
                        grid = torch.cat((grid, grid_row))

                if device == torch.device('cuda'):
                    orig_image = orig_image.cpu()

                orig_image = orig_image.numpy()
                orig_image = np.stack((orig_image[2], orig_image[1], orig_image[0]), axis=2)
                orig_image = (orig_image*255).astype('uint8')

                if detected_bbox is not None:
                    detected_bbox *= 4
                    cv2.rectangle(orig_image, (detected_bbox[0, 0], detected_bbox[0, 1]), (
                        detected_bbox[0, 0]+detected_bbox[1, 0], detected_bbox[0, 1]+detected_bbox[1, 1]), (0, 0, 0))

                writer.append_data(orig_image)

    test_loss /= len(test_loader.dataset)
    writer.close()
    return test_loss.item(), confusion_matrix, grid
