import torch
import torch.nn.functional as F
import torchvision.transforms.functional as func
import numpy as np
from utils.processing import BoundingBox
import imageio
import cv2


def test(model, test_loader, criterion, device, writer=None):
    """ Function to test the model
    Args:
        model (nn.model object): Model to be tested
        test_loader (utils.dataloader object): Dataloader for test data
        criterion (nn.loss object): loss object to calculate MSE loss
        device (torch.device object): device to load data on
        writer (imageio.writer object): Imageio Writer to write frames to output video file
    
    Return:
        test_loss (double): Test loss over whole test dataset
        confusion_matrix (2D numpy array): Confusion matrix calculated over one epoch
        grid (tensor): multidimensional tensors to plot image grid
    """
    model.eval()
    confusion_matrix = np.zeros((2, 2))
    test_loss = 0.0
    prev_frame = None
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
            output, prev_frame = model(data['image'], prev_frame)
            test_loss += criterion(ground_truth, output)

            for idx in range(b):
                bounding_box = BoundingBox(device)
                conf_mat, detected_bbox_list = bounding_box.post_process(
                    output[idx], data['bbox'][idx])
                confusion_matrix += conf_mat
                orig_image = data['image'][idx]
                if batchIdx == 30:
                    orig = F.interpolate(data['image'][idx].clone().to(device)[None, ...], size=(
                        [output.shape[2], output.shape[3]]), mode='bilinear', align_corners=True)
                    gt_mask = torch.cat((ground_truth[idx], torch.zeros(
                        2, output.shape[2], output.shape[3]).to(device)))
                    gt_mask = torch.clamp(gt_mask, 0, 1)
                    final_out = torch.clamp(output[idx], 0, 1)
                    final_out = torch.cat(
                        (final_out, torch.zeros(2, output.shape[2], output.shape[3]).to(device)))                    
                    grid_row = torch.cat(
                        (orig, gt_mask[None, ...], final_out[None, ...]))

                    if idx == 0:
                        grid = grid_row
                    else:
                        grid = torch.cat((grid, grid_row))

                if writer is not None:
                    if device == torch.device('cuda'):
                        orig_image = orig_image.cpu()

                    orig_image = orig_image.numpy()
                    orig_image = np.stack(
                        (orig_image[2], orig_image[1], orig_image[0]), axis=2)
                    orig_image = (orig_image*255).astype('uint8')

                    if len(detected_bbox_list) > 0:
                        for detected_bbox in detected_bbox_list:
                            detected_bbox *= 4
                            cv2.rectangle(orig_image, (detected_bbox[0, 0], detected_bbox[0, 1]), (
                                detected_bbox[0, 0]+detected_bbox[1, 0], detected_bbox[0, 1]+detected_bbox[1, 1]), (0, 0, 0))

                    writer.append_data(orig_image)

    test_loss /= len(test_loader.dataset)
    return test_loss.item(), confusion_matrix, grid


def detect_ball(model, reader, output_video, device):
    """ Function to detect ball in frames for an un-annotated video file
    Args:
        model (nn.model object): Model to be used
        reader (imageio.reader object): Imageio Reader to read from from video file
        output_video (string): Output video file name
        device (torch.device object): device to load data on
    """
    model.eval()
    with torch.no_grad():
        writer = imageio.get_writer(output_video, fps=25)
        bounding_box = BoundingBox(device)
        for data in reader:
            in_frame = func.to_tensor(np.array(data)).to(device)
            output, _ = model(in_frame[None, ...], None)
            out_frame = np.array(data).astype('uint8')
            merged_frame = np.ones((out_frame.shape[0], out_frame.shape[1]+16, out_frame.shape[2])).astype('uint8')
            merged_frame *= 255
            small_frame = cv2.resize(out_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            row_start = int(merged_frame.shape[0]/4)
            _, detected_bbox_list = bounding_box.post_process(output[0], bbox=None)
            if len(detected_bbox_list) > 0:
                for detected_bbox in detected_bbox_list:
                    detected_bbox *= 2
                    cv2.rectangle(small_frame, (detected_bbox[0, 0], detected_bbox[0, 1]), (
                        detected_bbox[0, 0]+detected_bbox[1, 0], detected_bbox[0, 1]+detected_bbox[1, 1]), (0, 0, 0))
            merged_frame[row_start:row_start+small_frame.shape[0], 0:small_frame.shape[1], :] = small_frame
            heatmap = output[0][0].cpu().numpy()
            heatmap = (heatmap*255).astype('uint8')
            heatmap = cv2.resize(heatmap, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)            
            merged_frame[row_start:row_start+small_frame.shape[0], small_frame.shape[1]+16:, 0] = heatmap
            merged_frame[row_start:row_start+small_frame.shape[0], small_frame.shape[1]+16:, 1:] = 0
            writer.append_data(merged_frame)
        writer.close()
