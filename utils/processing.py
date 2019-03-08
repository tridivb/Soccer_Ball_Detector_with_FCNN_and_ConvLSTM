import torch
import numpy as np
import cv2


class BoundingBox(object):
    """ Class to process the data as necessary
    """
    def __init__(self, device):
        self.device = device

    def pre_process(self, bbox, input_size, output_size):
        """ Pre-process the data and create ground truth by fitting a gaussian at
             the location of the ball
        Args:
            bbox (Tensor): Input bounding box
            input_size (tuple): Size of input image
            output_size (tuple): Size of output image (ground truth)

        Returns:
            img_heatmap (Tensor): Ground truth heatmap
            bbox (Tensor): Scaled bounding box coordinates as per output_size
        """
        # Check if ball is present or not from the bounding box coordinates
        # Center of bounding box must be greater than 0 for a ball to be present
        if not torch.equal(bbox[2], torch.DoubleTensor([0.0, 0.0]).to(self.device)):
            img_heatmap = torch.zeros(
                (output_size[0], output_size[1], output_size[2])).to(self.device)
            # Check if bounding box needs to be scaled or not
            if input_size != output_size:
                scale = torch.DoubleTensor([output_size[1]/input_size[1],
                                            output_size[2]/input_size[2]]).to(self.device)
                bbox[0] = torch.round(bbox[0] * scale)
                bbox[1] = torch.round(bbox[1] * scale)
                bbox[3][0] = torch.abs(bbox[0, 0]-bbox[1, 0])
                bbox[3][1] = torch.abs(bbox[0, 1]-bbox[1, 1])
                bbox[2][0], bbox[2][1] = bbox[0, 0]+bbox[3, 0] / \
                    2, bbox[0, 1]+bbox[3, 1]/2

            pt1, pt2 = bbox[0], bbox[1]
            dist = torch.abs(pt1-pt2)
            width, length = dist[0].item(), dist[1].item()

            # Choose kernel size for gaussian
            if length > width:
                ksize = int(max(length, 15))
            else:
                ksize = int(max(width, 15))

            kernel = cv2.getGaussianKernel(ksize, 4)
            kernel = np.dot(kernel, kernel.T)
            kernel *= 100

            if pt1[1].item()+ksize > img_heatmap.shape[1]-1:
                kY_start = img_heatmap.shape[1]-1-ksize
            else:
                kY_start = int(pt1[1].item())

            if pt1[0].item()+ksize > img_heatmap.shape[2]-1:
                kX_start = img_heatmap.shape[2]-1-ksize
            else:
                kX_start = int(pt1[0].item())

            # Fit gaussian on the heatmap at bounding box location
            img_heatmap[0, kY_start:kY_start+ksize, kX_start:kX_start +
                        ksize] = torch.from_numpy(kernel).to(self.device)

        else:
            # When no ball is present
            img_heatmap = torch.zeros(
                (output_size[0], output_size[1], output_size[2])).to(self.device)

        return img_heatmap, bbox

    def post_process(self, input, bbox=None):
        """ Post-process the output data from model and detect contours in it
            Extract bound box coordinates using the detected contours
        Args:
            input (Tensor): Input for post processing
            bbox (Tensor): Input bounding box to compare against(optional) [Default: None]

        Returns:
            confusion_matrix (2D Numpy array): Confusion matrix over input image
            detected_bbox_list (List): List of all detected bounding boxes in image
        """
        # Convert to numpy and blur the image
        image = input.cpu().numpy()
        if input.shape[0] == 1:
            image = image.reshape(image.shape[1], image.shape[2])
        else:
            image = np.stack((image[2], image[1], image[0]), axis=2)
        image = (image*255).astype('uint8')
        image = cv2.medianBlur(image, 5)

        detected_bbox_list = []
        confusion_matrix = np.zeros((2, 2))
        # Set area threshold to filter out too small or too large blobs
        area_threshold_min = 20.0
        area_threshold_max = 5000.0
        # If annotation data is available calculate metrics
        if bbox is not None:
            ball_present = False
            ball_detected = False

            if bbox[3][0].item() > bbox[3][1].item():
                dist_threshold = bbox[3][0].item()
            else:
                dist_threshold = bbox[3][1].item()

            gt_cX, gt_cY = int(bbox[2][0].item()), int(bbox[2][1].item())

            if gt_cX > 0 or gt_cY > 0:
                ball_present = True

        # Erode and Dilute the image
        erosion_size = 4
        element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
        image = cv2.erode(image, element)
        dilation_size = 4
        element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*dilation_size + 1, 2*dilation_size+1), (dilation_size, dilation_size))
        image = cv2.dilate(image, element)

        _, img_thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(
            img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour filter out invalid contours and get bounding boxes for the rest
        for c in contours:
            epsilon = 0.1*cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, epsilon, True)
            area = cv2.contourArea(c)
            if area >= area_threshold_min and area <= area_threshold_max:
                x, y, width, height = cv2.boundingRect(c)
                detected_bbox_list.append(np.array([[x, y], [width, height]]))
                if bbox is not None:
                    ball_detected = True                    
                    if ball_present:
                        M = cv2.moments(c)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        dist = np.sqrt((cX - gt_cX)**2 + (cY - gt_cY)**2)
                        if dist <= dist_threshold:
                            # True Positive
                            confusion_matrix[1, 1] += 1
                        else:
                            # False Positive
                            confusion_matrix[0, 1] += 1
                    else:
                        # False Positive
                        confusion_matrix[0, 1] += 1

        if bbox is not None:
            if ball_present and not ball_detected:
                # False Negative
                confusion_matrix[1, 0] += 1
            if not ball_present and not ball_detected:
                # True Negative
                confusion_matrix[0, 0] += 1

        return confusion_matrix, detected_bbox_list
