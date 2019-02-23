import torch
import numpy as np
import cv2


class BoundingBox(object):
    def __init__(self, device):
        self.device = device

    def pre_process(self, bbox, input_size, output_size):
        if torch.equal(bbox[2], torch.DoubleTensor([0.0, 0.0]).to(self.device)) == False:
            img_heatmap = torch.ones(
                (output_size[0], output_size[1], output_size[2])).to(self.device)
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
            centerX, centerY = int(bbox[2][0].item()), int(
                bbox[2][1].item())
            dist = torch.abs(pt1-pt2)
            width, length = dist[0].item(), dist[1].item()


            if length > width:
                ksize = int(length)
            else:
                ksize = int(width)
            # if ksize % 2 == 0:
            #     ksize += 1
            # kernel = cv2.getGaussianKernel(ksize, 4).astype('float32')
            # kernel = torch.from_numpy(np.dot(kernel, kernel.T)).to(self.device)

            kernel = torch.ones((ksize, ksize)).to(self.device)
            for y in range(kernel.shape[0]):
                for x in range(kernel.shape[1]):
                    dist = np.sqrt((centerX-(x+pt1[0].item()))**2 + (centerY-(y+pt1[1].item()))**2)
                    if dist <= (ksize/2):
                        kernel[y, x] = 64/255

            kY_start = kernel.shape[0] - int(bbox[3][1].item())
            kY_end = kernel.shape[0]
            kX_start = kernel.shape[1] - int(bbox[3][0].item())
            kX_end = kernel.shape[1]

            img_heatmap[0, int(pt1[1].item()):int(pt2[1].item()), int(pt1[0].item()):int(
                pt2[0].item())] = kernel[kY_start:kY_end, kX_start:kX_end]
            # img_heatmap[img_heatmap < 1/255] = 1.0
        else:
            img_heatmap = torch.ones(
                (output_size[0], output_size[1], output_size[2])).to(self.device)

        return img_heatmap, bbox

    def post_process(self, input, bbox):
        image = input.cpu().numpy()
        if input.shape[0] == 1:
            image = image.reshape(image.shape[1], image.shape[2])
        else:
            image = np.stack((image[2], image[1], image[0]), axis=2)
        image = (image*255).astype('uint8')
        image = cv2.medianBlur(image, 5)

        ball_present = False
        ball_detected = False
        # true_positive = False
        area_threshold_min = 20.0
        area_threshold_max = 5000.0
        height_threshold = width_threshold = 5.0
        confusion_matrix = np.zeros((2, 2))
        # detected_bbox = []
        detected_bbox = None

        if bbox[2][0] > 0 or bbox[2][1] > 0:
            ball_present = True

        if bbox[3][0] >= bbox[3][1]:
            threshold = bbox[3][0].item()/2
        else:
            threshold = bbox[3][0].item()/2

        erosion_size = 4
        element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
        image = cv2.erode(image, element)
        dilation_size = 4
        element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*dilation_size + 1, 2*dilation_size+1), (dilation_size, dilation_size))
        image = cv2.dilate(image, element)

        if input.shape[0] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, img_thresh = cv2.threshold(
            image, 15, 255, cv2.THRESH_BINARY_INV)
        _, contours, _ = cv2.findContours(
            img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            epsilon = 0.1*cv2.arcLength(c,True)
            c = cv2.approxPolyDP(c,epsilon,True)
            area = cv2.contourArea(c)
            x, y, width, height = cv2.boundingRect(c)
            if area >= area_threshold_min and area < area_threshold_max and height >= height_threshold and width >= width_threshold:
                ball_detected = True
                if ball_present:
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    squared_dist = np.sqrt(
                        (cX - bbox[2][0])**2 + (cY - bbox[2][1])**2)
                    if squared_dist <= threshold:
                        # true_positive = True
                        confusion_matrix[1, 1] += 1
                        # detected_bbox.append([[x, y], [height, width]])
                        detected_bbox = np.array([[x, y], [width, height]])
                    else:
                        confusion_matrix[0, 1] += 1
                else:
                    confusion_matrix[0, 1] += 1

        # if true_positive:
        #     confusion_matrix[1, 1] += 1
        if ball_present and ball_detected == False:
            confusion_matrix[1, 0] += 1
        elif ball_present == False and ball_detected == False:
            confusion_matrix[0, 0] += 1

        # return confusion_matrix, detected_bbox, image, img_thresh, contours
        return confusion_matrix, detected_bbox
