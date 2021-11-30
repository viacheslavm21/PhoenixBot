import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import yolov5.detect

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

class croppedStream():
    def __init__(self,
                 device='',
                 weights= 'yolov5/weights/draft1a.pt',
                 source = 3,
                 imgsz=640,
                 conf_thres = 0.2,
                 iou_thres = 0.4,
                 classes = None,
                 agnostic_nms = False,
                 max_det= 5,
                 view = True,
                 crop = True):

        #self.detectSocket = yolov5.detect.run
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self. max_det = max_det
        self.view = view
        self.crop = crop

        self.device = select_device(device)

        # Load model
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False)
        self.stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Model setting
        # Half
        """
        half = (pt or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        """

        # Load data
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.dataset = LoadStreams(str(source), img_size=imgsz, stride=self.stride, auto=pt and not jit)
        bs = len(self.dataset)  # batch_size

    def run(self):
        seen = 0
        for path, im, im0s, vid_cap, s in self.dataset:
            #t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            #t2 = time_sync()
            #dt[0] += t2 - t1

            # Inference
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(im, augment=False, visualize=False)
            #t3 = time_sync()
            #dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            #dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                s += f'{i}: '

                p = Path(p)  # to Path

                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy()# if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=2, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    mostProbableDet = det[0]
                    print( (mostProbableDet))
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.view:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{self.names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))

                # Print time (inference-only)
                #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                # Stream results
                im0 = annotator.result()
                if self.view:
                    if self.crop and len(det):
                        print("cropped")
                        im0 = self.runcrop(imc,[mostProbableDet[0],mostProbableDet[1]],[mostProbableDet[2],mostProbableDet[3]])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

    def runcrop(self, img, UL, BR):
        #UL,BR = np.array(UL), np.array(BR)
        return img[int(UL[1]):int(BR[1]), int(UL[0]):int(BR[0])]

if __name__ == "__main__":
    stream = croppedStream()
    stream.run()
