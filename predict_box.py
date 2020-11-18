import time
import cv2
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from model import craft_utils
from model.craft_utils import copyStateDict
from model.craft import CRAFT
from utils import imgproc
from model.refinenet import RefineNet
import numpy as np
from utils.file_utils import displayResult
from dotenv import load_dotenv
from os import getenv


load_dotenv('./.env')


USE_CUDA = getenv('CRAFT_USE_CUDA') == '1'
TRAINED_MODEL = getenv('CRAFT_TRAINED_MODEL')
CANVAS_SIZE = int(getenv('CRAFT_CANVAS_SIZE'))
TEXT_THRESHOLD = float(getenv('CRAFT_TEXT_THRESHOLD'))
LOW_TEXT = float(getenv('CRAFT_LOW_TEXT'))
LINK_THRESHOLD = float(getenv('CRAFT_LINK_THRESHOLD'))
MAG_RATIO = float(getenv('CRAFT_MAG_RATIO'))
ENABLE_POLYGON = getenv('CRAFT_ENABLE_POLYGON') == '1'
ENABLE_REFINER = getenv('CRAFT_ENABLE_REFINER') == '1'
REFINER_MODEL = getenv('CRAFT_REFINER_MODEL')
HORIZONTAL_MODE = getenv('CRAFT_HORIZONTAL_MODE') == '1'
RATIO_BOX_HORIZONTAL = float(getenv('CRAFT_RATIO_BOX_HORIZONTAL'))
EXPAND_RATIO = float(getenv('CRAFT_EXPAND_RATIO'))
ENABLE_VISUALIZE = getenv('CRAFT_ENABLE_VISUALIZE') == '1'
SHOW_TIME = getenv('CRAFT_SHOW_TIME') == '1'


class CraftDetection:
    def __init__(self):
        self.model = CRAFT()
        if USE_CUDA:
            self.model.load_state_dict(
                copyStateDict(torch.load(TRAINED_MODEL)))
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = False
        else:
            self.model.load_state_dict(copyStateDict(
                torch.load(TRAINED_MODEL, map_location='cpu')))
        self.model.eval()

        self.refine_model = None
        self.enable_polygon = ENABLE_POLYGON
        if ENABLE_REFINER:
            self.refine_model = RefineNet()
            if USE_CUDA:
                self.refine_model.load_state_dict(
                    copyStateDict(torch.load(REFINER_MODEL)))
                self.refine_model = self.refine_net.cuda()
                self.refine_model = torch.nn.DataParallel(self.refine_model)
            else:
                self.refine_model.load_state_dict(copyStateDict(
                    torch.load(REFINER_MODEL, map_location='cpu')))
            self.refine_model.eval()
            self.enable_polygon = True

    def text_detect(self, image):
        time0 = time.time()

        # resize
        img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
            image,
            CANVAS_SIZE,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=MAG_RATIO
        )
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if USE_CUDA:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.model(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_model is not None:
            with torch.no_grad():
                y_refiner = self.refine_model(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        time0 = time.time() - time0
        time1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, TEXT_THRESHOLD, LINK_THRESHOLD, LOW_TEXT, self.enable_polygon)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        if HORIZONTAL_MODE:
            if self.check_horizontal(polys):
                height, _, _ = image.shape
                new_polys = []
                for box in polys:
                    [[l1, t1], [r1, t2], [r2, b1], [l2, b2]] = box
                    if t1 < t2:
                        l, r, t, b = l2, r1, t1, b1
                    elif t1 > t2:
                        l, r, t, b = l1, r2, t2, b2
                    else:
                        l, r, t, b = l1, r1, t1, b1
                    h_box = abs(b - t)
                    t = max(0, t - h_box * EXPAND_RATIO)
                    b = min(b + h_box * EXPAND_RATIO, height)
                    new_box = [[l, t], [r, t], [r, b], [l, b]]
                    new_polys.append(new_box)

                polys = np.array(new_polys, dtype=np.float32)

        # for box in polys:
        time1 = time.time() - time1
        total_time = round(time0 + time1, 2)

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))

        if SHOW_TIME:
            print("\nInfer/postproc time: {:.3f}/{:.3f}".format(time0, time1))

        img_draw = None
        if ENABLE_VISUALIZE:
            img_draw = displayResult(img=image[:, :, ::-1], boxes=polys)

        result_boxes = []
        for box in polys:
            result_boxes.append(box.tolist())
        return result_boxes, img_draw, total_time

    def check_horizontal(self, boxes):
        total_box = len(boxes)
        if (total_box == 0):
            return False
        num_box_horizontal = 0
        for box in boxes:
            [[l1, t1], [r1, t2], [r2, b1], [l2, b2]] = box
            if t1 == t2:
                num_box_horizontal += 1

        ratio_box_horizontal = num_box_horizontal / float(total_box)
        print("Ratio box horizontal: ", ratio_box_horizontal)
        if ratio_box_horizontal >= RATIO_BOX_HORIZONTAL:
            return True
        else:
            return False


if __name__ == "__main__":
    app = CraftDetection()
