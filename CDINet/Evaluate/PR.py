import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, convolve, distance_transform_edt as bwdist


class CalFM(object):
    # Fmeasure(maxFm, meanFm)---Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, num, thds=255):
        self.precision = np.zeros((num, thds))
        self.recall = np.zeros((num, thds))
        self.meanF = np.zeros(num)
        self.idx = 0
        self.num = num

    def update(self, pred, gt):
        if gt.max() != 0:
            prediction, recall, mfmeasure = self.cal(pred, gt)
            self.precision[self.idx, :] = prediction
            self.recall[self.idx, :] = recall
            self.meanF[self.idx] = mfmeasure
        self.idx += 1

    def cal(self, pred, gt):
        ########################meanF##############################
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        binary = np.zeros_like(pred)
        binary[pred >= th] = 1
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        tp = (binary * hard_gt).sum()
        if tp == 0:
            mfmeasure = 0
        else:
            pre = tp / binary.sum()
            rec = tp / hard_gt.sum()
            mfmeasure = 1.3 * pre * rec / (0.3 * pre + rec)

        ########################maxF##############################
        pred = np.uint8(pred * 255)
        target = pred[gt > 0.5]
        nontarget = pred[gt <= 0.5]
        targetHist, _ = np.histogram(target, bins=range(256))
        nontargetHist, _ = np.histogram(nontarget, bins=range(256))
        targetHist = np.cumsum(np.flip(targetHist), axis=0)
        nontargetHist = np.cumsum(np.flip(nontargetHist), axis=0)
        precision = targetHist / (targetHist + nontargetHist + 1e-8)
        recall = targetHist / np.sum(gt)
        return precision, recall, mfmeasure

    def show(self):
        assert self.num == self.idx, f"{self.num}, {self.idx}"
        precision = self.precision.mean(axis=0)
        recall = self.recall.mean(axis=0)
        fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + 1e-8)
        mmfmeasure = self.meanF.mean()
        return fmeasure, fmeasure.max(), mmfmeasure, precision, recall