import cv2
import tqdm
import os
import pathlib
import numpy as np
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]

class merge(object):
    def __init__(self,jpgDir,pngDir,mergeDir):

        self.jpgDir = jpgDir
        self.pngDir = pngDir
        self.merge = mergeDir
        self._mkdir(self.merge)
        self.lanes = 4
    def _mkdir(self,p):
        os.makedirs(p,exist_ok=True)
    # def _core(self,jpg,png):
    #     img = cv2.imread(jpg)
    #     mask_ = cv2.imread(png)
    #     for i in range(1,self.lanes+1,1):
    #         ele = (i,i,i)
    #         mask_[mask_==ele] = COLORS[i-1]
    #     merge = cv2.addWeighted(img,0.7,mask_,0.4,0.0)
    #     cv2.imwrite(merge,os.path.join(self.merge,os.path.basename(jpg)))
    def _core(self,jpg,png):
        img = cv2.imread(jpg)
        mask_ = cv2.imread(png,0) 
        bgr = np.zeros_like(img)
        for i in range(1,np.max(mask_)+1,1):
            # ele = (i,i,i)
            bgr[mask_==i] = COLORS[i-1]
        merge = cv2.addWeighted(img,0.7,bgr,0.2,0.0)
        cv2.imwrite(os.path.join(self.merge,os.path.basename(jpg)),merge)
    def loop(self):
        jpg_lst = list(pathlib.Path(self.jpgDir).glob('**/*.jpg'))
        for jpg in tqdm.tqdm(jpg_lst):
            jpg = str(jpg)
            b,a = os.path.split(jpg)
            dirn = os.path.basename(b)
            png = os.path.join(self.pngDir,dirn,a.replace(".jpg",'.png'))
            self._core(jpg,png)



if __name__ == "__main__":
    jpgDir = r'E:\Users\Administrator\Desktop\del\37_det_lane\CULane_ex\driver_161_90frame'
    pngDir = r'E:\Users\Administrator\Desktop\del\37_det_lane\CULane_ex\laneseg_label_w16\driver_161_90frame'
    mergeDir = r'E:\Users\Administrator\Desktop\del\37_det_lane\CULane_ex\merge'
    func = merge(jpgDir,pngDir,mergeDir)
    func.loop()  
