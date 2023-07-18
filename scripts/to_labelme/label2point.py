import base64
import json
import os
import pathlib
import pprint

import cv2
import numpy as np
import tqdm

LABELS = {"background":0,"0_shixian":1,"0_xuxian":2,"1_shixian":3,"1_xuxian":4,"2_shixian":5,"2_xuxian":6,"3_shixian":7,"3_xuxian":8}
"""
重新生成 train_gt.txt

"""

def image_to_base64(image_np):
    image = cv2.imencode('.jpg',image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    if np.__version__ <"1.18.00":
        img_array = np.fromstring(img_data, np.uint8)
    else:
        img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img
def mkdir(p):
    os.makedirs(p,exist_ok=True)

class gender_Mask:
    ##生成mask img图像
    def __init__(self,jsonDir,root,name='06030819_0755.MP4'):
        self.w,self.h = 1640,590
        self.jsonDir = jsonDir
        self.imgDir = os.path.join(root,'driver_custom',name)
        self.maskpth = os.path.join(root,'laneseg_label_w16','driver_custom',name)
        for p in [self.imgDir,self.maskpth]:
            mkdir(p)
    def _core(self,json_p):
        json_p = str(json_p)
        orgin_img_pth = os.path.join(self.imgDir,os.path.basename(json_p).replace(".json",'.jpg'))
        mask_img_pth = os.path.join(self.maskpth,os.path.basename(json_p).replace(".json",'.png'))
        with open(json_p, 'r', encoding='utf-8') as file:
            data = json.load(file)
            img_str = data['imageData']
            img = base64_to_image(img_str)
            bg = np.zeros_like(img)
            for shape_data in data['shapes']:
                if shape_data['label'].strip() in list(LABELS.keys()):
                    color = int(LABELS[shape_data['label'].strip()])
                    color = [color,color,color]
                    points = np.int32(shape_data['points'])
                    bg = cv2.polylines(bg,[points],isClosed=True,color=color,thickness=2)
            img = cv2.resize(img,(self.w,self.h))
            cv2.imwrite(orgin_img_pth,img)
            bg = cv2.resize(bg,(self.w,self.h))
            cv2.imwrite(mask_img_pth,bg)
    def loop(self):
        json_pths = list(pathlib.Path(self.jsonDir).glob("**/*.json"))
        for json_p in tqdm.tqdm(json_pths,desc='gender_Mask '):
            self._core(json_p)

class gender_GT:
    ##生成mask img图像
    def __init__(self,lines,jsonDir,root,name='06030819_0755.MP4'):
        self.jsonDir = jsonDir
        self.root = os.path.join(root,name)
        self.train_gt = os.path.join(root,'list','train_gt.txt')
        self.imgDir = os.path.join(root,'driver_custom',name)
        self.maskpth = os.path.join(root,'laneseg_label_w16','driver_custom',name)
        self.lines = lines
        for p in [os.path.dirname(self.train_gt)]:
            mkdir(p)
    def _core(self,json_p,train_gt_f):
        json_p = str(json_p)
        orgin_img_pth = os.path.join(self.imgDir,os.path.basename(json_p).replace(".json",'.jpg'))
        mask_img_pth = os.path.join(self.maskpth,os.path.basename(json_p).replace(".json",'.png'))
        
        lp =['0']*self.lines
        with open(json_p, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for shape_data in data['shapes']:
                ll = shape_data['label'].strip()
                if ll in list(LABELS.keys()):
                    lp[LABELS[ll]-1]=str(LABELS[ll])
                    
        train_gt_line = f"{orgin_img_pth} {mask_img_pth} {' '.join(lp)}\n"
        train_gt_f.write(train_gt_line)
    def loop(self):
        json_pths = list(pathlib.Path(self.jsonDir).glob("**/*.json"))
        with open(self.train_gt,'w') as train_gt_f:
            for json_p in tqdm.tqdm(json_pths,desc='gender_GT '):
                self._core(json_p,train_gt_f)


class gender_Points:  # 9
    def __init__(self,lines,root,name='06030819_0755.MP4'):
        self.row_anchor = [250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 
                                    350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 
                                    450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 
                                    550, 560, 570, 580, 590]
        self.w,self.h = 1640,590
        self.maskDir = os.path.join(root,'laneseg_label_w16','driver_custom',name)  # mask 
        self.imgDir = os.path.join(root,'driver_custom',name)                       # 00000.lines.txt
        
        self.lines = lines
        for p in [self.imgDir,self.maskDir]:
            mkdir(p)
    def _get_x(self,line,thred):
        if thred in line:
            indx = line==thred
            return str(int((np.argmax(indx) + np.argmin(indx))*.5))
        else:
            return "-9999"
    def _core(self,mask_p,lines_p):
        mask_p = str(mask_p)
        mask_ = cv2.imread(mask_p,0)
    
        points = {f"{k+1}":[] for k in range(self.lines)}
        # print('points  ',points)
        f = open(lines_p,'w')
        for row in self.row_anchor:
            line = mask_[row-1,:] 
            # print(line.shape)
            for label in list(LABELS.values())[1:]:
                x = self._get_x(line,label)
                if x!="-9999":
                    points[str(label)].extend([x,str(row)])
        for ind in range(self.lines):
            values = points[str(ind+1)]
            if values!=[]:
                f.write(' '.join(values)+'\n')
        f.close()

        
    def loop(self):
        mask_pths = list(pathlib.Path(self.maskDir).glob("**/*.png"))
      
        for mask_p in tqdm.tqdm(mask_pths,desc='gender_Points '):
            lines_p = os.path.join(self.imgDir,os.path.basename(mask_p).replace(".png",'.lines.txt'))
            self._core(mask_p,lines_p)
                        
if __name__ == "__main__":
    jsonDir='/mnt/10t/chenglong/code/ufld/data'
    root='/mnt/10t/chenglong/code/ufld/data/culane'
    lines=8
    # gm = gender_Mask(jsonDir,root)
    # gm.loop()
    gt =gender_GT(lines,jsonDir,root)
    gt.loop()
    gp = gender_Points(lines,root)
    gp.loop()