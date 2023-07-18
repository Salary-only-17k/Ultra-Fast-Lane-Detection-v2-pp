import json
import base64
import tqdm
import pathlib
#上面这几个忘了哪个有用了，干脆都加上来了
import os
import cv2
import shutil
import numpy as np


def mkdir(p):
    os.makedirs(p,exist_ok=True)
    
def image_to_base64(image_np):
    image = cv2.imencode('.jpg',image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

def pth_pth(srcDir:str,asd:str):
    """
    asd - srcDir 
    """
    l1 = srcDir.split(os.path.sep)  # 长路径
    l0 = asd.split(os.path.sep)     # 短路径
    for l in l0:
        l1.remove(l)
    return os.path.join(*l1)

class points2json(object):
    def __init__(self,linesDir,resDir) -> None:
        self.linesDir=linesDir
        self.resDir=resDir
        
    def _json_struct1(self,shapes:list,basefile:str,base64img:str,h:int,w:int):
        return  {"version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": basefile,
            "imageData":base64img,   # base64
            "imageHeight": h,
            "imageWidth": w}
    def _json_struct2(self,label:str,xy:list):
        return  {"label": label,
                "points": xy,   # [[x,y],[x,y],[x,y],...]
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}}
               
    def _json_info(self,lines_pth,imgpth):
        with open(lines_pth,'r', encoding='utf-8') as f:
            lines = f.readlines()
        shapes_info = []
        for line in lines:
            x = line.strip().split(" ")[::2]
            y = line.strip().split(" ")[1::2]
            xy = np.vstack([x,y]).astype("float")
            xy = xy.T
            xy_tmp = []
            for xy_v in xy:
                xy_tmp.append(list(xy_v))
            shapes_info.append(self._json_struct2('line',xy_tmp))
            
        img = cv2.imread(imgpth)
        file = os.path.basename(imgpth)
        base64_img = image_to_base64(img)
        h,w = img.shape[:2]
        info = self._json_struct1(shapes_info,file,base64_img,h,w)
        return info,file
            
            
    def _core(self,lines_pth,pp):
        imgpth = lines_pth.replace(".lines.txt",'.jpg')
        p1 = os.path.join(self.resDir,pp)
        mkdir(p1)
        shutil.copy(imgpth,p1)
        info,file = self._json_info(lines_pth,imgpth)
        with open(os.path.join(p1,file).replace(".jpg",'.json'),'w', encoding='utf-8') as f:
            json.dump(info,f)
        
        
    def loop(self):
        lines_lst = list(pathlib.Path(self.linesDir).glob("**/*.lines.txt"))
        for lines_pth in tqdm.tqdm(lines_lst):
            lines_pth = str(lines_pth)
            dirpth = os.path.dirname(lines_pth)
            pp = pth_pth(dirpth,self.linesDir)
            self._core(lines_pth,pp)
            
        
if __name__ == "__main__":
    # linesDir=r'E:\Users\Administrator\Desktop\del\37_det_lane\CULane_ex\driver_161_90frame\06030819_0755.MP4'

    linesDir=r'E:\Users\Administrator\Desktop\del\37_det_lane\CULane_ex\driver_161_90frame'
    resDir=r'E:\Users\Administrator\Desktop\del\37_det_lane\CULane_ex\cope_\test'
    func= points2json(linesDir,resDir)
    func.loop()