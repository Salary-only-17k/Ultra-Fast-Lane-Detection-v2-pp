import numpy as np
import torch
from model.backbone import resnet
# from model.layer import CoordConv
from model.seg_model import SegHead
# from torch.nn import functional as F
# from utils.common import initialize_weights


class parsingNet(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row = None, num_cls_row = None, \
                 num_grid_col = None, num_cls_col = None, num_lane_on_row = None, num_lane_on_col = None, \
                 use_aux=False,input_height = None, input_width = None, fc_norm = False,cls_lane=5,train_method=0):
        super(parsingNet, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux
        self.train_method = train_method
        self.num_lane = 4
        self.cls_lane = cls_lane
        self.cls_out = self.num_lane*self.cls_lane    #  num_lane * cls_lane 
        
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4      +  self.cls_out
 
        self.input_dim = input_height // 32 * input_width // 32 *  8

        if self.train_method==0 or self.train_method==1:
            self.model = resnet(backbone, pretrained=pretrained).requires_grad_(requires_grad=True)
            if backbone in ['34','18', '34fca']:
                self.pool = torch.nn.Conv2d(512,256,1).requires_grad_(requires_grad=True)     
            else:
                raise ValueError
            # 57600 32400 576 648 20
            print('self.dim1,self.dim2,self.dim3,self.dim4,self.cls_out:  ',self.dim1,self.dim2,self.dim3,self.dim4,self.cls_out)
            self.cls_loc_row = torch.nn.Sequential(
                torch.nn.Conv2d(32,4,3,1),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(1536) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(1536,self.dim1) # 57600)
            ).requires_grad_(requires_grad=True)
            self.cls_loc_col = torch.nn.Sequential(
                torch.nn.Conv2d(32,4,3,1),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(1536) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(1536,self.dim2) #32400)
                ).requires_grad_(requires_grad=True)
    
            self.cls_ext_row = torch.nn.Sequential(
                torch.nn.Conv2d(32,2,3,2),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(192) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(192,self.dim3) #576)
            ).requires_grad_(requires_grad=True)
            self.cls_ext_col = torch.nn.Sequential(
                torch.nn.Conv2d(32,2,3,2),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(192) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(192,self.dim4) #648)  
            ).requires_grad_(requires_grad=True)
        else:
            self.model = resnet(backbone, pretrained=pretrained).requires_grad_(requires_grad=False) 
            if backbone in ['34','18', '34fca']:
                self.pool = torch.nn.Conv2d(512,256,1).requires_grad_(requires_grad=False)      
            else:
                raise ValueError
            # 57600 32400 576 648 20
            # print('self.dim1,self.dim2,self.dim3,self.dim4,self.cls_out:  ',self.dim1,self.dim2,self.dim3,self.dim4,self.cls_out)
            self.cls_loc_row = torch.nn.Sequential(
                torch.nn.Conv2d(32,4,3,1),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(1536) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(1536,self.dim1)# 57600)
            ).requires_grad_(requires_grad=False) 
            self.cls_loc_col = torch.nn.Sequential(
                torch.nn.Conv2d(32,4,3,1),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(1536) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(1536,self.dim2) #32400)
                ).requires_grad_(requires_grad=False) 
    
            self.cls_ext_row = torch.nn.Sequential(
                torch.nn.Conv2d(32,2,3,2),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(192) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(192,self.dim3) #576)
            ).requires_grad_(requires_grad=False) 
            self.cls_ext_col = torch.nn.Sequential(
                torch.nn.Conv2d(32,2,3,2),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(192) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(192,self.dim4)#648)  
            ).requires_grad_(requires_grad=False) 
        if self.train_method==0 or self.train_method==2:
            self.cls_lan_cls = torch.nn.Sequential(
                torch.nn.Conv2d(256,128,3,2),
                torch.nn.Conv2d(128,4,3,2),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(384) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(384,self.cls_out)).requires_grad_(requires_grad=True)
        else:
            self.cls_lan_cls = torch.nn.Sequential(
                torch.nn.Conv2d(256,128,3,2),
                torch.nn.Conv2d(128,4,3,2),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(384) if fc_norm else torch.nn.Identity(),
                torch.nn.ReLU(),
                torch.nn.Linear(384,self.cls_out)).requires_grad_(requires_grad=False)

        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)
        # initialize_weights(self.cls)
    def forward(self, x):

        x2,x3,fea = self.model(x)  # 是layer-2，layer-3，layer-4
        if self.use_aux:
            seg_out = self.seg_head(x2, x3,fea)
        fea = self.pool(fea)
    
        cls_loc_row = fea[:,128:160,...]
        cls_loc_row = self.cls_loc_row(cls_loc_row)
        
        cls_loc_col = fea[:,160:192,...]
        cls_loc_col = self.cls_loc_col(cls_loc_col)
        
        cls_ext_row = fea[:,192:224,...]
        cls_ext_row = self.cls_ext_row(cls_ext_row)
        
        cls_ext_col = fea[:,224:,...]
        cls_ext_col = self.cls_ext_col(cls_ext_col)
        
        cls_lan_cls = x3
        cls_lan_cls = self.cls_lan_cls(cls_lan_cls)
        
        fea = cls_ext_row
        pred_dict = {
                'loc_row': cls_loc_row.view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                'loc_col': cls_loc_col.view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row':cls_ext_row.view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                'exist_col': cls_ext_col.view(-1, 2, self.num_cls_col, self.num_lane_on_col),
                # # 'lane_labels': F.softmax(out[:,-self.cls_out:].reshape([-1,  self.num_lane,self.cls_lane]),dim=1)}  #! gai  加了一个softmax  收敛慢
                 'lane_labels': cls_lan_cls.reshape([-1,  self.num_lane,self.cls_lane])
                }  #! gai  
        if self.use_aux:
            pred_dict['seg_out'] = seg_out
        return pred_dict
        # for k,v in pred_dict.items():
        #     print(k,' : ',v.shape)
        # return fea

    def forward_tta(self, x):
        x2,x3,fea = self.model(x)
        
        pooled_fea = self.pool(fea)
        n,c,h,w = pooled_fea.shape

        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)

        left_pooled_fea[:,:,:,:w-1] = pooled_fea[:,:,:,1:]
        left_pooled_fea[:,:,:,-1] = pooled_fea.mean(-1)
        
        right_pooled_fea[:,:,:,1:] = pooled_fea[:,:,:,:w-1]
        right_pooled_fea[:,:,:,0] = pooled_fea.mean(-1)

        up_pooled_fea[:,:,:h-1,:] = pooled_fea[:,:,1:,:]
        up_pooled_fea[:,:,-1,:] = pooled_fea.mean(-2)

        down_pooled_fea[:,:,1:,:] = pooled_fea[:,:,:h-1,:]
        down_pooled_fea[:,:,0,:] = pooled_fea.mean(-2)
        # 10 x 25
        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea, up_pooled_fea, down_pooled_fea], dim = 0)
        fea = fea.view(-1, self.input_dim)

        out = self.cls(fea)

        return {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col),
                'lane_labels': out[:,-self.cls_out:].reshape([-1,  self.num_lane,self.cls_lane])} 

def get_model(cfg):
    return parsingNet(pretrained = True, backbone=cfg.backbone, 
                      num_grid_row = cfg.num_cell_row, num_cls_row = cfg.num_row, num_grid_col = cfg.num_cell_col, \
                        num_cls_col = cfg.num_col, num_lane_on_row = cfg.num_lanes, num_lane_on_col = cfg.num_lanes, \
                          use_aux = cfg.use_aux, input_height = cfg.train_height, \
                      input_width = cfg.train_width, fc_norm = cfg.fc_norm, \
                      cls_lane=cfg.cls_lane,train_method=cfg.train_method).cuda()