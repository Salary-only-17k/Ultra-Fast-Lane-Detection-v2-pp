import json
import os
import random

import my_interp  # ~主要是用来插值车道线坐标的
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class LaneExternalIterator(object):
    def __init__(self, path, list_path, batch_size=None, shard_id=None, num_shards=None, mode = 'train', dataset_name=None,lines=4):
        """_summary_
        (data_root, list_path, batch_size=batch_size, shard_id=shard_id, num_shards=num_shards, dataset_name = dataset_name)
        Args:
            path (_type_): _description_
            list_path (_type_): _description_
            batch_size (_type_, optional): _description_. Defaults to None.
            shard_id (_type_, optional): _description_. Defaults to None.
            num_shards (_type_, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to 'train'.
            dataset_name (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            
        TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'list/train_gt.txt'), get_rank(), get_world_size(), 
        cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, 
        cfg.dataset, cfg.crop_ratio)
        """
        assert mode in ['train', 'val']
        self.mode = mode
        self.path = path
        self.list_path = list_path
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.lines = lines
        
        if isinstance(list_path, str):
            # print('list_path: ',list_path)     # 修改
            with open(os.path.join('/home/root/code/ufld/done_culane',list_path), 'r') as f:  #^ list/train_gt.txt
                total_list = f.readlines()   #^ img-pth  mask-pth 1 1 1 0 
        elif isinstance(list_path, list) or isinstance(list_path, tuple):
            total_list = []    # 原始图path  mask-path  1 1 1 1
            for lst_path in list_path:
                with open(lst_path, 'r') as f:
                    total_list.extend(f.readlines())
        else:
            raise NotImplementedError
        # print("total_list:  ", total_list)
        if self.mode == 'train':
            if dataset_name == 'CULane':
                cache_path = os.path.join(path, 'culane_anno_cache_train.json')
           
            else:
                raise NotImplementedError

            if shard_id == 0:
                print('loading cached data')
            cache_fp = open(cache_path, 'r')
            self.cached_points = json.load(cache_fp)
            if shard_id == 0:
                print('cached data loaded')
        else:
            if dataset_name == 'CULane':
                cache_path = os.path.join(path, 'culane_anno_cache_val.json')
           
            else:
                raise NotImplementedError

            if shard_id == 0:
                print('loading cached data')
            cache_fp = open(cache_path, 'r')
            self.cached_points = json.load(cache_fp)
            if shard_id == 0:
                print('cached data loaded')

        self.total_len = len(total_list)    # 数据对长 
    
        self.list = total_list[self.total_len * shard_id // num_shards:
                                self.total_len * (shard_id + 1) // num_shards]   # 一个batch-size的数据路径
        # print("self.list: ",self.list)
        self.n = len(self.list)  # 每个epoch中分成多少个内循环
        #^ self.list  <-  total_list  <-  img-pth  mask-pth 1 1 1 0 
        #^ self.cached_points  <-  img-pth:[[[x,y],...], [[x,y],...], [[x,y],...], [[x,y],...]]  4 条线
    def __iter__(self):
        self.i = 0
        if self.mode == 'train':
            random.shuffle(self.list)
        return self
    def __gender_labels(self,label_cls):
        tmp = np.zeros([self.lines])  # int 
        for i,v in enumerate(label_cls):
            label_cls[i]=v
        return tmp
    def _prepare_train_batch(self):
        images = []
        seg_images = []
        labels = []
        label_cls_lst = []
        for _ in range(self.batch_size):
            l = self.list[self.i % self.n]
            l_info = l.strip().split()
            img_name = l_info[0]    # img
            seg_name = l_info[1]    # mask
            label_cls = self.__gender_labels(l_info[2:])  #!  分类label
            label_cls_lst.append(label_cls)
            # if img_name[0] == '/':
            #     img_name = img_name[1:]  # 路径
            # if seg_name[0] == '/':
            #     seg_name = seg_name[1:]  # 路径
                
            img_name = img_name.strip()  # 
            seg_name = seg_name.strip()  # 去空格
            
            img_path = os.path.join(self.path, img_name)
            # print("read img-pth:  ",img_path)
            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))  # img

            img_path = os.path.join(self.path, seg_name)  
            # print("read mask-pth:  ",img_path)   
            with open(img_path, 'rb') as f:
                seg_images.append(np.frombuffer(f.read(), dtype=np.uint8))

            points = np.array(self.cached_points[img_name])   #^ img-pth:[[[x,y],...], [[x,y],...], [[x,y],...], [[x,y],...]]  4 条线
            labels.append(points.astype(np.float32))

            self.i = self.i + 1
        """
        images        img
        seg_images    mask
        labels        1 1 1 0
        """
        # print("labels：",labels)
        # print(label_cls_lst)
        return (images, seg_images, labels, label_cls_lst)    #^ img  mask [[x,y],...]*4

    
    def _prepare_test_batch(self):
        images = []
        names = []
        for _ in range(self.batch_size):
            img_name = self.list[self.i % self.n].split()[0]

            # if img_name[0] == '/':
            #     img_name = img_name[1:]
            img_name = img_name.strip()

            img_path = os.path.join(self.path, img_name)

            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))   #^ 图像通过byte读取，然后转为1维的数组
            names.append(np.array(list(map(ord,img_name))))  #^ 图像路径转成了16进制     这么做会非常快？ 
            self.i = self.i + 1
            
        return images, names

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        if self.mode == 'train':
            res = self._prepare_train_batch()
        elif self.mode == 'test':
            res = self._prepare_test_batch()
        else:
            raise NotImplementedError

        return res
    def __len__(self):
        return self.total_len

    next = __next__

def encoded_images_sizes(jpegs):
    shapes = fn.peek_image_shape(jpegs)  # the shapes are HWC
    h = fn.slice(shapes, 0, 1, axes=[0]) # extract height...
    w = fn.slice(shapes, 1, 1, axes=[0]) # ...and width...
    return fn.cat(w, h)               # ...and concatenate

def ExternalSourceTrainPipeline(batch_size, num_threads, device_id, external_data, train_width, train_height, top_crop, \
                                normalize_image_scale = False, nscale_w = None, nscale_h = None):
    """
    (batch_size, num_threads, shard_id, eii, train_width, train_height,top_crop)
    """
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, seg_images, labels,labels_cls = fn.external_source(source=external_data, num_outputs=4) #! gai 
        images = fn.decoders.image(jpegs, device="mixed")
        seg_images = fn.decoders.image(seg_images, device="mixed")
        #^ img和mask预处理
        if normalize_image_scale:
            images = fn.resize(images, resize_x=nscale_w, resize_y=nscale_h)
            seg_images = fn.resize(seg_images, resize_x=nscale_w, resize_y=nscale_h, interp_type=types.INTERP_NN)
            # make all images at the same size
        
        size = encoded_images_sizes(jpegs)
        center = size / 2
        #^ 类似transformer
        mt = fn.transforms.scale(scale = fn.random.uniform(range=(0.8, 1.2), shape=[2]), center = center)
        mt = fn.transforms.rotation(mt, angle = fn.random.uniform(range=(-6, 6)), center = center)

        off = fn.cat(fn.random.uniform(range=(-200, 200), shape = [1]), fn.random.uniform(range=(-100, 100), shape = [1]))
        mt = fn.transforms.translation(mt, offset = off)

        images = fn.warp_affine(images, matrix = mt, fill_value=0, inverse_map=False)
        seg_images = fn.warp_affine(seg_images, matrix = mt, fill_value=0, inverse_map=False)
        labels = fn.coord_transform(labels.gpu(), MT = mt)


        images = fn.resize(images, resize_x=train_width, resize_y=int(train_height/top_crop))
        seg_images = fn.resize(seg_images, resize_x=train_width, resize_y=int(train_height/top_crop), interp_type=types.INTERP_NN)


        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        seg_images = fn.crop_mirror_normalize(seg_images, 
                                            dtype=types.FLOAT, 
                                            mean = [0., 0., 0.],
                                            std = [1., 1., 1.],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        pipe.set_outputs(images, seg_images, labels,labels_cls)
    return pipe
def ExternalSourceTrainPipeline1(batch_size, num_threads, device_id, external_data, train_width, train_height, top_crop, normalize_image_scale = False, nscale_w = None, nscale_h = None):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, seg_images, labels = fn.external_source(source=external_data, num_outputs=3)
        images = fn.decoders.image(jpegs, device="mixed")
        seg_images = fn.decoders.image(seg_images, device="mixed")
        if normalize_image_scale:
            images = fn.resize(images, resize_x=nscale_w, resize_y=nscale_h)
            seg_images = fn.resize(seg_images, resize_x=nscale_w, resize_y=nscale_h, interp_type=types.INTERP_NN)
            # make all images at the same size

        size = encoded_images_sizes(jpegs)
        center = size / 2

        mt = fn.transforms.scale(scale = fn.random.uniform(range=(0.8, 1.2), shape=[2]), center = center)
        mt = fn.transforms.rotation(mt, angle = fn.random.uniform(range=(-6, 6)), center = center)

        off = fn.cat(fn.random.uniform(range=(-200, 200), shape = [1]), fn.random.uniform(range=(-100, 100), shape = [1]))
        mt = fn.transforms.translation(mt, offset = off)

        images = fn.warp_affine(images, matrix = mt, fill_value=0, inverse_map=False)
        seg_images = fn.warp_affine(seg_images, matrix = mt, fill_value=0, inverse_map=False)
        labels = fn.coord_transform(labels.gpu(), MT = mt)


        images = fn.resize(images, resize_x=train_width, resize_y=int(train_height/top_crop))
        seg_images = fn.resize(seg_images, resize_x=train_width, resize_y=int(train_height/top_crop), interp_type=types.INTERP_NN)


        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        seg_images = fn.crop_mirror_normalize(seg_images, 
                                            dtype=types.FLOAT, 
                                            mean = [0., 0., 0.],
                                            std = [1., 1., 1.],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        pipe.set_outputs(images, seg_images, labels)
    return pipe
def ExternalSourceValPipeline(batch_size, num_threads, device_id, external_data, train_width, train_height):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=train_width, resize_y=int(train_height/0.6)+1)
        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        pipe.set_outputs(images, labels.gpu())
    return pipe

def ExternalSourceTestPipeline(batch_size, num_threads, device_id, external_data):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, names = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")

        images = fn.resize(images, resize_x=800, resize_y=288)
        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255])

        names = fn.pad(names, axes=0, fill_value = -1, shape = 46)
        pipe.set_outputs(images, names)
    return pipe
# from data.constant import culane_row_anchor, culane_col_anchor
class TrainCollect:   # 数据加载的主函数
    def __init__(self, batch_size, num_threads, data_root, list_path, shard_id, num_shards, row_anchor, col_anchor, train_width, train_height, num_cell_row, num_cell_col,
    dataset_name, top_crop):
        eii = LaneExternalIterator(data_root, list_path, batch_size=batch_size, shard_id=shard_id, num_shards=num_shards, dataset_name = dataset_name)
        # print("eii:  ",eii)
        """
        TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'list/train_gt.txt'), get_rank(), get_world_size(), 
        cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, 
        cfg.dataset, cfg.crop_ratio)
        
        row_anchor
        col_anchor
        """
        #^ eii   img, mask, points
        if dataset_name in ['CULane','custom']:
            self.original_image_width = 1640
            self.original_image_height = 590
        elif dataset_name == 'Tusimple':
            self.original_image_width = 1280
            self.original_image_height = 720
        elif dataset_name == 'CurveLanes':
            self.original_image_width = 2560
            self.original_image_height = 1440

        if dataset_name == 'CurveLanes':
            #^ 预处理-数据增强  ExternalSourceTrainPipeline 
            pipe = ExternalSourceTrainPipeline(batch_size, num_threads, shard_id, eii, train_width, train_height,top_crop, normalize_image_scale = True, nscale_w = 2560, nscale_h = 1440)
        elif  dataset_name == 'custom':
            pipe = ExternalSourceTrainPipeline1(batch_size, num_threads, shard_id, eii, train_width, train_height,top_crop, normalize_image_scale = True, nscale_w = 2560, nscale_h = 1440)
        else:
            pipe = ExternalSourceTrainPipeline(batch_size, num_threads, shard_id, eii, train_width, train_height,top_crop)
            # print("pipe:  ",pipe)
       
        self.pii = DALIGenericIterator(pipe, output_map = ['images', 'seg_images', 'points','labels_cls'], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        self.eii_n = eii.n   #^ len(list/train_gt.txt)
        self.batch_size = batch_size

        self.interp_loc_row = torch.tensor(row_anchor, dtype=torch.float32).cuda() * self.original_image_height  # row向的线
        self.interp_loc_col = torch.tensor(col_anchor, dtype=torch.float32).cuda() * self.original_image_width   # col向的线
        # print('self.interp_loc_row: ',self.interp_loc_row,self.original_image_height )  # 590
        # print('self.interp_loc_col: ',self.interp_loc_col,self.original_image_width  )  # 1640
        self.num_cell_row = num_cell_row
        self.num_cell_col = num_cell_col

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.pii)
        images = data[0]['images']   # img
        seg_images = data[0]['seg_images']  # mask
        points = data[0]['points']   # point点坐标，xy
        labels_cls = data[0]['labels_cls']   #^ 分类 
        labels_cls = torch.tensor(labels_cls,dtype=torch.int64).cuda()
        
        points_row = my_interp.run(points, self.interp_loc_row, 0)  # row
        # print('points_row:  ',points_row.shape)
        points_row_extend = self._extend(points_row[:,:,:,0]).transpose(1,2)    # row x  -99999
        # print('points_row_extend:  ',points_row_extend.shape)
        labels_row = (points_row_extend / self.original_image_width * (self.num_cell_row - 1)).long()
        labels_row[points_row_extend < 0] = -1  #^ -99999 
        labels_row[points_row_extend > self.original_image_width] = -1  #^ 大于原始图像框 -1
        labels_row[labels_row < 0] = -1   #^ 插值里面小于0的
        labels_row[labels_row > (self.num_cell_row - 1)] = -1  #^ num_cell_row以外的

        points_col = my_interp.run(points, self.interp_loc_col, 1)  # col  y
        points_col = points_col[:,:,:,1].transpose(1,2)
        labels_col = (points_col / self.original_image_height * (self.num_cell_col - 1)).long()
        labels_col[points_col < 0] = -1
        labels_col[points_col > self.original_image_height] = -1
        
        labels_col[labels_col < 0] = -1
        labels_col[labels_col > (self.num_cell_col - 1)] = -1

        labels_row_float = points_row_extend / self.original_image_width
        labels_row_float[labels_row_float<0] = -1
        labels_row_float[labels_row_float>1] = -1

        labels_col_float = points_col / self.original_image_height
        labels_col_float[labels_col_float<0] = -1
        labels_col_float[labels_col_float>1] = -1
        # print('labels_row:  ',labels_row,labels_row.shape)
        # print('labels_col:  ',labels_col,labels_col.shape)
        # print('labels_row_float:  ',labels_row_float,labels_row_float.shape)
        # print('labels_col_float:  ',labels_col_float,labels_col_float.shape)
        return {'images':images,                        # img
                'seg_images':seg_images,                # mask
                'labels_cls':labels_cls,                #^ 分类 
                'labels_row':labels_row,                # 类别   类别 
                'labels_col':labels_col,                # 类别  类别 
                'labels_row_float':labels_row_float,    # 是否存在
                'labels_col_float':labels_col_float,}    # 是否存在
    
    def __len__(self):
        return int((self.eii_n + self.batch_size - 1) / self.batch_size)
    def reset(self):
        self.pii.reset()
    next = __next__

    def _extend(self, coords):
        # coords : n x num_lane x num_cls
        n, num_lanes, num_cls = coords.shape
        coords_np = coords.cpu().numpy()
        coords_axis = np.arange(num_cls)
        fitted_coords = coords.clone()
        for i in range(n):
            for j in range(num_lanes):
                lane = coords_np[i,j]
                if lane[-1] > 0:
                    continue

                valid = lane > 0
                num_valid_pts = np.sum(valid)
                if num_valid_pts < 6:
                    continue
                p = np.polyfit(coords_axis[valid][num_valid_pts//2:], lane[valid][num_valid_pts//2:], deg = 1)   
                start_point = coords_axis[valid][num_valid_pts//2]
                fitted_lane = np.polyval(p, np.arange(start_point, num_cls))

                
                fitted_coords[i,j,start_point:] = torch.tensor(fitted_lane, device = coords.device)
        return fitted_coords
    def _extend_col(self, coords):
        pass


class TestCollect:
    def __init__(self, batch_size, num_threads, data_root, list_path, shard_id, num_shards):
        self.batch_size = batch_size
        eii = LaneExternalIterator(data_root, list_path, batch_size=batch_size, shard_id=shard_id, num_shards=num_shards, mode = 'test')
        pipe = ExternalSourceTestPipeline(batch_size, num_threads, shard_id, eii)
        self.pii = DALIGenericIterator(pipe, output_map = ['images', 'names'], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        self.eii_n = eii.n
    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.pii)
        images = data[0]['images']
        names = data[0]['names']
        restored_names = []
        for name in names:
            if name[-1] == -1:
                restored_name = ''.join(list(map(chr,name[:-1])))
            else:
                restored_name = ''.join(list(map(chr,name)))   #! 转回string类型  
            restored_names.append(restored_name)
            
        out_dict = {'images': images, 'names': restored_names}    
        return out_dict
    
    def __len__(self):
        return int((self.eii_n + self.batch_size - 1) / self.batch_size)

    def reset(self):
        self.pii.reset()
    next = __next__


