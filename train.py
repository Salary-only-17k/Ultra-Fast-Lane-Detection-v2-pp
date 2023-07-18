import copy
import datetime
import os
import time

import torch
from evaluation.eval_wrapper import eval_lane
from utils.common import (calc_loss, cp_projects, get_loader, get_logger,
                          get_model, get_work_dir, inference, merge_config,
                          save_model)
from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import (get_loss_dict, get_metric_dict, get_optimizer,
                           get_scheduler)
from utils.metrics import reset_metrics, update_metrics


def train(net,data_loader, loss_dict, optimizer, scheduler,logger, epoch, dataset, epoches, \
          work_dir=r'runs' \
          ):
    os.makedirs(work_dir,exist_ok=True)
    net.train()
    progress_bar = dist_tqdm(data_loader)
    progress_bar.desc = f'{epoch+1}/{epoches} Train'
    for b_idx, data_label in enumerate(progress_bar):
        global_step = epoch * len(data_loader) + b_idx
        results = inference(net, data_label, dataset)
        loss,loss_total = calc_loss(loss_dict, results, logger, global_step, epoch)
     
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        if global_step % 20 == 0:
            for loss_name, loss_value in loss_total.items():
                logger.add_scalar('train_' + loss_name, loss_value, global_step=global_step)
            if hasattr(progress_bar,'set_postfix'):
                new_kwargs = {}
                for k,v in loss_total.items():
                    # if 'lane' in k:
                    #     continue
                    new_kwargs[k] = v
           
                # progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                #                         **new_kwargs)
                progress_bar.set_postfix(**new_kwargs)
        return copy.deepcopy(optimizer)
   
    
               

def val(net,data_loader,optimizer,logger, epoch, metric_dict, dataset, \
          epoches, \
          best_score, \
          best_net, \
          work_dir=r'runs' \
          ):
    os.makedirs(work_dir,exist_ok=True)
    net.eval()
    progress_bar = dist_tqdm(data_loader)
    progress_bar.desc = f'{epoch+1}/{epoches} Val'
    with torch.no_grad():
        for b_idx, data_label in enumerate(progress_bar):
            global_step = epoch * len(data_loader) + b_idx

            results = inference(net, data_label, dataset)
        
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)
            # print(results)
            
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                print(f"{me_name} ->   {me_op.get():.4f}")
                logger.add_scalar('val metric/' + me_name, me_op.get(), global_step=global_step)
          
            if hasattr(progress_bar,'set_postfix'):
        
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
                new_kwargs = {}
                for k,v in kwargs.items():
                    # if 'lane' in k:
                    #     continue
                    new_kwargs[k] = v
           
                progress_bar.set_postfix( **new_kwargs)
                    
            local_acc = 0
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                if me_name in ['ext_row','ext_col','lane_cls_acc']:
                    local_acc += me_op.get()
            local_acc /= 3
            if local_acc > best_score:
                best_score = local_acc
                best_net = copy.deepcopy(net.state_dict())
                best_optimizer = copy.deepcopy(optimizer)
            else:
                continue
        return best_score,best_net,best_optimizer
               
         

if __name__ == "__main__":
    #~ 驱动分布式训练的  多gpu
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()
    if args.local_rank == 0:
        work_dir = get_work_dir(cfg)
        print()
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        if args.local_rank == 0:
            with open('.work_dir_tmp_file.txt', 'w') as f:
                f.write(work_dir)
        else:
            while not os.path.exists('.work_dir_tmp_file.txt'):
                time.sleep(0.1)
            with open('.work_dir_tmp_file.txt', 'r') as f:
                work_dir = f.read().strip()
    synchronize()
    # 结束
    
    if args.local_rank == 0:
        os.system('rm .work_dir_tmp_file.txt')
    
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    #~ 加载数据
    train_loader = get_loader(cfg)
    val_loader = get_loader(cfg,is_val=True)
    #~ 构建网络
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide', '34fca']
    net = get_model(cfg)  # build net

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    optimizer = get_optimizer(net, cfg)
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    # 重头训练
    # 只训练分类车道线
    
    if cfg.finetune is not None:
        """
        微调模型-det-head
        """
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune, map_location='cpu')['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        """
        加载预训练模型,加载作者提供的模型
        """
        dist_print('==> Resume model from ' + cfg.resume)

        resume_dict = torch.load(cfg.resume, map_location='cpu')['model']
        model_dict =  net.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)
        
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        # 只训练识别车道线
        resume_epoch = 0

    cfg.test_work_dir = work_dir
    cfg.distributed = distributed
    logger = get_logger(work_dir, cfg)
    # cp_projects(cfg.auto_backup, work_dir)
    #~ 开始训练
    max_res = 0
    
    best_score=0                                # 最佳精度
    best_net = copy.deepcopy(net.state_dict())  # 最佳模型
    best_optimizer = copy.deepcopy(optimizer)
    best_epoch = 0
    res = None
    for epoch in range(resume_epoch, cfg.epoch):
        tmp_optimizer = train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, cfg.dataset,
                           cfg.epoch,
                           'runs')
        train_loader.reset()
        best_score,best_net,best_optimizer = val(net,val_loader,tmp_optimizer,logger,epoch,metric_dict,cfg.dataset,
                            cfg.epoch,
                            best_score,
                            best_net,
                            'runs'
                            )
        val_loader.reset()
    
    save_model(best_net, best_optimizer, work_dir,best_score)          #^  最好的模型保存
    save_model(net.state_dict(), tmp_optimizer, work_dir, best_score,True)   #^ 最后模型保存

    logger.close()
    print("[best-acc] :",best_score)
    print("[save-pth] :",work_dir)