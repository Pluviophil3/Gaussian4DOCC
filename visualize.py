try:
    from vis import save_occ, save_gaussian
except:
    print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
import shutil
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    
    cfg.val_dataset_config.update({
        "vis_indices": args.vis_index,
        "num_samples": args.num_samples})

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)
    
    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        raw_model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
         True, 17, filter_minmax=False)
    miou_metric.reset()

    my_model.eval()
    os.environ['eval'] = 'true'
    if args.vis_occ or args.vis_gaussian:
        os.makedirs(os.path.join(args.work_dir, 'vis'), exist_ok=True)

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            input_data, prev_data = data
            # print(f'data: {input_data}')
            for k in list(input_data.keys()):
                if isinstance(input_data[k], torch.Tensor):
                    input_data[k] = input_data[k].cuda()
            for k in list(prev_data.keys()):
                if isinstance(prev_data[k], torch.Tensor):
                    prev_data[k] = prev_data[k].cuda()
            input_imgs = input_data.pop('img')
            # img_paths = input_data.pop('img_dict')[0]

            # for cam, path in img_paths.items():
            #     print(f'{cam} {path}')
            #     root = os.path.join(args.work_dir, 'vis')
            #     new_path = os.path.join(root,'val_'+ str(i_iter_val) +'_' + cam+'.jpg')
            #     shutil.copy(path, new_path)

            prev_imgs = prev_data.pop('img')
            imgs = (input_imgs, prev_imgs)
            data = (input_data, prev_data)  
            
            result_dict = my_model(imgs=imgs, metas=data)
            for idx, pred in enumerate(result_dict['pred_occ'][-1]):
                pred_occ = pred.argmax(0)
                gt_occ = result_dict['sampled_label'][idx]
                if args.vis_occ:
                    save_occ(
                        os.path.join(args.work_dir, 'vis'),
                        pred_occ.reshape(1, 200, 200, 16),
                        f'val_{i_iter_val}_pred',
                        True, 0)
                    save_occ(
                        os.path.join(args.work_dir, 'vis'),
                        gt_occ.reshape(1, 200, 200, 16),
                        f'val_{i_iter_val}_gt',
                        True, 0)
                if args.vis_gaussian:
                    save_gaussian(
                        os.path.join(args.work_dir, 'vis'),
                        result_dict['gaussian'], gt_occ.reshape(1, 200, 200, 16),
                        f'val_{i_iter_val}_gaussian')
                    
                miou_metric._after_step(pred_occ, gt_occ)
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
                    
    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()
    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    # fps_list = [range(1,4090, 20)]
    # fps_list = [range(0,600)]
    # fps_list = [range(0,4000,10)]
    # o_list = range(0,4000,10)
    # fps_list = [o_list[382],o_list[23], o_list[40]]
    # fps_list = [o_list[382],o_list[23],o_list[40]]
    # HGF ablation例子
    # idx = [91,108,68,81,167,166,191,175]
    # fps_list = [i * 20 + 1 for i in idx]
    # fps_list = [3010, 1818, 2560, 2276, 1314, 2320, 3630, 4010, 4060, 4090]
    # fps_list = [range(17, 27), range(43, 48), range(66, 79), range(153, 160), range(201, 226), 
    #             339, 346, 347, 351, 358,376,386,387,392,401,404,408, 411,range(414,440),477,481,range(532,562),
    #             608, 626, 641, 650, 668, 699, 731, range(735, 737), 742, 748, range(798, 804), 806, range(813, 815), 850, 859, 860, 892,
    #             910, 923,951,988,989,1024,1029,range(1039, 1041),1052,1073,1082,1087,1087,1094,
    #             1352,1378,1382,1391,1415,1459,1478,1509,1523, range(2152, 2167), range(2175, 2183), 2193, 2206, range(2235, 2245), range(2271, 2276), range(2314, 2323)
    #             ]
    # fps_list = [range(15, 80), range(330, 500), range(734, 815), range(890, 1100)] 
    # expanded_fps_list = []
    # for item in fps_list:
    #     if isinstance(item, range):
    #         expanded_fps_list.extend(item)
    #     else:
    #         expanded_fps_list.append(item)

    # fps_list = expanded_fps_list
    # print(len(fps_list))
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--vis-gaussian', action='store_true', default=False)
    parser.add_argument('--vis-index', type=int, nargs='+', default=[])
    # parser.add_argument('--vis-index', type=int, nargs='+', default=fps_list)
    parser.add_argument('--num-samples', type=int, default=1)
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
