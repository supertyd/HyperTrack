import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, TNL2k, Whisper
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader, HSI_loader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["Whisper"]
        if name == "Whisper":
            if settings.script_name == "hypertrack_nir":
                datasets.append(Whisper(settings.env.whisper_nir_dir, image_loader=image_loader))
            elif settings.script_name == "hypertrack_rednir":
                datasets.append(Whisper(settings.env.whisper_rednir_dir, image_loader=image_loader))
            elif settings.script_name == "hypertrack_vis":
                datasets.append(Whisper(settings.env.whisper_vis_dir, image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transformn
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2, normalize=True),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    # transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
    #                                 tfm.RandomHorizontalFlip_Norm(probability=0.5))




    # transform_val = tfm.Transform(tfm.ToTensor(),
    #                               tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    print("sampler_mode", sampler_mode)

    data_processing_train = processing.hypertrackProcessing(search_area_factor=search_area_factor,
                                                        output_sz=output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        transform=transform_train,
                                                        joint_transform=transform_joint,
                                                        settings=settings,
                                                        train_score=train_score)

    # data_processing_val = processing.MixformerProcessing(search_area_factor=search_area_factor,
    #                                                      output_sz=output_sz,
    #                                                      center_jitter_factor=settings.center_jitter_factor,
    #                                                      scale_jitter_factor=settings.scale_jitter_factor,
    #                                                      mode='sequence',
    #                                                      transform=transform_val,
    #                                                      joint_transform=transform_joint,
    #                                                      settings=settings,
    #                                                      train_score=train_score)


    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, HSI_loader),    #  这个需要改成whisper的数据集loader，然后image的loader需要改成unchannel的
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    # dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
    #                                       p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
    #                                       samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
    #                                       max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
    #                                       num_template_frames=settings.num_template, processing=data_processing_val,
    #                                       frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)
    # val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    # loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
    #                        num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
    #                        epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train


def get_optimizer_scheduler(net, cfg):
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    freeze_first_6layers = getattr(cfg.TRAIN, "FREEZE_FIRST_6LAYERS", False)
    train_trans = getattr(cfg.TRAIN, "TRAIN_Trans", False)

    # for name,param in net.named_parameters():
    #     print(name,param.requires_grad,param.shape)

    if train_trans:
        print("Only training Trans_branch. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "band" or "backbone" in n and p.requires_grad]}
        ]
        # param_dicts = [
        #     {"params": [p for n, p in net.named_parameters() if "band" in n and p.requires_grad]}
        # ]
        # for n, p in net.named_parameters():
        #     if "bandsgate" not in n:
        #         p.requires_grad = False
        #     else:
        #         if is_main_process():
        #             print(n)
        for n, p in net.named_parameters():
            if "band" not in n and "backbone" not in n:
                p.requires_grad = True
            else:
                if is_main_process():
                    print(n)
    elif train_score:
        print("Only training score_branch. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "score" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "score" not in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
    elif freeze_first_6layers:  # only for ViT-Large backbone
        assert "large_patch16" == cfg.MODEL.VIT_TYPE
        print("Freeze the first 6 layers of MixFormer vit backbone. Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if 'blocks.0.' in n or 'blocks.1.' in n or 'blocks.2.' in n or 'blocks.3.' in n or 'blocks.4.' in n or 'blocks.5.' in n \
                or 'patch_embed' in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]
    else: # train network except for score prediction module
        for n, p in net.named_parameters():
            if "score" in n:
                p.requires_grad = False
        param_dicts = [
            {
                "params": [p for n, p in net.named_parameters() if "backbone" or "band" in n and p.requires_grad],
            },
        ]

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.LR_DROP_EPOCH,
                                                            gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
