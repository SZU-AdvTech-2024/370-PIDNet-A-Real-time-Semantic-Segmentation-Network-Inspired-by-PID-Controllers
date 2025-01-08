crop_size = (
    680,
    680,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        72.39239876,
        82.90891754,
        73.15835921,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        680,
        680,
    ),
    std=[
        1,
        1,
        1,
    ],
    type='SegDataPreProcessor')
data_root = 'data/cityscapes/'
dataset_type = 'CityscapesDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        dilations=(
            2,
            4,
        ),
        in_channels=3,
        norm_cfg=dict(eps=0.001, requires_grad=True, type='SyncBN'),
        num_blocks=(
            3,
            21,
        ),
        num_channels=(
            32,
            64,
            128,
        ),
        reductions=(
            8,
            16,
        ),
        type='CGNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            72.39239876,
            82.90891754,
            73.15835921,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            680,
            680,
        ),
        std=[
            1,
            1,
            1,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        channels=256,
        concat_input=False,
        dropout_ratio=0,
        in_channels=256,
        in_index=2,
        loss_decode=dict(
            class_weight=[
                2.5959933,
                6.7415504,
                3.5354059,
                9.8663225,
                9.690899,
                9.369352,
                10.289121,
                9.953208,
                4.3097677,
                9.490387,
                7.674431,
                9.396905,
                10.347791,
                6.3927646,
                10.226669,
                10.241062,
                10.280587,
                10.396974,
                10.055647,
            ],
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(eps=0.001, requires_grad=True, type='SyncBN'),
        num_classes=19,
        num_convs=0,
        type='FCNHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(sampler=None),
    type='EncoderDecoder')
norm_cfg = dict(eps=0.001, requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(eps=1e-08, lr=0.001, type='Adam', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(eps=1e-08, lr=0.001, type='Adam', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=60000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root='data/cityscapes/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
total_iters = 60000
train_cfg = dict(max_iters=60000, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root='data/cityscapes/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    1024,
                ),
                type='RandomResize'),
            dict(crop_size=(
                680,
                680,
            ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            1024,
        ),
        type='RandomResize'),
    dict(crop_size=(
        680,
        680,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root='data/cityscapes/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
