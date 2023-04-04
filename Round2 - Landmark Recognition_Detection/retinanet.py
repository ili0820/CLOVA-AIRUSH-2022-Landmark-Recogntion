# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
CLASSES = ('place1', 'place2', 'place3', 'place4', 'place5', 'place6', 'place7', 'place8', 'place9', 'place10', 
'place11', 'place12', 'place13', 'place14', 'place15', 'place16', 'place17', 'place18', 'place19', 'place20', 
'place21', 'place22', 'place23', 'place24', 'place25', 'place26', 'place27', 'place28', 'place29', 'place30', 
'place31', 'place32', 'place33', 'place34', 'place35', 'place36', 'place37', 'place38', 'place39', 'place40', 
'place41', 'place42', 'place43', 'place44', 'place45', 'place46', 'place47', 'place48', 'place49', 'place50', 
'place51', 'place52', 'place53', 'place54', 'place55', 'place56', 'place57', 'place58', 'place59', 'place60', 
'place61', 'place62', 'place63', 'place64', 'place65', 'place66', 'place67', 'place68', 'place69', 'place70', 
'place71', 'place72', 'place73', 'place74', 'place75', 'place76', 'place77', 'place78', 'place79', 'place80', 
'place81', 'place82', 'place83', 'place84', 'place85', 'place86', 'place87', 'place88', 'place89', 'place90', 
'place91', 'place92', 'place93', 'place94', 'place95', 'place96', 'place97', 'place98', 'place99', 'place100', 
'place101', 'place102', 'place103', 'place104', 'place105', 'place106', 'place107', 'place108', 'place109', 
'place110', 'place111', 'place112', 'place113', 'place114', 'place115', 'place116', 'place117', 'place118', 
'place119', 'place120', 'place121', 'place122', 'place123', 'place124', 'place125', 'place126', 'place127', 
'place128', 'place129', 'place130', 'place131', 'place132', 'place133', 'place134', 'place135', 'place136', 
'place137', 'place138', 'place139', 'place140', 'place141', 'place142', 'place143')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=len(CLASSES),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.8,
            neg_iou_thr=0.2,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
load_from='https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_mstrain_3x_coco/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pthhttps://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_mstrain_3x_coco/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth'
# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=10)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
