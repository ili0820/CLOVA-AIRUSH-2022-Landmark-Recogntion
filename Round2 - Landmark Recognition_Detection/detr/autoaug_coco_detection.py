CLASSES = [
'place1', 'place2', 'place3', 'place4', 'place5', 'place6', 'place7', 'place8', 'place9', 'place10', 
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
'place137', 'place138', 'place139', 'place140', 'place141', 'place142', 'place143'

]

# dataset settings
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(
        type='MMAutoAugment',
        policies=[[
            dict(
                type='MMResize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='MMResize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='MMRandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='MMResize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'ori_img_shape',
                   'img_shape', 'pad_shape', 'scale_factor', 'flip',
                   'flip_direction', 'img_norm_cfg'))
]
test_pipeline = [
    dict(
        type='MMMultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='MMResize', keep_ratio=True),
            dict(type='MMRandomFlip'),
            dict(type='MMNormalize', **img_norm_cfg),
            dict(type='MMPad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'ori_img_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'))
        ])
]

train_dataset = dict(
    type='DetDataset',
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        classes=CLASSES,
        test_mode=False,
        filter_empty_gt=True,
        iscrowd=False),
    pipeline=train_pipeline)

val_dataset = dict(
    type='DetDataset',
    imgs_per_gpu=1,
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        classes=CLASSES,
        test_mode=True,
        filter_empty_gt=False,
        iscrowd=True),
    pipeline=test_pipeline)

data = dict(
    imgs_per_gpu=2, workers_per_gpu=2, train=train_dataset, val=val_dataset)

# evaluation
eval_config = dict(interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        evaluators=[
            dict(type='CocoDetectionEvaluator', classes=CLASSES),
        ],
    )
]