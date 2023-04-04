_base_ = 'paa/paa_r50_fpn_mstrain_3x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
load_from='https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_mstrain_3x_coco/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth'