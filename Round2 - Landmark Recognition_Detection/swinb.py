_base_ = [
    'swinb/coco_detection.py',
    'swinb/schedule.py', 'swinb/runtime.py','swinb/atssswin.py'
]


# 총 epochs 사이즈
runner = dict(max_epochs=10)

# samples_per_gpu -> batch size라 생각하면 됨
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

checkpoint_config = dict(interval=-1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512, 1024])
    
)

find_unused_parameters = True