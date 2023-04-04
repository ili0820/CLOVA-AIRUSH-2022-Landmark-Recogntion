_base_ = [
    'cascadeswin/coco_detection.py',
    'cascadeswin/schedule.py', 'cascadeswin/runtime.py','cascadeswin/cascadeswin_origin.py'
]

# 총 epochs 사이즈
runner = dict(max_epochs=5)

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