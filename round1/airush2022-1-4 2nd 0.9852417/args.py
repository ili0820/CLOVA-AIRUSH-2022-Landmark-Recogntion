import albumentations as A

class args:
    DEBUG = False
    num_workers = 8
    gpus = '0'
    distributed_backend = None
    sync_batchnorm = True
    gradient_accumulation_steps = 4
    precision = 16
    warmup_epo = 1
    cosine_epo = 29
    lr = 0.002
    weight_decay = 0.0001
    p_trainable = True
    crit = 'bce'
    # backbone = 'efficientnet_b0'
    backbone = 'tinynet_e'
    embedding_size = 512
    pool = 'gem'
    arcface_s = 45.0
    arcface_m = 0.4
    neck = 'option-D'
    head = 'arc_margin'
    pretrained_weights = None
    optim = 'adamw'
    batch_size = 128
    n_splits = 5
    fold = 0
    seed = 42
    device = 'cuda:0'
    out_dim = 1049
    n_classes = 619
    class_weights = 'log'
    class_weights_norm = 'batch'
    normalization = 'imagenet'
    # normalization = 'inception'
    img_size = 512
    print_iter=50

    


tr_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),    
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.Resize(args.img_size, args.img_size),
        A.Cutout(max_h_size=int(args.img_size * 0.4), max_w_size=int(args.img_size * 0.4), num_holes=1, p=0.5),
    ])

val_aug = A.Compose([
        A.ImageCompression(quality_lower=99, quality_upper=100),    
        A.Resize(args.img_size, args.img_size),
    ])