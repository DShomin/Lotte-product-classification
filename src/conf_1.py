import albumentations as A

args = dict()
args['DEBUG'] = False
args['num_workers'] = 8
args['gpus'] = '0'
args['distributed_backend'] = None
args['sync_batchnorm'] = True
args['gradient_accumulation_steps'] = 4
args['precision'] = 16
args['warmup_epo'] = 1
args['cosine_epo'] = 29
args['lr'] = 0.002
args['weight_decay'] = 0.0001
args['p_trainable'] = True
args['crit'] = 'bce'
args['backbone'] = 'tf_efficientnet_b1_ns'
args['embedding_size'] = 512
args['pool'] = 'gem'
args['arcface_s'] = 45.0
args['arcface_m'] = 0.4
args['neck'] = 'option-D'
args['head'] = 'arc_margin'
args['pretrained_weights'] = None
args['optim'] = 'sgd'
args['batch_size'] = 32
args['n_splits'] = 5
args['fold'] = 0
args['seed'] = 9999
args['device'] = 'cuda:0'
args['out_dim'] = 1049
args['n_classes'] = 1000
args['class_weights'] = 'log'
args['class_weights_norm'] = 'batch'
args['normalization'] = 'imagenet'
args['crop_size'] = 448
args['tr_aug'] = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),    
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.Cutout(max_h_size=int(256 * 0.4), max_w_size=int(256 * 0.4), num_holes=1, p=0.5),
    ])
args['val_aug'] = A.Compose([
        A.ImageCompression(quality_lower=99, quality_upper=100),    
    ])