# Copyright (c) Howto100M authors and Facebook, Inc. All Rights Reserved

import torch as th

from torch import nn


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])


def get_model(args):
    assert args.type in ['2d', '3d', 'vmz', 's3d', 'vae']
    if args.type == '2d':
        print('Loading 2D-ResNet-152 ...')
        import torchvision.models as models
        model = models.resnet152(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        model = model.cuda()
    elif args.type == 'vmz':
        print('Loading VMZ ...')
        from vmz34 import r2plus1d_34
        model = r2plus1d_34(pretrained_path=args.vmz_model_path, pretrained_num_classes=487)
        model = model.cuda()
    elif args.type == 's3d':
        # we use one copy of s3d instead of dup another one for feature extraction.
        from mmpt.processors.models.s3dg import S3D
        model = S3D('pretrained_models/s3d_dict.npy', 512)
        model.load_state_dict(th.load('pretrained_models/s3d_howto100m.pth'))
        model = model.cuda()

    elif args.type == '3d':
        print('Loading 3D-ResneXt-101 ...')
        from videocnn.models import resnext
        model = resnext.resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            last_fc=False)
        model = model.cuda()
        model_data = th.load(args.resnext101_model_path)
        model.load_state_dict(model_data)
    elif args.type == 'vae':
        from openaivae import OpenAIParallelDiscreteVAE
        model = OpenAIParallelDiscreteVAE()
        model = model.cuda()
    else:
        raise ValueError("model not supported yet.")

    model.eval()
    print('loaded')
    return model
