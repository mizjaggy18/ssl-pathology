import random
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F

def load_model_weights(model, weights):

    model_dict = model.state_dict()
    # print("model_dict: ", model_dict.keys())
    # print("weights: ", weights.keys())

    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model


def reload_weights(args, model, optimizer):
    # Load the pretrained model
    print(args.device.type)
    checkpoint = torch.load(args.load_checkpoint_dir, map_location=args.device.type)

    ## reload weights for training of the linear classifier
    if 'model' in checkpoint.keys():
        model.load_state_dict(checkpoint['model'])
    else:
        # model.load_state_dict(checkpoint)

        # mod for loading ozanciga's ckpt
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', 'f.')] = state_dict.pop(key)

        model = load_model_weights(model, state_dict)


    ## reload weights and optimizers for continuing training
    if args.start_epoch > 0:
        print("Continuing training from epoch ", args.start_epoch)

        try:
            optimizer.load_state_dict(checkpoint['optimiser'])
        except KeyError:
            raise KeyError('Sry, no optimizer saved. Set start_epoch=0 to start from pretrained weights')

    return model, optimizer


def distribute_over_GPUs(opt, model):
    ## distribute over GPUs
    if opt.device.type != "cpu":
        model = nn.DataParallel(model)
        num_GPU = torch.cuda.device_count()
        opt.batch_size_multiGPU = opt.batch_size * num_GPU
    else:
        model = nn.DataParallel(model)
        opt.batch_size_multiGPU = opt.batch_size

    model = model.to(opt.device)
    print("Let's use", num_GPU, "GPUs!")

    return model, num_GPU

def validate_arguments(opt):
    if not opt.use_album:
        # Albumnetations are needed if these augmentations are to be used
        if (opt.rgb_grid_distort_p > 0):
            raise ValueError('Grid distort needs use_album to be true')
        if (opt.rgb_grid_shuffle_p > 0):
            raise ValueError('Grid shuffle needs use_album to be true')
        if (opt.rgb_contrast_p > 0):
            raise ValueError('Contrast needs use_album to be true')

    if not opt.data_input_dir_test:
        opt.data_input_dir_test = opt.data_input_dir

    return opt
