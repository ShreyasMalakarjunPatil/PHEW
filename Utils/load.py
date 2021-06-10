# Based on the Code provided by authors of Pruning neural networks without any data by iteratively conserving synaptic flow.
# # We thank the authors of SynFlow. https://github.com/ganguli-lab/Synaptic-Flow

import torch
from torchvision import datasets, transforms
import torch.optim as optim
from Utils import custom_dataset
from Models import vgg_cifar
from Models import vgg_tinyimagenet
from Models import resnet_cifar
from Models import resnet_cifar2
from Models import resnet_tinyimagenet

# Use gpu if available and set the gpu number to the one entered by the user.
def device(gpu):
    use_cuda = torch.cuda.is_available()
    print('Use Cuda',use_cuda)
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")

# Returns the image dimensions and the number of classes in the dataset provided by the user.
def dimension(dataset):
    if dataset == 'mnist':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny-imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes

def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)

def dataloader(dataset, batch_size, train, workers, prune_size = None):

    if dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        transform = get_transform(size=28, padding=0, mean=mean, std=std, preprocess=False)
        dataset = datasets.MNIST('Data', train=train, download=True, transform=transform)

    if dataset == 'cifar10':
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR10('Data', train=train, download=True, transform=transform)

    if dataset == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR100('Data', train=train, download=True, transform=transform)

    if dataset == 'tiny-imagenet':
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = get_transform(size=64, padding=4, mean=mean, std=std, preprocess=train)
        dataset = custom_dataset.TINYIMAGENET('Data', train=train, download=True, transform=transform)

    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    shuffle = train is True
    if prune_size is not None:
        indices = torch.randperm(len(dataset))[:prune_size]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                             shuffle=shuffle, **kwargs)

    return dataloader

def model(architecture, dataset):
    print(architecture)
    if dataset == 'cifar10' or 'cifar100':
        models = {
            'vgg11': vgg_cifar.vgg11,
            'vgg11-bn': vgg_cifar.vgg11_bn,
            'vgg13': vgg_cifar.vgg13,
            'vgg13-bn': vgg_cifar.vgg13_bn,
            'vgg16': vgg_cifar.vgg16,
            'vgg16-bn': vgg_cifar.vgg16_bn,
            'vgg19': vgg_cifar.vgg19,
            'vgg19-bn': vgg_cifar.vgg19_bn,
            'ResNet18': resnet_cifar.ResNet18,
            'ResNet34': resnet_cifar.ResNet34,
            'ResNet50': resnet_cifar.ResNet50,
            'ResNet101': resnet_cifar.ResNet101,
            'ResNet152': resnet_cifar.ResNet152,
            'ResNet20' : resnet_cifar2.resnet20,
            'Wide_ResNet20': resnet_cifar2.wide_resnet20,
            'ResNet32' : resnet_cifar2.resnet32
        }
    if dataset == 'tiny-imagenet':
        models = {
            'vgg11': vgg_tinyimagenet.vgg11,
            'vgg11-bn': vgg_tinyimagenet.vgg11_bn,
            'vgg13': vgg_tinyimagenet.vgg13,
            'vgg13-bn': vgg_tinyimagenet.vgg13_bn,
            'vgg16': vgg_tinyimagenet.vgg16,
            'vgg16-bn': vgg_tinyimagenet.vgg16_bn,
            'vgg19': vgg_tinyimagenet.vgg19,
            'vgg19-bn': vgg_tinyimagenet.vgg19_bn,
            'ResNet34': resnet_tinyimagenet.resnet34,
            'ResNet18' : resnet_tinyimagenet.resnet18
        }

    return models[architecture]

def optimizer(optimizer):
    optimizers = {
        'adam' : (optim.Adam, {}),
        'sgd' : (optim.SGD, {}),
        'momentum' : (optim.SGD, {'momentum' : 0.9, 'nesterov' : True}),
        'rms' : (optim.RMSprop, {})
    }
    return optimizers[optimizer]