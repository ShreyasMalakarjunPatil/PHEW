import argparse
from Experiments import Baseline_FC # Dense baseline network
from Experiments import SNIP
from Experiments import GRASP
from Experiments import SynFlow
from Experiments import SynFlowL2
from Experiments import Random
from Experiments import MAG # Magnitude Pruning
from Experiments import PHEW_Res # PHEW with increasing density and using masks from a lower density level.
from Experiments import PHEW # PHEW starting with zero masks.
from Experiments.Ablations import Ablation_main

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Network Construction')
    parser.add_argument('--experiment', type=str, default='PHEW',
                        choices=['Baseline_FC', 'PHEW', 'SNIP', 'GRASP', 'IMAG', 'Random', 'SynFlow', 'SynFlowL2',
                                 'MAG', 'PHEW_Res','Ablation'])
    parser.add_argument('--ablation_experiment', type=str, default='SynFlow_Width',
                        choices=['SynFlow_Width', 'SynFlowL2_Width','SynFlowL2_LWRS', 'SynFlow_LWRS',
                                 'Kernel_PHEW', 'Normal_Init_PHEW', 'Xavier_Init_PHEW',
                                 'Inverse_PHEW', 'Uniform_PHEW', 'PHEW2',
                                 'ReInit_PHEW', 'ReInit_Evci_PHEW', 'ReInit_Liu_PHEW', 'ReInit_Evci_Random', 'ReInit_Liu_Random',
                                 'Normal_Init_SNIP', 'Xavier_Init_SNIP', 'Normal_Init_IMAG', 'Xavier_Init_IMAG'])
    parser.add_argument('--expid', type=str, default='0')
    parser.add_argument('--model', type=str, default='ResNet20',
                        choices=['vgg11', 'vgg11-bn', 'vgg13', 'vgg13-bn', 'vgg16', 'vgg16-bn', 'vgg19', 'vgg19-bn',
                                 'ResNet20','ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'lenet5', 'ResNet32', ])
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['mnist', 'cifar10', 'cifar100', 'tiny-imagenet'])
    parser.add_argument('--optimizer', type=str, default='momentum',
                        choices=['sgd', 'adam', 'momentum', 'rms'])
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_drops', type=int, nargs='*', default=[80, 120])
    parser.add_argument('--lr_drop_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--prune_perc', type=float,
                        default=[48.8, 59.0, 67.2, 73.8, 79.0, 83.2, 86.6, 89.3, 91.4, 93.1, 94.5, 95.6, 96.5, 97.2, 97.7, 98.2]) # [10.0,20.0,29.8,36.0,43.5,48.8,54.3,59.0,63.4,67.2,73.8,79.0,83.2,86.6,89.3,93.1,95.6]
    parser.add_argument('--pre_epochs', type=int, default=160) # For Magnitude Pruning after training
    parser.add_argument('--prune_iterations', type=int, default=10) # For Magnitude Pruning after training
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--prune_dataset_size', type=int, default=10) # For SNIP and GraSP, number of samples per class
    parser.add_argument('--prune_batch_size', type=int, default=256) # For SNIP and GraSP
    parser.add_argument('--synflow_iterations', type=int, default=100) # For SynFlow and SynFlowL2
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print(args.experiment)

    if args.experiment == 'Baseline_FC':
        Baseline_FC.run(args)

    if args.experiment == 'PHEW_Res':
        PHEW_Res.run(args)

    if args.experiment == 'PHEW':
        PHEW.run(args)

    if args.experiment == 'SNIP':
        SNIP.run(args)

    if args.experiment == 'GRASP':
        GRASP.run(args)

    if args.experiment == 'SynFlow':
        SynFlow.run(args)

    if args.experiment == 'SynFlowL2':
        SynFlowL2.run(args)

    if args.experiment == 'Random':
        Random.run(args)

    if args.experiment == 'MAG':
        MAG.run(args)

    if args.experiment == 'Ablation':
        Ablation_main.run(args)