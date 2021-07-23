# PHEW : Constructing Sparse Networks that Learn Fast and Generalize Well Without Training Data



This repository is the official implementation of [PHEW](http://proceedings.mlr.press/v139/patil21a.html). 


### Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```
### Code Details (PHEW)

If you are already working with a code base and just wish to add PHEW as a baseline, just copy the script PHEW / phew.py 
and call the function within it.

```train
weight_masks, bias_masks = phew.phew_masks(network, prune_percentage, verbose = True, kernel_conserved = False)
```

The function returns two masks, the first is for weight matrices and the second is for biases, which can be used for batchnorm as well.

## Code Details 

A detailed description of the code base is provided below. 
### Datasets

The code base currently supports three classification datasets, CIFAR-10, CIFAR-100 and Tiny-ImageNet. The data would automatically be downloaded for all the three datasets.

### Prune

Given a network at initialization and a desired pruning ratio,

1. PHEW : The mask is initialized with all zeros and then the connections are placed according to the random walks.
2. PHEW_Res : The mask is initialized using a previously used PHEW mask at a higher pruning ratio, and connections are added using the same random walk procedure.

### Experiments

Run the following command for a detailed list of experiment baseline choices and hyper-parameters.

```train
python3 main.py --help
```

#### SynFlow and SynFlowL2 increasing width experiments

Please find the command for running increasing width experiments with SynFlow and SynFlowL2.

```train
python3 main.py --experiment Ablation --ablation_experiment SynFlow_Width --model ResNet20 --dataset cifar10 --optimizer momentum --epochs 160 --prune_perc 93.1 

python3 main.py --experiment Ablation --ablation_experiment SynFlowL2_Width --model ResNet20 --dataset cifar10 --optimizer momentum --epochs 160 --prune_perc 93.1 
```

#### Baselines

Please find an example of command for a pruning method. 
```train
python3 main.py --experiment PHEW_Res --model ResNet20 --dataset cifar10 --optimizer momentum --epochs 160 
```

#### Ablations

Please find an example of command for a particular ablation.

```train
python3 main.py --experiment Ablation --ablation_experiment Normal_Init_PHEW --model ResNet20 --dataset cifar10 --optimizer momentum --epochs 160 
```

For detailed list of hyper-parameters used for each of the experiments please refer to the appendix section of our paper.


### Contributing

Please feel free to contact us or submit issues regarding suggestions for improving the repository or any specific implementation details you wish to know about. 

### Acknowledgement

We thank the authors of [GraSP](https://github.com/alecwangcq/GraSP) and [SynFlow](https://github.com/ganguli-lab/Synaptic-Flow) for making their code public, many of the scripts used are taken and modified from those repositories. We have provided citations in individual files.

### Citation

Please cite our [paper](http://proceedings.mlr.press/v139/patil21a.html) if you find this repository useful.