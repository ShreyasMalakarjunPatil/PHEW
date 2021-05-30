#PHEW : Constructing Sparse Networks that Learn Fast and Generalize Well Without Training Data



This repository is the official implementation of [PHEW](https://arxiv.org/abs/2010.11354). 


## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```
## Code Details (PHEW)

PHEW : If you are already working with a code base and just wish to add PHEW as a baseline, just copy the script PHEW/phew.py 
and call the function within it.

```train
weight_masks, bias_masks = phew.phew_masks(network, prune_percentage, verbose = True, kernel_conserved = False)
```

The function returns two masks, the first is for weight matrices and the second is for biases, which can be used for batchnorm as well.

## Code Details (Experiments)

The experiments have been divided into two sections. First, section consist of all the baseline pruning methods and second all the ablation studies conducted with PHEW and the baselines.

Run the following command for a detailed list of experiment baseline choices and hyper-parameters.

```train
python3 main.py --help
```

Please find an example of command for a pruning method. The choices for the baselines are SNIP, GraSP, SynFlow, SynFlowL2, MAG, Random, PHEW, PHEW_Res

```train
python3 main.py --experiment PHEW --model ResNet20 --dataset cifar10 --optimizer momentum --epochs 160 
```

Please find an example of command for a particular ablation.

```train
python3 main.py --experiment Ablation --ablation_experiment Normal_Init_PHEW --model ResNet20 --dataset cifar10 --optimizer momentum --epochs 160 
```

For detailed list of hyper-parameters used for each of the experiments please refer to the appendix section of our paper.
 
## Contributing



## Citation