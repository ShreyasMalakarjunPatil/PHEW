from Experiments.Ablations import SynFlow_Width
from Experiments.Ablations import SynFlowL2_Width
from Experiments.Ablations import SynFlow_LWRS
from Experiments.Ablations import SynFlowL2_LWRS

from Experiments.Ablations import Kernel_PHEW
from Experiments.Ablations import Normal_Init_PHEW
from Experiments.Ablations import Xavier_Init_PHEW
from Experiments.Ablations import Inverse_PHEW
from Experiments.Ablations import Uniform_PHEW
from Experiments.Ablations import PHEW2

from Experiments.Ablations import ReInit_PHEW
from Experiments.Ablations import ReInit_Evci_PHEW
from Experiments.Ablations import ReInit_Liu_PHEW
from Experiments.Ablations import ReInit_Evci_Random
from Experiments.Ablations import ReInit_Liu_Random

def run(args):

    if args.ablation_experiment == 'SynFlow_Width':
        SynFlow_Width.run(args)
    if args.ablation_experiment == 'SynFlowL2_Width':
        SynFlowL2_Width.run(args)
    if args.ablation_experiment == 'SynFlow_LWRS':
        SynFlow_LWRS.run(args)
    if args.ablation_experiment == 'SynFlowL2_LWRS':
        SynFlowL2_LWRS.run(args)

    if args.ablation_experiment == 'Kernel_PHEW':
        Kernel_PHEW.run(args)
    if args.ablation_experiment == 'Normal_Init_PHEW':
        Normal_Init_PHEW.run(args)
    if args.ablation_experiment == 'Xavier_Init_PHEW':
        Xavier_Init_PHEW.run(args)
    if args.ablation_experiment == 'Uniform_PHEW':
        Uniform_PHEW.run(args)
    if args.ablation_experiment == 'Inverse_PHEW':
        Inverse_PHEW.run(args)
    if args.ablation_experiment == 'PHEW2':
        PHEW2.run(args)

    if args.ablation_experiment == 'ReInit_PHEW':
        ReInit_PHEW.run(args)
    if args.ablation_experiment == 'ReInit_Evci_PHEW':
        ReInit_Evci_PHEW.run(args)
    if args.ablation_experiment == 'ReInit_Liu_PHEW':
        ReInit_Liu_PHEW.run(args)
    if args.ablation_experiment == 'ReInit_Evci_Random':
        ReInit_Evci_Random.run(args)
    if args.ablation_experiment == 'ReInit_Liu_Random':
        ReInit_Liu_Random.run(args)