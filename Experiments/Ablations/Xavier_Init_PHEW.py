import torch
import torch.nn as nn
from Utils import load
from Utils import train
from Prune import phew_utils
import copy
import pickle as pkl
from Prune import Utils

def run(args):
    torch.manual_seed(args.seed)
    dev = load.device(args.gpu)

    input_shape, num_classes = load.dimension(args.dataset)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.train_batch_size, False, args.workers)

    model = load.model(args.model, args.dataset)(input_shape, num_classes)
    model._initialize_weights_uniform()
    model.to(dev)
    prob, reverse_prob, kernel_prob = phew_utils.generate_probability(model, verbose = True)

    net = copy.deepcopy(model)
    num = 0
    for p in net.parameters():
        p.data.fill_(0)
        if len(p.data.size()) != 1:
            num = num + 1
    weight_masks, bias_masks = phew_utils.generate_masks(net)
    for i in range(len(args.prune_perc)):
        print(args.experiment, str(args.prune_perc[len(args.prune_perc) - i - 1]), str(args.model), str(args.dataset))

        prune_perc = args.prune_perc[len(args.prune_perc)-i-1]
        sparse_model = copy.deepcopy(model)
        loss = nn.CrossEntropyLoss()
        opt, opt_kwargs = load.optimizer(args.optimizer)

        weight_masks, bias_masks = phew_utils.phew_masks(model, prune_perc, prob, reverse_prob, kernel_prob,weight_masks,bias_masks,verbose=True,kernel_conserved=False)
        sparse_model.set_masks(weight_masks, bias_masks)
        sparse_model.to(dev)

        optimizer = opt(sparse_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

        (trained_model,train_curve, test_loss, accuracy1, accuracy5) = train.train(sparse_model, loss, optimizer, train_loader,
                                                                        test_loader, dev, args.epochs, scheduler)

        results = []

        results.append(train_curve)
        results.append(test_loss)
        results.append(accuracy1)
        results.append(accuracy5)

        with open(args.ablation_experiment + str(args.prune_perc[len(args.prune_perc)-i-1]) + str(args.model) + str(args.dataset) +str(args.seed)+ '.pkl', "wb") as fout:
            pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)

