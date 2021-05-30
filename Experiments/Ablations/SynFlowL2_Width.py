import torch
import torch.nn as nn
from Utils import load
from Utils import train
from Prune import synflowl2_utils
import copy
import pickle as pkl
from Prune import Utils

def run(args):
    torch.manual_seed(args.seed)
    dev = load.device(args.gpu)

    input_shape, num_classes = load.dimension(args.dataset)
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_size*num_classes)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.train_batch_size, False, args.workers)

    model = load.model(args.model, args.dataset)(input_shape, num_classes).to(dev)
    fraction = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    prune_perc = args.prune_perc

    sparse_model = copy.deepcopy(model)
    loss = nn.CrossEntropyLoss()
    opt, opt_kwargs = load.optimizer(args.optimizer)
    print(prune_perc)
    weight_masks, bias_masks = synflowl2_utils.synflowl2_prune_masks(sparse_model, prune_perc, prune_loader,
                                                                 args.synflow_iterations, dev)
    del sparse_model
    for i in range(len(fraction)):
        print(args.experiment, fraction[i], str(args.model), str(args.dataset))
        wm = copy.deepcopy(weight_masks)
        bm = copy.deepcopy(bias_masks)
        net = copy.deepcopy(model)
        wm, bm = Utils.layerwise_randomshuffle(net,wm,fraction[i],dev)
        del net
        sparse_model = copy.deepcopy(model)
        sparse_model.set_masks(wm, bm)
        sparse_model.to(dev)
        optimizer = opt(sparse_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)
        (trained_model, train_curve, test_loss, accuracy1, accuracy5) = train.train(sparse_model, loss, optimizer, train_loader,
                                                                     test_loader, dev, args.epochs, scheduler)
        del sparse_model
        results = []

        results.append(train_curve)
        results.append(test_loss)
        results.append(accuracy1)
        results.append(accuracy5)

        with open(args.ablation_experiment + str(fraction[i]) + str(args.model) + str(args.dataset) + str(args.seed) + '.pkl',"wb") as fout:
            pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)