import torch
import torch.nn as nn
from Utils import load
from Utils import train
from Prune import synflow_utils
import copy
import pickle as pkl

def run(args):
    torch.manual_seed(args.seed)
    dev = load.device(args.gpu)

    input_shape, num_classes = load.dimension(args.dataset)
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_size*num_classes)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.train_batch_size, False, args.workers)

    model = load.model(args.model, args.dataset)(input_shape, num_classes).to(dev)

    for i in range(len(args.prune_perc)):
        print(args.experiment, str(args.prune_perc[len(args.prune_perc) - i - 1]), str(args.model), str(args.dataset))

        prune_perc = args.prune_perc[len(args.prune_perc)-i-1]
        sparse_model = copy.deepcopy(model)
        loss = nn.CrossEntropyLoss()
        opt, opt_kwargs = load.optimizer(args.optimizer)

        weight_masks, bias_masks = synflow_utils.synflow_prune_masks(sparse_model, prune_perc, prune_loader, args.synflow_iterations, dev)
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

        with open(args.experiment + str(args.prune_perc[len(args.prune_perc)-i-1]) + str(args.model) + str(args.dataset) +str(args.seed)+ '.pkl', "wb") as fout:
            pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)