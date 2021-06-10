import torch
import torch.nn as nn
from Utils import load
from Utils import train
import pickle as pkl
import torch

def run(args):

    torch.manual_seed(args.seed)
    dev = load.device(args.gpu)

    print(args.model, args.dataset)
    input_shape, num_classes = load.dimension(args.dataset)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.train_batch_size, False, args.workers)

    model = load.model(args.model, args.dataset)(input_shape, num_classes).to(dev)
    loss = nn.CrossEntropyLoss()
    opt, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt(model.parameters(),  lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)
    (net, train_curve, test_loss, accuracy1, accuracy5) = train.train(model, loss, optimizer, train_loader,
                                                                 test_loader, dev, args.epochs, scheduler)

    results = []

    results.append(train_curve)
    results.append(test_loss)
    results.append(accuracy1)
    results.append(accuracy5)

    with open('Baseline_FC'+ str(args.model) + str(args.dataset) + str(args.seed) + '.pkl', "wb") as fout:
        pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)