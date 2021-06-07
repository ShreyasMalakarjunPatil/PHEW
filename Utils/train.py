import torch
import pickle as pkl
import copy
from Prune import mag_utils
from Prune import Utils

def test(network, loss, dataloader, dev):
    network.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(dev)
            target = target.to(dev)
            output = network(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1,1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    avg_loss = total / len(dataloader.dataset)
    acc1 = 100.0 * correct1 / len(dataloader.dataset)
    acc5 = 100.0 * correct5 / len(dataloader.dataset)

    print('Top 1 Accuracy =', acc1)
    print('Top 5 Accuracy =', acc5)
    print('Average Loss =', avg_loss)

    return avg_loss, acc1, acc5

def train(network, loss, optimizer, train_loader, test_loader, dev, epochs, scheduler):

    train_curve = []
    accuracy1 = []
    accuracy5 = []
    test_loss = []
    acc_max = 0.0
    for epoch in range(epochs):
        network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(dev)
            target = target.to(dev)
            optimizer.zero_grad()
            output = network(data)
            batch_loss = loss(output, target)
            train_loss += batch_loss.item() * data.size(0)
            batch_loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), batch_loss.item()))

        train_curve.append(train_loss/len(train_loader.dataset))
        avg_loss, acc1, acc5 = test(network, loss, test_loader, dev)
        if acc1>acc_max:
            net = copy.deepcopy(network)
            acc_max = acc1
        accuracy1.append(acc1)
        accuracy5.append(acc5)
        test_loss.append(avg_loss)

        scheduler.step()

    return net, train_curve, test_loss, accuracy1, accuracy5

def train_mag(network, loss, optimizer, train_loader, test_loader, dev, epochs, scheduler):

    train_curve = []
    accuracy1 = []
    accuracy5 = []
    test_loss = []

    for epoch in range(epochs):
        network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(dev)
            target = target.to(dev)
            optimizer.zero_grad()
            output = network(data)
            batch_loss = loss(output, target)
            train_loss += batch_loss.item() * data.size(0)
            batch_loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), batch_loss.item()))

        train_curve.append(train_loss/len(train_loader.dataset))
        avg_loss, acc1, acc5 = test(network, loss, test_loader, dev)
        if acc1>acc_max:
            net = copy.deepcopy(network)
            acc_max = acc1
        accuracy1.append(acc1)
        accuracy5.append(acc5)
        test_loss.append(avg_loss)

        scheduler.step()

    return net

def train_mag2(network, loss, optimizer, train_loader, test_loader, dev, epochs, prune_perc):

    train_curve = []
    accuracy1 = []
    accuracy5 = []
    test_loss = []

    for epoch in range(epochs):

        train_loss = 0

        net = copy.deepcopy(network)
        e = epoch + 1.0
        p = prune_perc - prune_perc * ( 1.0 - e/epochs )**3
        print(p)
        if epoch > 0:
            lkj = 0
            for ppp in net.parameters():
                if len(ppp.data.size()) != 1:
                    ppp.data = ppp.data * weight_masks[lkj]
                    lkj += 1
        weight_masks, bias_masks = mag_utils.mag_prune_masks(net, p, dev)
        network.set_masks(weight_masks, bias_masks)
        network.to(dev)
        network.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(dev)
            target = target.to(dev)
            optimizer.zero_grad()
            output = network(data)
            batch_loss = loss(output, target)
            train_loss += batch_loss.item() * data.size(0)
            batch_loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), batch_loss.item()))

        train_curve.append(train_loss/len(train_loader.dataset))
        avg_loss, acc1, acc5 = test(network, loss, test_loader, dev)

        accuracy1.append(acc1)
        accuracy5.append(acc5)
        test_loss.append(avg_loss)
    Utils.ratio(net, weight_masks)
    return network,weight_masks, bias_masks


