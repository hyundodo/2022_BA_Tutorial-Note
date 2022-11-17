import torch
import torch.nn as nn
import torch.optim as optim

import sys
import time
import argparse
import numpy as np

from utils import get_network, get_training_dataloader, get_test_dataloader, loss_visualize



def train(epoch):
    start = time.time()
    net.train()
    
    if args.gpu:
        net.to(DEVICE)   
    
    for batch_index, images in enumerate(train_set_loader):
        if args.gpu:
            images = images.to(DEVICE)
        if not (args.net == 'CAE' or args.net == 'CDAE'):
            images = images.view(-1, 28*28).to(DEVICE)
        
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        loss_c = nn.MSELoss()
        _, decoded = net(images)

        loss = loss_c(decoded, images)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
    
    print('Training Epoch: {epoch} \tLoss: {:0.4f}\tLR: {:0.6f}'.format(
           loss.item(),
           optimizer.param_groups[0]['lr'],
           epoch=epoch))
    
    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    return net, loss

@torch.no_grad()
def test(epoch=0):
    start = time.time()
    net.eval()
    
    conc_out = []
    conc_origin = []
    with torch.no_grad():
        for images in test_set_loader:
            if args.gpu:
                images = images.to(DEVICE)
            if not (args.net == 'CAE' or args.net == 'CDAE'):
                images = images.view(-1, 28*28).to(DEVICE)

            _, decoded = net(images)
            
            conc_out.append(decoded.to('cpu'))
            conc_origin.append(images.to('cpu'))
        
        conc_out = torch.cat(conc_out)
        conc_origin = torch.cat(conc_origin)
        
        loss_c = nn.MSELoss()
        
        val_loss = loss_c(conc_out, conc_origin)
    finish = time.time()
    print('validation time consumed: {:.2f}s'.format(epoch, finish - start))
    
    return val_loss    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-epoch', type=int, default=300, help='num epochs')
    args = parser.parse_args()
    net = get_network(args)
    
    if args.gpu:
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
        
    train_set_loader = get_training_dataloader(
        num_workers=1,
        batch_size=args.b,
        shuffle=True
    )
    
    test_set_loader = get_test_dataloader(
        num_workers=1,
        batch_size=args.b,
        shuffle=True
    )


    t_train_loss = []
    t_valid_loss = []
    for epoch in range(1, args.epoch + 1):
        
        net, train_loss = train(epoch)
        val_loss = test()
        print('{epoch} epoch Val Loss {:0.4f}'.format(
            val_loss.item(),
            epoch=epoch
        ))
        t_train_loss.append(train_loss.item())
        t_valid_loss.append(val_loss.item())

        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'./save_CAE/ae_{epoch}.pth')
    
    print('Save loss fig')
    loss_visualize(t_train_loss, t_valid_loss, net=args.net)
    