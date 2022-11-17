import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

def get_network(args):
    if args.net == 'AE':
        from AutoEncoders import Autoencoder
        net = Autoencoder()
    elif args.net == 'DAE':
        from AutoEncoders import Denoising_Autoencoder
        net = Denoising_Autoencoder()
    elif args.net == 'CAE':
        from Conv_AutoEncoders import Conv_Autoencoder
        net = Conv_Autoencoder()
    elif args.net == 'CDAE':
        from Conv_AutoEncoders import Conv_Denoising_Autoencoder
        net = Conv_Denoising_Autoencoder()
    else:
        pass
    
    return net

def get_training_dataloader(batch_size=16, num_workers=1, shuffle=True):
    train_set = torchvision.datasets.MNIST(
        root = './data/MNIST',
        train = True,
        download = True,
        transform = transforms.ToTensor()
    )

    train_five = []
    #train_seven = []
    print('Get Five image from MNIST')
    for i in tqdm(range(len(train_set))):
        if train_set[i][1] == 5:
            train_five.append(train_set[i][0])
        #elif train_set[i][1] == 7:
        #    train_seven.append(train_set[i][0])
    
    train_five = torch.stack(train_five)    

    train_set_loader = DataLoader(
        dataset     = train_five,
        batch_size  = 128,
        shuffle     = True,
    )
    return train_set_loader

def get_test_dataloader(batch_size=16, num_workers=1, shuffle=True):
    test_set = torchvision.datasets.MNIST(
        root = './data/MNIST',
        train = False,
        download = True,
        transform = transforms.ToTensor()
    )

    test_five = []
    #train_seven = []
    print('Get Five image from MNIST')
    for i in tqdm(range(len(test_set))):
        if test_set[i][1] == 5:
            test_five.append(test_set[i][0])
        #elif train_set[i][1] == 7:
        #    train_seven.append(train_set[i][0])
    
    test_five = test_five[:700]
    
    test_five = torch.stack(test_five)    

    test_set_loader = DataLoader(
        dataset     = test_five,
        batch_size  = 128,
        shuffle     = True,
    )
    return test_set_loader

def loss_visualize(train_loss, val_loss, net):
    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./savefig_{net}')