import os
import torch
import torch.nn.functional as F

from models.UNet import UNet, DiceLoss

import numpy as np
import scipy.sparse as sp

from torch.optim.lr_scheduler import StepLR
from dataLoader import LandmarksDataset, ToTensor, Rescale, RandomScale, AugColor, RandomCrop, Rotate
from torchvision import transforms

from sklearn.metrics import mean_squared_error

from torch.utils.tensorboard import SummaryWriter

def trainer(train_dataset, val_dataset, model, args):   
    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)

    train_kwargs = {'batch_size':args['batch_size']}
    test_kwargs = {'batch_size':args['test_batch_size']}

    cuda_kwargs = {'num_workers': 0,
                    'pin_memory': False,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    
    tensorboard = "Training"
    folder = os.path.join(tensorboard, args['name'])
    
    try:
        os.mkdir(folder)
    except:
        pass 
    
    writer = SummaryWriter(log_dir = folder)
    
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'], weight_decay=1e-6)

    train_loss_avg = []
    val_loss_avg = []

    best = 0

    print('Training ...')
    
    loss_ce = torch.nn.CrossEntropyLoss().to(device)
    loss_dice = DiceLoss().to(device)
    
    scheduler = StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    for epoch in range(args['epochs']):
        model.train()
        
        train_loss_avg.append(0)
        num_batches = 0

        for sample_batched in train_loader:
            image, seg = sample_batched['image'].to(device), sample_batched['seg'].to(device)
            
            outseg = model(image)
                
            # backpropagation
            optimizer.zero_grad()

            loss = loss_ce(outseg, seg[:,0,:,:]) + loss_dice(outseg, seg[:,0,:,:])
            train_loss_avg[-1] += loss.item()
            
            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            num_batches += 1

        train_loss_avg[-1] /= num_batches
            
        #print('Epoch [%d / %d] train average loss: %f' % (epoch+1, args['epochs'], train_loss_avg[-1]))

        num_batches = 0
        
        model.eval()
        val_loss_avg.append(0)

        with torch.no_grad():
            for sample_batched in val_loader:
                image, seg = sample_batched['image'].to(device), sample_batched['seg'].to(device)

                outseg = model(image)

                # reconstruction error
                loss_rec = 1 - loss_dice(outseg, seg[:,0,:,:])
                val_loss_avg[-1] += loss_rec.item()
                num_batches += 1
               
                loss_rec = 0

        val_loss_avg[-1] /= num_batches
        
        writer.add_scalar('Loss/train', train_loss_avg[-1], epoch)
        writer.add_scalar('Dice/validation', val_loss_avg[-1], epoch)
        
        print('Epoch [%d / %d] validation average dice coefficient: %f' % (epoch+1, args['epochs'], val_loss_avg[-1]))

        if val_loss_avg[-1] > best:
            best = val_loss_avg[-1]
            print('Model Saved')
            torch.save(model.state_dict(), os.path.join(folder, "best.pt"))

        scheduler.step()
        
    torch.save(model.state_dict(), os.path.join(folder, "final.pt"))


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)
    parser.add_argument("--batch_size", default = 4, type=int)
    parser.add_argument("--epochs", default = 2000, type=int)
    parser.add_argument("--lr", default = 1e-4, type=float)
    parser.add_argument("--gamma", default = 0.95, type=float)
    parser.add_argument("--test_batch_size", default = 1, type=int)
    parser.add_argument("--step_size", default = 1000, type=int)
 
    args = parser.parse_args()
    
    args = vars(args)
    
    train_path = "Dataset/Train"
    test_path = "Dataset/Test"
    val_path = "Dataset/Val" 

    img_path = os.path.join(train_path, 'Images')
    label_path = os.path.join(train_path, 'landmarks')
    
    train_dataset = LandmarksDataset(img_path=img_path,
                                     label_path=label_path,
                                     transform = transforms.Compose([
                                                 RandomScale(512, 6),
                                                 RandomCrop(512),
                                                 AugColor(0.15),
                                                 Rotate(3),
                                                 ToTensor()])
                                     )


    img_path = os.path.join(val_path, 'Images')
    label_path = os.path.join(val_path, 'landmarks')
    val_dataset = LandmarksDataset(img_path=img_path,
                                     label_path=label_path,
                                     transform = transforms.Compose([
                                                 Rescale(512),
                                                 ToTensor()])
                                     )
    

    UNet = UNet()
    
    trainer(train_dataset, val_dataset, UNet, args)