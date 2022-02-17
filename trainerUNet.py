import os
import torch
import torch.nn.functional as F
import argparse

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import numpy as np

from utils.dataLoader import LandmarksDataset, ToTensorSeg, ToTensorSegLH, Rescale, RandomScale, AugColor, Rotate

from models.unet import UNet, DiceLoss
from torch.nn import CrossEntropyLoss

from medpy.metric.binary import dc

def evalImageMetrics(output, target):
    dcp = dc(output == 1, target == 1)
    dcc = dc(output == 2, target == 2)
    #dccla = dc(output == 3, target == 3)
    
    return dcp, dcc#, dccla

def trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['val_batch_size'], num_workers = 0)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    train_loss_avg = []
    val_loss_avg = []
    val_dicelungs_avg = []
    val_diceheart_avg = []
    val_dicecla_avg = []

    tensorboard = "Training"
        
    folder = os.path.join(tensorboard, config['name'])

    try:
        os.mkdir(folder)
    except:
        pass 

    writer = SummaryWriter(log_dir = folder)  

    best = 0
    suffix = ".pt"
    
    print('Training ...')
    
    dice_loss = DiceLoss().to(device)
    ce_loss = CrossEntropyLoss().to(device)
    
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])
    
    for epoch in range(config['epochs']):
        model.train()

        train_loss_avg.append(0)
        num_batches = 0
        
        for sample_batched in train_loader:
            image, target = sample_batched['image'].to(device), sample_batched['seg'].to(device)
            
            out = model(image)

            # backpropagation
            optimizer.zero_grad()
            
            loss = dice_loss(out, target) + ce_loss(out, target)
            train_loss_avg[-1] += loss.item()

            loss.backward()
            optimizer.step()

            num_batches += 1
        
        print('')

        train_loss_avg[-1] /= num_batches
        num_batches = 0

        model.eval()
        val_loss_avg.append(0)
        val_dicelungs_avg.append(0)
        val_diceheart_avg.append(0)
        val_dicecla_avg.append(0)
        
        with torch.no_grad():
            for sample_batched in val_loader:                
                image, target = sample_batched['image'].to(device), sample_batched['seg'].cpu().numpy()

                out = model(image)
                seg = torch.argmax(out[0,:,:,:], axis = 0).cpu().numpy()
                dcl, dch = evalImageMetrics(seg, target[0,:,:])
                
                val_dicelungs_avg[-1] += dcl
                val_diceheart_avg[-1] += dch
                #val_dicecla_avg[-1] += dccla
                val_loss_avg[-1] += (dcl + dch) / 2
                
                num_batches += 1   
                loss_rec = 0
        print('')

        val_loss_avg[-1] /= num_batches
        val_dicelungs_avg[-1] /= num_batches
        val_diceheart_avg[-1] /= num_batches
        #val_dicecla_avg[-1] /= num_batches
        
        print('Epoch [%d / %d] validation Dice: %f' % (epoch+1, config['epochs'], val_loss_avg[-1]))

        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Validation/Dice', val_loss_avg[-1], epoch)
        writer.add_scalar('Validation/Dice Lungs', val_dicelungs_avg[-1], epoch)
        writer.add_scalar('Validation/Dice Heart', val_diceheart_avg[-1], epoch)
        #writer.add_scalar('Validation/Dice Cla', val_dicecla_avg[-1], epoch)
        
        if epoch % 500 == 0:
            suffix = "_%s.pt" % epoch
            best = 0
            
        if val_loss_avg[-1] > best:
            best = val_loss_avg[-1]
            print('Model Saved Dice')
            out = "bestDice.pt"
            torch.save(model.state_dict(), os.path.join(folder, out.replace('.pt',suffix)))

        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(folder, "final.pt"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)    
    parser.add_argument("--load", help="enter the folder where the weights are saved", default = "None", type=str)
    parser.add_argument("--inputsize", default = 1024, type=int)
    parser.add_argument("--epochs", default = 1500, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 3000, type = int)
    parser.add_argument("--gamma", default = 0.1, type = float)
    
    parser.add_argument('--extended', dest='extended', action='store_true')
    parser.set_defaults(extended=False)
    
    config = parser.parse_args()
    config = vars(config)

    inputSize = config['inputsize']

    if config['extended']:
        train_path = "Datasets/Extended/Train"
        val_path = "Datasets/Extended/Val" 
    else:
        train_path = "Datasets/JSRT/Train"
        val_path = "Datasets/JSRT/Val" 

    img_path = os.path.join(train_path, 'Images')
    label_path = os.path.join(train_path, 'landmarks')

    train_dataset = LandmarksDataset(img_path=img_path,
                                     label_path=label_path,
                                     transform = transforms.Compose([
                                                 RandomScale(),
                                                 Rotate(3),
                                                 AugColor(0.40),
                                                 ToTensorSegLH()])
                                     )

    img_path = os.path.join(val_path, 'Images')
    label_path = os.path.join(val_path, 'landmarks')
    val_dataset = LandmarksDataset(img_path=img_path,
                                     label_path=label_path,
                                     transform = transforms.Compose([
                                                 Rescale(inputSize),
                                                 ToTensorSegLH()])
                                     )

    config['latents'] = 64
    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5
    
    model = UNet(n_classes = 3)
    
    trainer(train_dataset, val_dataset, model, config)