import os
import argparse

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import scipy.sparse as sp
import numpy as np

from sklearn.metrics import mean_squared_error
from utils.utils import scipy_to_torch_sparse, genAdyacencyMatrix
from dataLoader import LandmarksDataset, ToTensor, Rescale, RandomScale, AugColor, RandomCrop, Rotate

from models.hybridDual import HybridDual
from models.UNet import DiceLoss

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
        
    if not args['test']:
        folder = os.path.join(tensorboard, args['name'])
        
        try:
            os.mkdir(folder)
        except:
            pass 
        
        writer = SummaryWriter(log_dir = folder)
    
    if args['load2'] != "None":
        weights = os.path.join(tensorboard, args['load2'])
        weights = os.path.join(weights, 'best.pt')
        print('Encoder weights loaded')
        model.load_state_dict(torch.load(weights), strict = False)
    
    if args['load'] != "None":
        weights = os.path.join(tensorboard, args['load'])
        weights = os.path.join(weights, 'best.pt')
        print('Decoder weights loaded')
        model.load_state_dict(torch.load(weights), strict = False)
        
        if args['finetune']:
            print('Decoder weights are not trainable')
            for p in model.cheb_dec.parameters():
                p.requires_grad = False
            for p in model.dec_lin.parameters():
                p.requires_grad = False          
    
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'], weight_decay=1e-6)

    train_loss_avg = []
    train_kld_loss_avg = []
    train_seg_loss_avg = []
    train_rec_loss_avg = []
    val_loss_avg = []
    val_loss_seg_avg = []

    best = 1e12

    print('Training ...')
    
    loss_seg = torch.nn.CrossEntropyLoss().to(device)
    loss_dice = DiceLoss().to(device)
    
    scheduler = StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    if args['test']:
        nepochs = 1
    else:
        nepochs = args['epochs']
    
    for epoch in range(nepochs):
        if args['finetune'] and epoch == args['finetune_epoch']:
            print('Decoder weights are now trainable')
            for p in model.cheb_dec.parameters():
                p.requires_grad = True
            for p in model.dec_lin.parameters():
                p.requires_grad = True
            
            optimizer.add_param_group({'params': model.cheb_dec.parameters()})
            optimizer.add_param_group({'params': model.dec_lin.parameters()})
            
        model.train()
        
        train_loss_avg.append(0)
        train_rec_loss_avg.append(0)
        train_seg_loss_avg.append(0)
        train_kld_loss_avg.append(0)

        num_batches = 0

        for sample_batched in train_loader:
            image, target, seg = sample_batched['image'].to(device), sample_batched['landmarks'].to(device), sample_batched['seg'].to(device)
            
            out, outseg = model(image)
                        
            # backpropagation
            optimizer.zero_grad()

            loss = F.mse_loss(out, target)
            
            train_rec_loss_avg[-1] += loss.item()
           
            a = loss_seg(outseg, seg[:,0,:,:]) 
            b = loss_dice(outseg, seg[:,0,:,:])
            
            aux = 0.1 * (a + b)
            
            train_seg_loss_avg[-1] += aux.item()
            
            loss += aux
            
            kld_loss = -0.5 * torch.mean(torch.mean(1 + model.log_var - model.mu ** 2 - model.log_var.exp(), dim=1), dim=0)
            loss += model.kld_weight * kld_loss
            
            train_loss_avg[-1] += loss.item()
            
            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            train_kld_loss_avg[-1] += model.kld_weight * kld_loss.item()
            
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        train_seg_loss_avg[-1] /= num_batches
        train_rec_loss_avg[-1] /= num_batches
        train_kld_loss_avg[-1] /= num_batches
        
        #print('Epoch [%d / %d] train average loss: %f' % (epoch+1, args['epochs'], train_loss_avg[-1]))

        num_batches = 0
        
        model.eval()
        val_loss_avg.append(0)
        val_loss_seg_avg.append(0)

        with torch.no_grad():
            for sample_batched in val_loader:
                image, target, seg = sample_batched['image'].to(device), sample_batched['landmarks'].to(device), sample_batched['seg'].to(device)

                out, outseg = model(image)

                out = out.reshape(-1, 2)
                target = target.reshape(-1, 2)

        #        # reconstruction error
                loss_rec = mean_squared_error(out.cpu().numpy(), target.cpu().numpy())
                val_loss_avg[-1] += loss_rec
                num_batches += 1
                
                loss_out = 1 - loss_dice(outseg, seg[:,0,:,:])
                val_loss_seg_avg[-1] += loss_out.item()
        #        
                loss_rec = 0
                loss_out = 0

        val_loss_avg[-1] /= num_batches
        val_loss_seg_avg[-1] /= num_batches
        
        if not args['test']:
            writer.add_scalar('Loss/train', train_loss_avg[-1], epoch)
            writer.add_scalar('MSE/validation', val_loss_avg[-1]  * 512 * 512, epoch)
            writer.add_scalar('Loss kld/train', train_kld_loss_avg[-1], epoch)
            writer.add_scalar('Loss rec/train', train_rec_loss_avg[-1], epoch)
            writer.add_scalar('Loss seg/train', train_seg_loss_avg[-1], epoch)
            writer.add_scalar('Dice/validation', val_loss_seg_avg[-1], epoch)
        
        print('Epoch [%d / %d] validation average reconstruction error: %f' % (epoch+1, args['epochs'], val_loss_avg[-1] * 512 * 512))

        if val_loss_avg[-1] < best and not args['test']:
            best = val_loss_avg[-1]
            print('Model Saved')
            torch.save(model.state_dict(), os.path.join(folder, "best.pt"))

        scheduler.step()
    
    if not args['test']:
        torch.save(model.state_dict(), os.path.join(folder, "final.pt"))
        
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)
    parser.add_argument("--batch_size", default = 4, type=int)
    parser.add_argument("--epochs", default = 2000, type=int)
    parser.add_argument("--lr", default = 1e-4, type=float)
    parser.add_argument("--gamma", default = 0.1, type=float)
    parser.add_argument("--test_batch_size", default = 1, type=int)
    parser.add_argument("--step_size", default = 1000, type=int)
    parser.add_argument("--variational_beta", default = 1e-5, type=float)
    parser.add_argument("--model_loss", default = "mse", type=str)
    parser.add_argument("--load", help="enter the folder where the weights are saved", default = "None", type=str)
    parser.add_argument("--load2", help="enter the folder where the weights are saved", default = "None", type=str)

    parser.add_argument('--finetune', dest='finetune', action='store_true')
    parser.set_defaults(finetune=False)

    parser.add_argument("--finetune_epoch", default = 500, type=int)
    parser.add_argument("--latent", default = 64, type=int)
    parser.add_argument("--polygon_order", default = 6, type=int)

    parser.add_argument('--skip_connections', dest='skip', action='store_true')
    parser.set_defaults(skip=False)
    
    parser.add_argument('--test_run', dest='test', action='store_true')
    parser.set_defaults(test=False)
    
    parser.add_argument("--model", help="choose model architecture from: Coma Hybrid", default = 'Hybrid', type=str)
    
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
    

    A = genAdyacencyMatrix()
    A = sp.csc_matrix(A).tocoo()
    
    I = np.eye(166)
    I = sp.csc_matrix(I).tocoo()
    
    D = [I.copy(), I.copy(), I.copy(), I.copy()]
    U = [I.copy(), I.copy(), I.copy(), I.copy()]
    A = [A.copy(), A.copy(), A.copy(), A.copy()]

    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to('cuda:0') for x in X] for X in (A, D, U))

    num_nodes = [166, 166, 166, 166]
    num_features = 2

    config = {}

    p = args['polygon_order']
    
    config['n_layers'] = 4
    config['num_conv_filters'] = [16, 16, 16, 16, 16]
    config['polygon_order'] = [p, p, p, p, p]
    config['kld_weight'] = args['variational_beta']
    config['z'] = args["latent"] # latent space

    model = HybridDual(config, num_features, D_t, U_t, A_t, num_nodes)
    
    trainer(train_dataset, val_dataset, model, args)