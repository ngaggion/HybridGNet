import os 

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import numpy as np

from models.vae import VariationalAutoencoderMixto, VariationalAutoencoderFC, vae_loss

from dataLoader import LandmarksDataset, ToTensor, Rescale, RandomScale, AugColor, RandomCrop, Rotate

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
    if not args['test']:
        folder = os.path.join(tensorboard, args['name'])
        
        try:
            os.mkdir(folder)
        except:
            pass 
        
        writer = SummaryWriter(log_dir = folder)   
    
    if args['load'] != "None":
        weights = os.path.join(tensorboard, args['load'])
        weights = os.path.join(weights, 'final.pt')
        
        model.load_state_dict(torch.load(weights), strict = False)
        
    if args['load2'] != "None":
        weights = os.path.join(tensorboard, args['load2'])
        weights = os.path.join(weights, 'final.pt')
        
        model.load_state_dict(torch.load(weights), strict = False)

    if args['load'] != "None" and args['finetune']:
        print('Decoder weights are not trainable')
        for p in model.decoder.parameters():
            p.requires_grad = False
        
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'], weight_decay=1e-5)

    train_loss_avg = []
    val_loss_avg = []

    best = 1e12

    print('Training ...')

    scheduler = StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    if args['test']:
        nepochs = 1
    else:
        nepochs = args['epochs']
    
    for epoch in range(nepochs):
        if args['finetune'] and epoch == 100:
            print('Decoder weights are now trainable')
            for p in model.decoder.parameters():
                p.requires_grad = True
            
            optimizer.add_param_group({'params': model.decoder.parameters()})
            
        model.train()
        train_loss_avg.append(0)
        num_batches = 0

        for sample_batched in train_loader:
            image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
            target = target.reshape(image.shape[0], -1)
            # vae reconstruction
            if args['model'] == 'VaeFC':
                image_batch_recon, latent_mu, latent_logvar = model(target)
            else:
                image_batch_recon, latent_mu, latent_logvar = model(image)

            # reconstruction error
            if args['model_loss'] == 'mse':
                loss = vae_loss(image_batch_recon, target, latent_mu, latent_logvar, args['variational_beta'])
            else:
                raise Exception('Loss no implementada')
                
            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        
        num_batches = 0
        
        model.eval()
        val_loss_avg.append(0)

        with torch.no_grad():
            for sample_batched in val_loader:
                image_batch, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
                target = target.reshape(1, -1)
                
                if args['model'] == 'VaeFC':
                    image_batch_recon, latent_mu, latent_logvar = model(target)
                else:
                    image_batch_recon, latent_mu, latent_logvar = model(image_batch)
                
        #        # reconstruction error
                loss_rec = mean_squared_error(image_batch_recon.cpu().numpy(), target.cpu().numpy())
                val_loss_avg[-1] += loss_rec
                num_batches += 1
        #        
                loss_rec = 0

        val_loss_avg[-1] /= num_batches
        
        if not args['test']:
            writer.add_scalar('Loss/train', train_loss_avg[-1], epoch)
            writer.add_scalar('MSE/validation', val_loss_avg[-1]  * 512 * 512, epoch)
        
        print('Epoch [%d / %d] validation average reconstruction error: %f' % (epoch+1, args['epochs'], val_loss_avg[-1] * 512 * 512))

        if val_loss_avg[-1] < best and not args['test']:
            best = val_loss_avg[-1]
            print('Model Saved')
            torch.save(model.state_dict(), os.path.join(folder, "best.pt"))

        scheduler.step()
    
    if not args['test']:
        torch.save(model.state_dict(), os.path.join(folder, "final.pt"))
        
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)
    parser.add_argument("--batch_size", default = 8, type=int)
    parser.add_argument("--epochs", default = 3000, type=int)
    parser.add_argument("--lr", default = 1e-4, type=float)
    parser.add_argument("--gamma", default = 0.95, type=float)
    parser.add_argument("--test_batch_size", default = 1, type=int)
    parser.add_argument("--step_size", default = 10, type=int)
    parser.add_argument("--variational_beta", default = 1e-5, type=float)
    parser.add_argument("--latent", default = 128, type=int)
    parser.add_argument("--model_loss", default = "mse", type=str)
    parser.add_argument("--kc", help="kernel numbers kc * 2, 4, 8, 16", default = 4, type=int)
    
    parser.add_argument("--load", help="enter the folder where the decoder weights are saved", default = "None", type=str)
    parser.add_argument("--load2", help="enter the folder where the encoder weights are saved", default = "None", type=str)
    
    parser.add_argument('--finetune', dest='finetune', action='store_true')
    parser.set_defaults(finetune=False)
    
    parser.add_argument("--model", help="choose model architecture from: VaeFC VaeMixto", default = 'VaeFC', type=str)
    
    parser.add_argument('--test_run', dest='test', action='store_true')
    parser.set_defaults(test=False)
    
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
    
    if args['model'] == 'VaeFC':
        vae = VariationalAutoencoderFC(inputsize = 332, latents = args['latent'])
    else:
        vae = VariationalAutoencoderMixto(inputsize = 332, latents = args['latent'], c= args['kc'])

    trainer(train_dataset, val_dataset, vae, args)
