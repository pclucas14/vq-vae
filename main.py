import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from pydoc import locate
from os.path import join
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision import datasets, transforms, utils

from vqvae  import * 
from utils  import * 

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
parser.add_argument('--debug', action='store_true')

# VQ-VAE args
parser.add_argument('--hH', type=int, default=8)
parser.add_argument('--n_codebooks', type=int, default=1,       help='number of codebooks to use')
parser.add_argument('--n_res_blocks', type=int, default=2,      help='number of residual blocks')
parser.add_argument('--n_res_channels', type=int, default=32,   help='number of channels for residual blocks')
parser.add_argument('--n_channels', type=int, default=128,      help='number of channels for non residual convs')
parser.add_argument('--n_embeds', type=int, default=512,        help='number of embeddings in a codebook')
parser.add_argument('--embed_dim', type=int, default=64,        help='size of an embedding in the codebook')

parser.add_argument('--ema', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--commit_coef', type=float, default=0.5)
parser.add_argument('--loss_fn', type=str, default='logistic', choices=['logistic', 'gaussian'])

args = parser.parse_args()
args.input_size = (3, 32, 32)
args.downsample = args.input_size[-1] // args.hH

# get loss functions
loss_fn = locate('utils.%s_ll' % args.loss_fn)
import pdb; pdb.set_trace()

# create model and ship to GPU
test = load_model_from_file('runs/LR0.001_BS64_D64_K512_hH8_NC1_coef0.5')
model = VQVAE(args).cuda() 
print(model)
print("number of parameters:", sum([np.prod(p.size()) for p in model.parameters()]))

# reproducibility is da best
set_seed(0)

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

# create datasets /dataloaders
scale_inv = lambda x : x + 0.5
ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x : x - 0.5])
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cl-pytorch/data', train=True, 
    download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10('../cl-pytorch/data', train=False, 
    download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

# spawn writer
model_name = 'LR{}_BS{}_D{}_K{}_hH{}_NC{}_coef{}_Loss{}'.format(args.lr, args.batch_size, args.embed_dim, args.n_embeds,
                                                                args.hH, args.n_codebooks, args.commit_coef, args.loss_fn)
model_name = 'test' if args.debug else model_name
log_dir    = join('runs', model_name)
sample_dir = join(log_dir, 'samples')
writer     = SummaryWriter(log_dir=log_dir)

print_and_save_args(args, log_dir)
print('logging into %s' % log_dir)
maybe_create_dir(sample_dir)
best_test = float('inf')

# useful to track
KL = args.hH * args.hH * args.n_codebooks * np.log(args.n_embeds)
N  = np.prod(args.input_size)

print('starting training')
for epoch in range(args.n_epochs):
    model.train()
    train_log = reset_log()

    for batch_idx, (input,_) in enumerate(train_loader):
        input = input.cuda()
        x, latent_loss = model(input)

        log_pxz = loss_fn(x, model.dec_log_stdv, input).mean()
        loss = -1 * (log_pxz / N) + args.commit_coef * latent_loss

        elbo = (KL - log_pxz) / N
        bpd  = (elbo + (np.log(256) if 'gaussian' in args.loss_fn else 0)) / np.log(2.)
     
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_log['kl']         += [KL]
        train_log['bpd']        += [bpd.item()]
        train_log['elbo']       += [elbo.item()]
        train_log['commit']     += [latent_loss.item()]
        train_log['log p(x|z)'] += [log_pxz.item()]
        

    for key, value in train_log.items():
        print_and_log_scalar(writer, 'train/%s' % key, value, epoch)
    print()
    
    model.eval()
    test_log = reset_log()

    with torch.no_grad():
        for batch_idx, (input,_) in enumerate(test_loader):
            input = input.cuda()
            x, latent_loss = model(input)

            log_pxz = loss_fn(x, model.dec_log_stdv, input).mean()
           
            elbo = (KL - log_pxz) / N
            bpd  = (elbo + (np.log(256) if 'gaussian' in args.loss_fn else 0)) / np.log(2.)
            
            test_log['kl']         += [KL]
            test_log['bpd']        += [bpd.item()]
            test_log['elbo']       += [elbo.item()]
            test_log['commit']     += [latent_loss.item()]
            test_log['log p(x|z)'] += [log_pxz.item()]
            
        # save reconstructions
        out = torch.stack((x, input))               # 2, bs, 3, 32, 32
        out = out.transpose(1,0).contiguous()       # bs, 2, 3, 32, 32
        out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))
        
        save_image(scale_inv(out), join(sample_dir, 'test_recon_{}.png'.format(epoch)), nrow=12)

    for key, value in test_log.items():
        print_and_log_scalar(writer, 'test/%s' % key, value, epoch)
    print()

    current_test = sum(test_log['bpd']) / batch_idx
    if current_test < best_test:
        best_test = current_test
        print('saving best model')
        torch.save(model.state_dict(), join(log_dir, 'best_model.pth'))

