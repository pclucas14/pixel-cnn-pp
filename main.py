import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from utils import * 
from model import * 
import gzip
import cPickle
from PIL import Image

nr_logistic_mix = 10
batch_size = 128
sample_batch_size = 25
MNIST = False
obs = (1, 28, 28) if MNIST else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

if MNIST : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, 
                    transform=ds_transforms), batch_size=128, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, 
                    transform=ds_transforms), batch_size=128, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, nr_logistic_mix)

else : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, 
        download=True, transform=ds_transforms), batch_size=batch_size, shuffle=True)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=False, 
                    transform=ds_transforms), batch_size=128, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, nr_logistic_mix)


model = PixelCNN(nr_resnet=3, nr_filters=70, input_channels=input_channels, 
                    nr_logistic_mix=nr_logistic_mix)
model = model.cuda()

# optimizer = optim.Adamax(model.parameters(), lr=4e-4)
optimizer = optim.Adam(model.parameters())

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):
            data_v = Variable(data, volatile=True)
            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
for epoch in range(100):
    model.train(True)
    torch.cuda.synchronize()
    train_loss = 0.
    model.train()
    for batch_idx, (input,_) in enumerate(train_loader):
        if batch_idx > 100 : break
        input = input.cuda(async=True)
        input = Variable(input)
        output = model(input)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        if batch_idx % 10 == 9 : 
            print('loss : %s' % (train_loss / (10*np.prod((batch_size,) + obs))))
            train_loss = 0.
    
    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(train_loader):
        if batch_idx > 20 : break
        input = input.cuda(async=True)
        input = Variable(input)
        output = model(input)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        test_loss += loss.data[0]
    
    print('test loss : %s' % (test_loss / (batch_idx*np.prod((batch_size,) + obs))))
    
    print('sampling...')
    sample_t = sample(model)
    sample_t = rescaling_inv(sample_t)
    ds = 'mnist' if MNIST else 'cifar'
    utils.save_image(sample_t,'images/{}_{}.png'.format(ds, epoch), nrow=5, padding=0)
    
    if epoch % 10 == 9: 
        torch.save(model.state_dict(), 'models/{}_{}.pth'.format(ds, epoch))
