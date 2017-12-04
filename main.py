import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import * 
from model import * 
import gzip
import cPickle
from PIL import Image

nr_logistic_mix = 10
batch_size = 32
sample_batch_size = 9 
MNIST = True
obs = (1, 28, 28) if MNIST else (3, 32, 32)
input_channels = obs[0]

if MNIST : 
    with gzip.open('data/mnist.pkl.gz', 'rb') as f : 
        train_set, valid_set, test_set = cPickle.load(f)

    train_set = ((train_set[0] - 0.5) * 2).reshape((train_set[0].shape[0], 1, 28, 28))
    valid_set = ((valid_set[0] - 0.5) * 2).reshape((valid_set[0].shape[0], 1, 28, 28))

    kwargs = {'num_workers':1, 'pin_memory':True}
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(valid_set,
        batch_size=batch_size, shuffle=True, **kwargs)

    loss_op = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_logistic_mix_1d(x, nr_logistic_mix)

else : 
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    loss_op = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_logistic_mix(x, nr_logistic_mix)


model = PixelCNN(nr_resnet=3, nr_filters=60, input_channels=input_channels, nr_logistic_mix=nr_logistic_mix)
model = model.cuda()

optimizer = optim.Adamax(model.parameters(), lr=4e-4)

def sample(model):
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for yi in range(obs[1]):
        for xi in range(obs[2]):
            data_v = Variable(data)
            out   = model(data_v, sample=True)
            out_s = sample_op(out)
            data[:, :, yi, xi] = out_s.data[:, :, yi, xi]

    return data

def show_tensor(sample, epoch=0):
    grid = np.concatenate([sample[i] for i in range(sample.shape[0])], axis=1)
    grid = (grid * 0.5) + 0.5
    grid *= 255.
    grid = grid.transpose(1, 2, 0)
    grid = grid.astype('uint8')
    mode = 'mnist' if MNIST else 'cifar'
    if MNIST : grid = grid.squeeze()
    Image.fromarray(grid).save('images/{}_{}.png'.format(mode, epoch))


print('starting training')
for epoch in range(100):
    for batch_idx, data in enumerate(train_loader):
        data = data if MNIST else data[0]
        data = data.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        output = model(data)
        # loss = ((output - data) ** 2).sum() * .5 #discretized_mix_logistic_loss(data, output)
        loss = loss_op(data, output)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 9 : 
            print('loss : %s' % (loss.data[0] / np.prod((batch_size,) + obs)))
    print('sampling...')
    sample_t = sample(model).cpu().numpy()
    show_tensor(sample_t[:9], epoch=epoch)
    # show_tensor(data.data.cpu()[:9].numpy(), epoch=epoch+1000)
        
        
