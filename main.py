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
sample_batch_size = 144 
MNIST = True
obs = (1, 28, 28) if MNIST else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : x #(x - .5) * 2.
rescaling_inv = lambda x : x #.5 * x  + .5

if MNIST : 
    with gzip.open('data/mnist.pkl.gz', 'rb') as f : 
        train_set, valid_set, test_set = cPickle.load(f)

    train_set = ((train_set[0] - 0.5) * 2).reshape((train_set[0].shape[0], 1, 28, 28))
    valid_set = ((valid_set[0] - 0.5) * 2).reshape((valid_set[0].shape[0], 1, 28, 28))

    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
    
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, 
                    transform=transforms.ToTensor()), batch_size=128, shuffle=True, **kwargs)

    # train_loader = torch.utils.data.DataLoader(train_set,
    #     batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    # test_loader = torch.utils.data.DataLoader(valid_set,
    #     batch_size=batch_size, shuffle=True, **kwargs)

    loss_op = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, nr_logistic_mix)

else : 
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=
            transforms.Compose([transforms.ToTensor(), rescaling]))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    loss_op = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, nr_logistic_mix)


model = PixelCNN(nr_resnet=3, nr_filters=60, input_channels=input_channels, 
                    nr_logistic_mix=nr_logistic_mix)

'''
fm = 64
model = nn.Sequential(
    MaskedConv2d('A', 1,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    nn.Conv2d(fm, 256, 1))
print model
'''

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
            probs = F.softmax(out[:, :, i, j]).data
            data[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
            # out_sample = sample_op(out)
            # data[:, :, i, j] = out_sample[:, :, i, j]

    return data

def show_tensor(sample, epoch=0):
    grid = np.concatenate([sample[i] for i in range(sample.shape[0])], axis=1)
    # grid = (grid * 0.5) + 0.5
    grid *= 255.
    grid = grid.transpose(1, 2, 0)
    grid = grid.astype('uint8')
    mode = 'mnist' if MNIST else 'cifar'
    if MNIST : grid = grid.squeeze()
    Image.fromarray(grid).save('images/{}_{}.png'.format(mode, epoch))


print('starting training')
for epoch in range(100):
    model.train(True)
    torch.cuda.synchronize()
    for batch_idx, (input,_) in enumerate(train_loader):
        # data = data if MNIST else data[0]
        input = input.cuda(async=True)
        input = Variable(input)
        target = Variable((input.data[:, 0] * 255).long())
        output = model(input)
        # loss = loss_op(data, output)
        loss = F.cross_entropy(output, target, reduce=False).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 9 : 
            print('loss : %s' % (loss.data[0] / np.prod((batch_size,) + obs)))
    
    torch.cuda.synchronize()
    print('sampling...')
    sample_t = sample(model)
    sample_t = rescaling_inv(sample_t)
    utils.save_image(sample_t,'images/sample_{:02d}.png'.format(epoch), nrow=12, padding=0)
    print('max : %s, min : %s' % (sample_t.max(), sample_t.min()))
    # show_tensor(sample_t[:9].cpu().numpy(), epoch=epoch)
    # show_tensor(data.data.cpu()[:9].numpy(), epoch=epoch+1000)
        
        
