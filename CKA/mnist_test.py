from mnist import Net, test
import torch
from torchvision import datasets, transforms
import numpy as np

use_cuda=True
device = torch.device("cuda" if use_cuda else "cpu")

test_kwargs = {'batch_size': 1000}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}

    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

model=Net().to(device)

weights=torch.load("mnist_cnn.pt")

model.load_state_dict(weights)


test_data = datasets.CIFAR10(
    root='../data',
    train=False,
    download=True,
    transform=transform
)


test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

# dataset2 = datasets.MNIST('../data', train=False,
#                        transform=transform)

# test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

feature_maps_one_epoch=test(model, device, test_loader)

np.save( 'fms_oneepoch.npy', feature_maps_one_epoch)
