from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5,1)  #input:28*28x1channel kernal:5*5x1channel x20 output:24*24x20channel -> pooling 12*12x20channel
        self.conv2 = nn.Conv2d(20, 50, 5,1) #input:12*12*20channel kernal:5*5*20channel x50 output:8*8*50channel  -> pooling 4*4xchannel
        self.fc1 = nn.Linear(4*4*50, 500) # 4*4x50channel
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):  #relation of upper 4 layer
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)  #to one dim -> fnn
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)  #output as probability  #why write here? aviod bp
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)   #gpu or cpu
        optimizer.zero_grad()   #grad to 0
        output = model(data)   #forward       
        #print(output[0,1])  
        loss = F.nll_loss(output, target)  #count loss(cost)
        loss.backward()         #backup propagation   #auto gradient
        optimizer.step()        #adjust & update weight (cnn : kermal map)
        
        if batch_idx % args.log_interval == 10000:   #output status
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        print('\npred target: ')
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)     # test only forward
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() #right num
            
            tp = target.view_as(pred)  
            match = torch.cat([pred,tp],1)
            
            #print()
            i=0
            for m in match:
                if m[0].item()==m[1].item():
                    print(m[0].item(),'   ',m[1].item(),'  true')
                else:
                    print(m[0].item(),'   ',m[1].item())

    test_loss /= len(test_loader.dataset)  #avg loss

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('----------------')

def load_dataset():
    data_path = '/workspace/python_file/mnist_byme5/'

    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        #transform=torchvision.transforms.ToTensor()
        transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor()])  
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )

    """
    for batch_idx, (data, target) in enumerate(test_loader):
        a = data[:,1,:,:]
        b = data[:,2,:,:]
        c = data[:,0,:,:]
        d = (a+b+c)/3
        print(d.shape)
        e=torch.unsqueeze(d,1)
        print(e.shape)
        print(target)
    """
    return test_loader

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)  #random seed

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    """    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    """
    test_loader = load_dataset()

    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  #most basic optimizer in pytorch
                                                                                #Momentum 動量??
    for epoch in range(1, args.epochs + 1):  #10 times
        print('\nepoch : ', epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()


"""
動態建立模型
上述利用autograd更新的過程揭露了PyTorch和TensorFlow、Theano等其他深度學習框架最不一樣的差異：
PyTorch會動態的在每一次更新/計算結果的過程建立有向圖，每一行對Variable的操作都是建立模型的過程；
其他框架會先編譯整個模型再開始更新/計算。也許有人會懷疑，每一次都要重新建立模型是否會讓運算速度變慢，
但就我們的使用經驗是感覺不出來的。

這個動態建立有向圖的過程有兩個好處：

1.當我們的模型有錯誤的時候，PyTorch會被迫中止在發生錯誤的地方，並立即回報錯誤原因。其他框架如Keras，
因為需要靜態建立模型並呼叫compile，會在執行編譯時才回報錯誤的原因。要從錯誤的原因回推造成錯誤的程式碼不一定非常容易，
這方面的差異大大的影響我們除錯的速度。

2/動態的建立模型代表我們能夠根據每一次的輸入來建立對應的模型，這點對於某些特殊的RNN模型特別有用，
在TensorFlow這樣靜態建立模型的框架中是很難實踐的。

"""