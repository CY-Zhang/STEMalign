import torch.nn as nn
import torchvision
from torch.nn import Module
import torch.nn.functional as F

# Customized CNN model
class Net(Module):   
    def __init__(self, pretrained = False, dropout = 0.3, linear_shape = 512, device = device):
        super(Net, self).__init__()
        self.device = device
        print(self.device)

        self.conv1 = nn.Conv2d(3, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv5 = nn.Conv2d(128, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv6 = nn.Conv2d(256, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv7 = nn.Conv2d(256, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv8 = nn.Conv2d(256, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv9 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv10 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv11 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv12 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv13 = nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.fc1 = nn.Linear(4 * 4 * 512, linear_shape)
        self.dropout = nn.Dropout(p = dropout)

        self.fc2 = nn.Linear(linear_shape, 1)
    
    def lock_base(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.fc1.weight.requires_grad = True
        self.fc1.bias.requires_grad = True
        self.fc2.weight.requires_grad = True
        self.fc2.bias.requires_grad = True
    
    def unlock_base(self):
        for parameter in self.parameters():
            parameter.requires_grad = True
            
    def load_pretrained(self, weights):
        print("Loading weights and bias from VGG16.")
        vgg16 = torchvision.models.vgg16(pretrained = True)
        self.conv1.weight.data = vgg16.features[0].weight.data.to(device = self.device)
        self.conv1.bias.data = vgg16.features[0].bias.data.to(device = self.device)
        self.conv2.weight.data = vgg16.features[2].weight.data.to(device = self.device)
        self.conv2.bias.data = vgg16.features[2].bias.data.to(device = self.device)
        self.conv3.weight.data = vgg16.features[5].weight.data.to(device = self.device)
        self.conv3.bias.data = vgg16.features[5].bias.data.to(device = self.device)
        self.conv4.weight.data = vgg16.features[7].weight.data.to(device = self.device)
        self.conv4.bias.data = vgg16.features[7].bias.data.to(device = self.device)
        self.conv5.weight.data = vgg16.features[10].weight.data.to(device = self.device)
        self.conv5.bias.data = vgg16.features[10].bias.data.to(device = self.device)
        self.conv6.weight.data = vgg16.features[12].weight.data.to(device = self.device)
        self.conv6.bias.data = vgg16.features[12].bias.data.to(device = self.device)
        self.conv7.weight.data = vgg16.features[14].weight.data.to(device = self.device)
        self.conv7.bias.data = vgg16.features[14].bias.data.to(device = self.device)
        self.conv8.weight.data = vgg16.features[17].weight.data.to(device = self.device)
        self.conv8.bias.data = vgg16.features[17].bias.data.to(device = self.device)
        self.conv9.weight.data = vgg16.features[19].weight.data.to(device = self.device)
        self.conv9.bias.data = vgg16.features[19].bias.data.to(device = self.device)
        self.conv10.weight.data = vgg16.features[21].weight.data.to(device = self.device)
        self.conv10.bias.data = vgg16.features[21].bias.data.to(device = self.device)
        self.conv11.weight.data = vgg16.features[24].weight.data.to(device = self.device)
        self.conv11.bias.data = vgg16.features[24].bias.data.to(device = self.device)
        self.conv12.weight.data = vgg16.features[26].weight.data.to(device = self.device)
        self.conv12.bias.data = vgg16.features[26].bias.data.to(device = self.device)
        self.conv13.weight.data = vgg16.features[28].weight.data.to(device = self.device)
        self.conv13.bias.data = vgg16.features[28].bias.data.to(device = self.device)
    
    # Defining the forward pass    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x