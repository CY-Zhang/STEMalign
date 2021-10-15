from VGG16 import Net
from Hyperparameter import Hyperparameter
from RonchigramDataset import RonchigramDataset
from EarlyStopping import EarlyStopping
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn

class CNNtraining():
    def main(self):
        # initialize and select the device to use
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device('cuda:0')
        print(device)
        
        # build and modify hyperparameters
        par = Hyperparameter()
        '''
        Modify hyperparamters here if necessary
        '''

        # load raw data
        input_path = par.dict['training_data']['path']
        train_data = np.load(input_path + par.dict['training_data']['data'])
        print(train_data.shape)

        train_label = np.load(input_path + par.dict['training_data']['label_1'])
        train_label_1 = (train_label - np.amin(train_label))/(np.amax(train_label) - np.amin(train_label))
        print(train_label.shape)

        train_label = np.load(input_path + par.dict['training_data']['label_2'])
        train_label_2 = (train_label - np.amin(train_label))/(np.amax(train_label) - np.amin(train_label))
        print(train_label.shape)

        train_label = (train_label_1 + train_label_2) / 2

        # define image segmentation
        transform = transforms.Compose(
            [
                transforms.Resize(par.dict['segmentation']['resize']),
                transforms.RandomResizedCrop((128, 128), scale = par.dict['segmentation']['scale'], ratio = par.dict['segmentation']['ratio']),
                transforms.ToTensor(),
                # TODO: need to add random shear here
                # option to normalize a tensor with mean and standard deviation, similar to featurewise center in Keras
    #             transforms.Normalize((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            ]
        )

        batch_size = par.dict['batch_size']
        shuffle = True
        pin_memory = True
        num_workers = 1

        # build dataset
        dataset = RonchigramDataset(train_data, train_label, transform = transform)

        n_train = int(len(train_data) * par.dict['training_data']['split'])
        n_val = len(train_data) - n_train
        train_set, validation_set = torch.utils.data.random_split(dataset,[n_train,n_val])
        train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
        validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_meory=pin_memory)

        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        self.imshow(torchvision.utils.make_grid(images))

        # build model
        model = Net(dropout = 0.3, device = device, linear_shape = par.dict['architecture']['linear_shape'][0]).to(device)
        model.load_pretrained()
        model.lock_base()
        criterion = nn.MSELoss(reduction = 'mean')

        # start training
        stage = 0
        learning_rate = par.dict['optimizer']['lr'][stage]
        num_epochs = par.dict['epoch'][stage]
        betas = par.dict['optimizer']['beta'][stage]
        eps = par.dict['optimizer']['eps'][stage]
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, betas = betas, eps = eps)
        early_stopping = EarlyStopping(patience = par.dict['patience'], min_delta = par.dict['min_delta'])

        model.train()
        for epoch in range(num_epochs):
            loop = tqdm(train_loader, total = len(train_loader), leave = True)
            
            if epoch % 2 == 0:
                val_acc = self.check_accuracy(validation_loader, model, device)
                loop.set_postfix(val_acc = val_acc)
                early_stopping(float(val_acc))
                if early_stopping.early_stop:
                    par.add_training_process(acc_loss / counter , float(val_acc), epoch)
                    break
                    
            acc_loss = 0
            counter = 0
            
            for imgs, labels in loop:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = torch.squeeze(model(imgs))
                loss = criterion(outputs, labels)
                acc_loss += loss * labels.shape[0]
                counter += labels.shape[0]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss = loss.item())
            
            par.add_training_process(acc_loss / counter , val_acc, epoch)
            print(f"Training acc: {float(acc_loss) / float(counter):.6f}, Validation accuracy {float(val_acc):.6f}")

    def imshow(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def check_accuracy(self, loader, model, device):

        sum_MSE = 0
        counter = 0
        loss = nn.MSELoss(reduction = 'mean')
        model.eval()
        
        # define two lists to save truth and predictions, for the plot only.
        y_list = torch.empty(0).to(device = device)
        pred_list = torch.empty(0).to(device = device)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device)

                pred = torch.squeeze(model(x))
                pred_list = torch.cat((pred_list, pred), 0)
                y_list = torch.cat((y_list, y), 0)
                sum_MSE += loss(pred, y) * y.shape[0]
                counter += y.shape[0]
        
        fig, ax = plt.subplots(1,1, figsize = (5,5))
        img = ax.scatter(y_list.cpu().numpy(), pred_list.cpu().numpy(), s = 1)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.plot(np.linspace(0,1,100), np.linspace(0,1,100),'--', c = 'red')
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlabel('Truth',fontsize = 16)
        ax.set_ylabel('Prediction', fontsize = 16)
        plt.show()
        
        model.train()
    #     print( f"Got accuracy {float(sum_MSE)/float(counter):.6f}" )   
        return f"{float(sum_MSE)/float(counter):.6f}"