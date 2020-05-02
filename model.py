from lib import *
from image_transform import ImageTransform
from config import *
from utils import make_datapath_list, train_model, load_model
from dataset import MyDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*7*7, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = F.relu(x)
        x = max_pool2d(x, (2, 2))

        x = self.conv4(x)
        x = F.relu(x)
        x = max_pool2d(x, (2, 2))

        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    # Create dataset objects
    train_dataset = MyDataset(train_list, ImageTransform((HEIGHT, WIDTH), MEAN, STD), phase='train')
    val_dataset = MyDataset(val_list, ImageTransform((HEIGHT, WIDTH), MEAN, STD), phase='val')

    # Create dataloader objects
    train_dataloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

    # Build model
    net = Net()
    print(net)


    # Loss 
    criterior = nn.CrossEntropyLoss()

    # Optimizer
    params = net.parameters()
    optimizer = optim.RMSprop(params, lr=1e-4)

    # Training model
    # train_model(net, dataloader_dict, criterior, optimizer, NUM_EPOCHS)
    
    print(len(train_list))
    print(len(val_list))


if __name__ == "__main__":
    main()

    ## load model 
    # load_model(net, save_path)