from lib import *
from config import *

def make_datapath_list(phase='train'):
    rootpath = './dataset/'
    target_path = osp.join(rootpath + phase + "/**/*.jpg")

    path_list = [path for path in glob.glob(target_path)]
    return path_list

# Train model
def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):

    # device GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # move network to train on device 
    net.to(device)

    # boost network speed on gpu
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloader_dict[phase]):
                # move inputs, labels to GPU/CPU device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set gradients of optimizer to zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    labels = labels
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, axis=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.shape[0]
                    epoch_corrects += torch.sum(preds==labels.data)


            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

    torch.save(net.state_dict(), save_path)


def load_model(net, model_path):
    load_weights = torch.load(model_path)
    net.load_state_dict(load_weights)

    # train on gpu, load model on cpu machine
    # load_weights = torch.load(model_path, map_location=("cuda:0", "cpu"))
    # net.load_state_dict(load_weights)
    
    # display fine-tuning model's architecture
    for name, param in net.named_parameters():
        print(name, param)