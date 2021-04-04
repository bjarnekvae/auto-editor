import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
import time
import os
import glob

class SkydivingDataset(Dataset):
    def __init__(self, files, classes, transform, im_size=(256, 256)):
        print('Creating data set...\n')
        self.files = files
        self.classes = classes
        self.im_size = im_size
        self.transform = transform

        self.X, self.Y = self.build_data_set()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x = self.transform(self.X[index])

        return x, self.Y[index]

    def build_data_set(self):
        X = np.zeros([len(self.files), self.im_size[1], self.im_size[0], 3], dtype=np.uint8)
        Y = np.zeros([len(self.files), len(self.classes)], dtype=np.uint8)

        for idx, file in enumerate(tqdm(self.files)):
            img = cv2.imread(file)
            X[idx] = cv2.resize(img, (256, 256), cv2.INTER_AREA)
            Y[idx, self.classes.index(file.split('/')[-2])] = 1

        return X, Y

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=(-10, 10)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def TandemVideoNet(classes):
    # Load model
    model = models.resnet18(pretrained=True)

    # Add on classifier
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.Dropout(0.2),
        nn.Linear(256, len(classes)))

    return model


class TandemVideoDetector:
    def __init__(self, model_path, classes):
        use_cuda = torch.cuda.is_available()
        self.classes = classes

        if use_cuda:
            self.device = "cuda"
            print("Using GPU :D")
        else:
            self.device = "cpu"
            print("Using CPU :(")

        self.model = TandemVideoNet(classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, img, draw_graphics=False):
        inp_img = val_transform(img)
        inp_img = np.expand_dims(inp_img, axis=0)
        inp_img = torch.tensor(inp_img).type(torch.FloatTensor).to(self.device)

        out = self.model(inp_img)
        out = torch.softmax(out, dim=1)
        out = out.cpu()
        out = out.data.numpy()
        out = np.squeeze(out)
        pred = np.argmax(out)
        out = list(out)
        results = zip(self.classes, out)

        if draw_graphics:
            for k, c in enumerate(results):
                if k == pred:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.putText(img, c[0] + ": {:.2f}".format(c[1]), (10, 60 + k * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3,
                            cv2.LINE_AA)
                cv2.putText(img, c[0] + ": {:.2f}".format(c[1]), (10, 60 + k * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                            cv2.LINE_AA)

            return results, img

        return results, pred


def _train_loop(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        output = torch.squeeze(output)

        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.9f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print("Average train loss: {}".format(train_loss))
    return train_loss


def _val_loop(model, device, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)

            data, target = data.to(device), target.to(device)

            output = model(data)
            output = torch.squeeze(output)

            test_loss += F.binary_cross_entropy_with_logits(output, target).item()

            output = torch.softmax(output, dim=1)

            output = output.cpu()
            target = target.cpu()

            pred = output.data.numpy()
            target = target.data.numpy()

            correct += np.sum(np.argmax(pred, axis=1) == np.argmax(target, axis=1))

    test_loss /= len(val_loader)
    acc = correct / len(val_loader.dataset)

    print('Validation set: Average loss: {:.8f}, Acc {:.8f}\n'.format(test_loss, acc))

    return test_loss, acc

def train_nn(training_set, export_path, epochs=200, validate=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    classes_path = glob.glob(training_set + "/*")
    print(classes_path)

    dataset = []
    classes = []
    for cp in classes_path:
        dataset += glob.glob(cp + "/*")
        classes += [cp.split("/")[-1]]

    dataset.sort()
    classes.sort()

    print("Found these classes:", classes)
    print("Found", len(dataset), "images for training")

    val = []
    train = []
    val_part = 9
    for i in range(len(dataset)):
        if i % val_part == 0:
            val.append(dataset[i])
        else:
            train.append(dataset[i])

    print("Training sets:", len(train))
    print("Validation sets:", len(val))

    train_dataset = SkydivingDataset(train, classes, train_transform)
    val_dataset = SkydivingDataset(val, classes, val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)

    images, labels = next(iter(train_dataloader))
    print("images-size:", images.shape)

    '''
    #cv2.namedWindow("img")
    for data, target in train_dataloader:
        for i in range(target.shape[0]):
            c = classes[np.argmax(target[i].data.numpy())]
            print(c)
            img = data[i].data.numpy()
            img = np.squeeze(img)
            img = np.uint8((img + 2) / 4.0 * 255.0)
            img = np.moveaxis(img, 0, -1)
            cv2.imshow("img", img)
            cv2.waitKey()
    quit()
    '''

    model = TandemVideoNet(classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters())

    torch.manual_seed(np.random.randint(0, 10000))

    plt.figure(1, figsize=(10, 10))
    plt.ion()
    plt.show()

    train_loss_arr = np.array([])
    val_loss_arr = np.array([])
    val_acc_arr = np.array([])
    best_acc = 0
    best_loss = np.inf
    torch_model = None
    best_acc_t = 0
    for epoch in range(0, epochs):
        train_loss = _train_loop(model, device, train_dataloader, optimizer, epoch)
        val_loss, acc = _val_loop(model, device, val_dataloader)

        train_loss_arr = np.append(train_loss_arr, train_loss)
        val_loss_arr = np.append(val_loss_arr, val_loss)
        val_acc_arr = np.append(val_acc_arr, acc)
        t = np.arange(val_loss_arr.shape[0])

        if acc > best_acc:
            best_acc = acc
            best_acc_t = epoch
            torch_model = model.state_dict().copy()

        if val_loss < best_loss:
            best_loss = val_loss
            #best_acc_t = epoch
            #torch_model = model.state_dict().copy()

        plt.clf()
        plot_x_len = val_loss_arr.shape[0]
        plot_y_len = np.max([val_loss_arr, val_acc_arr, train_loss_arr])
        plt.axis([0, plot_x_len, 0, plot_y_len])
        plt.plot(t, train_loss_arr)
        plt.plot(t, val_loss_arr)
        plt.plot(t, val_acc_arr)
        plt.plot([best_acc_t, best_acc_t], [best_loss-0.1, best_loss+0.1])
        plt.legend(['train_loss', 'val_loss, best: {}'.format(best_loss), 'val_acc, best: {}'.format(best_acc)])
        plt.draw()
        plt.pause(0.01)

        if acc == 1.0:
            break

    export_path = export_path + '/' + time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    img_out_path = export_path + '/' + 'output/'
    os.makedirs(img_out_path)

    torch.save(torch_model, export_path + '/nn_model.pt')
    plt.savefig(export_path + "/loss.png")

    if validate:

        detector = TandemVideoDetector(export_path + '/nn_model.pt', classes)

        for validation_file in tqdm(val):
            img = cv2.imread(validation_file)
            out, img = detector.predict(img, draw_graphics=True)
            cv2.imwrite(img_out_path + validation_file.split("/")[-1], img)

