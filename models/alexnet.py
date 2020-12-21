import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Grayscale
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score


class AlexNet(nn.Module):
    # NOTE: The code of alexnet model definition is refered to pytorch, I only do some tiny modifications, such as stride, padding and channel numbers.
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexnetTrainer(object):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = AlexNet(num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def train(self, train_dir):
        train_dataset = ImageFolder(train_dir, transform=self.transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,)

        max_epoch = 50

        for epoch in range(max_epoch):
            self.model.train()
            epoch_average_meter = AverageMeter()
            for data, label in tqdm(train_dataloader):
                pred_prob = self.model(data)
                loss = self.loss_fn(pred_prob, label)
                self.optimizer.zero_grad()
                loss.backward()
                # print(loss.item())
                self.optimizer.step()
                epoch_average_meter.update(loss.item())
            print(f'epoch: {epoch+1}, avg_loss: {epoch_average_meter.average}')
        self.save_model()

    def val(self, val_dir,):
        self.model.eval()
        val_dataset = ImageFolder(val_dir, transform=self.transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False,)
        pred_scores=[]
        true_labels=[]
        for data, label in tqdm(val_dataloader):
            pred_prob = self.model(data)
            pred_prob = pred_prob.detach().cpu().numpy()
            pred_class = np.argmax(pred_prob, axis=1).tolist()
            label = label.cpu().numpy().tolist()
            pred_scores+=pred_class
            true_labels+=label
        acc=accuracy_score(true_labels,pred_scores)
        micro_f1_scores=f1_score(true_labels,pred_scores,average='micro')
        macro_f1_scores=f1_score(true_labels,pred_scores,average='macro')
        return acc,micro_f1_scores,macro_f1_scores

    def test(self, x):
        x = Image.fromarray(x)
        x = self.transforms(x)
        x = x.unsqueeze(dim=0)
        pred_prob = self.model(x)
        pred_prob = pred_prob.detach().cpu().numpy()
        pred_class = np.argmax(pred_prob, axis=1)
        return pred_class[0]

    def save_model(self, save_dir='./pretrained_models', model_name='alexnet_model.pth'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path='./pretrained_models/alexnet_model.pth'):
        self.model.load_state_dict(torch.load(model_path))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)
