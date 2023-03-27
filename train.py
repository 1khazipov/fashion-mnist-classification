import logging
import os
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torch.utils.data import random_split
import hydra
from omegaconf import DictConfig


log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config.yaml")
def func(cfg: DictConfig):
    log.info("Info level message")
    print(cfg)
    return cfg


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])

    def prepare_data(self):
        # download
        datasets.FashionMNIST(self.data_dir, download=True, train=True)
        datasets.FashionMNIST(self.data_dir, download=True, train=False)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = datasets.FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = \
                random_split(mnist_full, [int(len(mnist_full) * 0.8), int(len(mnist_full) * 0.2)])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = datasets.FashionMNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = datasets.FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


class Fashion_MNIST_Classifier(pl.LightningModule):
    def __init__(self,
                 optimizer='adam',
                 lr=1e-3,
                 batch_size=4,
                 epochs=30):
        super(Fashion_MNIST_Classifier, self).__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.fc1 = nn.Linear(in_features=28*28, out_features=32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 128)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(128, 64)
        self.act4 = nn.ReLU()
        self.output = nn.Linear(64, 10)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = self.fc4(x)
        x = self.act4(x)
        x = self.output(x)
        x = self.sm(x)
        return x

    def training_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        self.log("train_loss", F.nll_loss(y_hat, y))
        self.log('accuracy', (y_hat.argmax(dim=1) == y).sum()*1.0 / len(y))
        return {'loss': F.nll_loss(y_hat, y)}

    def validation_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        self.log('validation loss', F.nll_loss(y_hat, y))
        self.log('accuracy', (y_hat.argmax(dim=1) == y).sum()*1.0 / len(y))
        return {'loss': F.nll_loss(y_hat, y)}

    def test_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        self.log("test loss", F.nll_loss(y_hat, y))
        self.log('accuracy', (y_hat.argmax(dim=1) == y).sum()*1.0 / len(y))
        return {'loss': F.nll_loss(y_hat, y)}

    def predict_step(self, batch, batch_idx, dataloader_index=0):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
        else:
            optimizer = optim.SGD(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    cfg = func()
    model = Fashion_MNIST_Classifier()
    data_module = MNISTDataModule(os.getcwd())
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=30)
    trainer.fit(model, data_module)
    trainer.validate(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)
    predictions = trainer.predict(model=model, datamodule=data_module)
    # print(predictions)
