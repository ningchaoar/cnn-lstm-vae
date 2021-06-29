import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from model import PoemModel
from data import CustomDataset


def train():
    config = Config("resources/couplet/chars_sort.txt")
    couplet_dataset = CustomDataset(config)
    dataloader = DataLoader(couplet_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=couplet_dataset.custom_collate_fn)
    model = PoemModel(config)
    model.train()  # set mode to train
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters())
    logger_string = "epoch: {}, loss: {}\n"
    print("start training...")
    for epoch in range(config.epoch):
        for data in tqdm(dataloader):
            inputs, targets = data
            logits = model(inputs)
            loss = loss_fn(logits.permute(0, 2, 1), targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(logger_string.format(epoch, loss.item()))


if __name__ == "__main__":
    train()
