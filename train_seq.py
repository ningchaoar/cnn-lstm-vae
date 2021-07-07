import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from config_seq import Config
from data import CustomDataset
from model import CoupletSeqLabeling


def train_seq_labeling():
    config = Config("resources/couplet/chars_sort.txt")
    couplet_dataset = CustomDataset(config)
    dataloader = DataLoader(couplet_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=couplet_dataset.custom_collate_fn_1)
    model = CoupletSeqLabeling(config)
    model.to(config.device)
    loss_fn = nn.CrossEntropyLoss(reduction='mean').to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logger_string = "epoch: {}, loss: {}\n"
    print("start training...")
    for epoch in range(config.epoch):
        model.train()  # set mode to train
        for data in tqdm(dataloader):
            inputs, targets = data
            logits = model(inputs)
            loss = loss_fn(logits.permute(0, 2, 1), targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(logger_string.format(epoch, loss.item()))
        preview_result(model, config)
        save_model(model, "model_{}_{:.3f}.pt".format(epoch, loss.item()), config.output_dir)


def save_model(model: nn.Module, model_name: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, model_name)
    torch.save(model.state_dict(), output_path)


def preview_result(model: nn.Module, config: Config):
    model.eval()
    examples = ["兴漫香江红袖舞",
                "河伯捧觞，游观沧海",
                "清风凝白雪",
                "木",
                "讲道德，树新风，遵守法纪"]
    targets = ["龙腾粤海碧云飞",
               "天帝悬车，凭乘风云",
               "妙舞散红霞",
               "花",
               "学雷锋，做好事，为国利民"]
    for e, t in zip(examples, targets):
        ids = torch.Tensor(np.array([config.char2id.get(c, config.char_table_size + 1) for c in e])).long().unsqueeze(0).to(config.device)
        masks = torch.ones((1, len(e))).to(config.device)
        lengths = [len(e)]
        logits = model((ids, masks, lengths))
        outputs = torch.argmax(logits, dim=2).long()
        predict = "".join([config.id2char.get(i, "<UNK>") if i != 0 else " " for i in outputs[0].tolist()])
        print("input: {}\ntarget: {}\npredict: {}\n".format(e, t, predict))


if __name__ == "__main__":
    train_seq_labeling()
