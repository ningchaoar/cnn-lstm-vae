import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from config_vae import Config
from data import CustomDataset
from model import CoupletSeq2SeqWithVAE


def train_vae():
    config = Config("resources/couplet/chars_sort.txt")
    couplet_dataset = CustomDataset(config)
    dataloader = DataLoader(couplet_dataset, batch_size=config.batch_size, shuffle=True,
                            collate_fn=couplet_dataset.custom_collate_fn_2)
    model = CoupletSeq2SeqWithVAE(config)
    model.to(config.device)
    if config.wandb:
        import wandb
        wandb.init()
        wandb.watch(model, log_freq=100)
    loss_fn = nn.CrossEntropyLoss(reduction='sum').to(config.device)
    optimizer = torch.optim.Adam(model.parameters())
    logger_string = "epoch: {}, cross_entropy_loss: {}, kld_loss: {}\n"
    print("start training...")
    for epoch in range(config.epoch):
        model.train()  # set mode to train
        step = 0
        for data in tqdm(dataloader, disable=not config.show_process_bar):
            inputs, targets = data
            logits, kld_loss = model(inputs)
            label_loss = loss_fn(logits.permute(0, 2, 1), targets) / config.batch_size
            loss = label_loss + config.kl_weight * kld_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if config.wandb:
                if step % 10 == 0:
                    wandb.log({"loss": loss})
        print(logger_string.format(epoch, label_loss.item(), kld_loss.item()))
        preview_result(model, config)
        save_model(model, "model_{}_{:.3f}.pt".format(epoch, loss.item()), config.output_dir)


def save_model(model: nn.Module, model_name: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, model_name)
    torch.save(model.state_dict(), output_path)


def preview_result(model: nn.Module, config: Config):
    generator = model.get_decoder()
    generator.eval()

    for _ in range(5):
        x = torch.randn((1, config.hidden_dim)).to(config.device)
        mask = torch.ones((1, 2 * config.max_length + 1)).to(config.device)
        logits = generator((x, mask))
        unk_mask = torch.zeros(logits.shape[-1]).to(config.device)
        unk_mask[-1] = -10000
        unk_mask = torch.tile(unk_mask, (1, logits.shape[1], 1))
        outputs = torch.argmax(logits + unk_mask, dim=2).long()
        predict = ""
        for i in outputs[0].tolist():
            if i == 0:
                predict += "<PAD>"
            elif i == 1:
                predict += "<SEP>"
            else:
                predict += config.id2char[i]
        print("result: {}\n".format(predict))


if __name__ == "__main__":
    train_vae()
