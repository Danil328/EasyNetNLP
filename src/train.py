import os
import shutil
import torch
import torch.optim as optim
import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import config
from dataset.dataset import StatusDataset
from model.dan import DAN, init_weights
from model.transformer import Net
from utils.loss import FocalLoss2
from utils.scores import calculate_score

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = StatusDataset(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)

    val_dataset = StatusDataset(mode='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    if config.network_type.lower() == "transformer":
        easynet = Net(embeddings=torch.tensor(train_dataset.bpemb_ru.emb.vectors)).to(device)
    elif config.network_type.lower() == "dan":
        easynet = DAN(emb_weights=torch.tensor(train_dataset.bpemb_ru.emb.vectors)).to(device)
        init_weights(easynet, init_type='normal', scale=0.1)
    else:
        raise ("Error: {} is not implement".format(config.network_type))

    # criterion = nn.BCELoss()
    criterion = FocalLoss2()
    criterion.to(device)
    optimizer = optim.Adam(easynet.parameters(), lr=config.learning_rate, weight_decay=0)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)

    try:
        shutil.rmtree('output/logs', ignore_errors=True)
    except Exception:
        pass
    os.mkdir('output/logs')

    writer = SummaryWriter(log_dir='output/logs')
    global_step = 0
    for epoch in range(config.n_epochs):
        easynet.train()
        train_bar = tqdm.tqdm(train_loader)
        train_bar.set_description_str(desc=f"N epochs - {epoch+1}")
        for step, (token_ids, label) in enumerate(train_bar):
            global_step += 1
            token_ids = token_ids.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            predict = easynet(token_ids)
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()

            writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=global_step)

        # validation
        easynet.eval()
        valid_bar = tqdm.tqdm(val_loader)
        y_pred, y_true = [], []
        valid_loss = 0
        for step, (token_ids, label) in enumerate(valid_bar):
            token_ids = token_ids.to(device)
            label = label.to(device)
            with torch.no_grad():
                predict = easynet(token_ids)
                loss = criterion(predict, label)

            valid_loss += loss.item()
            y_pred += predict.data.cpu().numpy().squeeze().tolist()
            y_true += label.data.cpu().numpy().squeeze().tolist()

        aucROC, f1, eer, aucPR = calculate_score(y_true, y_pred)
        writer.add_scalar(tag="val_loss", scalar_value=valid_loss / val_loader.__len__(), global_step=epoch)
        writer.add_scalar(tag="aucROC", scalar_value=aucROC, global_step=epoch)
        writer.add_scalar(tag="aucPR", scalar_value=aucPR, global_step=epoch)
        writer.add_scalar(tag="f1", scalar_value=f1, global_step=epoch)
        writer.add_scalar(tag="eer", scalar_value=eer, global_step=epoch)

        scheduler.step()

        torch.save(easynet.state_dict(), config.path_to_save_model + '_epoch_{}'.format(epoch+1) + '.h5')


