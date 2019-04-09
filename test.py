import os
import shutil
import torch
import torch.optim as optim
import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import config
from Dataset.dataset import StatusDataset
from model.dan import DAN, init_weights
from model.transformer import Net
from utils.loss import FocalLoss2
from utils.scores import calculate_score

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = StatusDataset(path=config.path_to_test_data, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    if config.network_type.lower() == "transformer":
        easynet = Net(embeddings=torch.tensor(test_dataset.bpemb_ru.emb.vectors)).to(device)
    elif config.network_type.lower() == "dan":
        easynet = DAN(emb_weights=torch.tensor(test_dataset.bpemb_ru.emb.vectors)).to(device)
    else:
        raise ("Error: {} is not implement".format(config.network_type))

    easynet.load_state_dict(torch.load(config.path_to_save_model))
    easynet.eval()

    test_bar = tqdm.tqdm(test_loader)
    y_pred, y_true = [], []
    for step, (token_ids, label) in enumerate(test_bar):
        token_ids = token_ids.to(device)
        label = label.to(device)
        with torch.no_grad():
            predict = easynet(token_ids)

        y_pred += predict.data.cpu().numpy().squeeze().tolist()
        y_true += label.data.cpu().numpy().squeeze().tolist()

    aucROC, f1, eer, aucPR = calculate_score(y_true, y_pred)



