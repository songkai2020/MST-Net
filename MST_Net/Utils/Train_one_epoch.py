import torch
from tqdm import tqdm
from Utils.tools import Loss,exponential_decay
import sys

def train_one_epoch(model,train_data_loader,epoch,epochs,optimizer,LR, optim,device):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_data_loader, file=sys.stdout)
    for step,train_data in enumerate(train_bar):
        data, label = train_data
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        predict = model(data)
        loss = Loss(predict,label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        lr_temp = exponential_decay(LR, epoch, 5, 0.9)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_temp

        train_bar.desc = "train epoch[{}/{}] loss:{:.10f} LR {}".format(epoch + 1,epochs,loss,lr_temp)
    return running_loss


def valdate(model,val_data_loader,epoch,epochs,LR, optim,device):

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_data_loader, file=sys.stdout)
        for step,val_data in enumerate(val_bar):
            data,label = val_data
            data = data.to(device)
            label = label.to(device)
            predict = model(data)
            loss = Loss(predict, label)
            val_loss += loss

            lr_temp = exponential_decay(LR, epoch, 2, 0.9)
            for param_group in optim.param_groups:
                param_group['lr'] = lr_temp

            val_bar.desc = "val epoch[{}/{}] loss:{:.8f}".format(epoch + 1, epochs, loss)

    return val_loss



