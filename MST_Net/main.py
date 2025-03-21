import torch
import argparse
from Utils.tools import read_split_data_img2img
from Utils.MyDataSet import MyDataSetimg2img
from Utils.Train_one_epoch import train_one_epoch,valdate
from torchvision import transforms,datasets,utils
from torch.optim import Adam
from Model.MST_Net import MST_Net
# from Model.MST_Net_Gate import MST_Net



def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_loss = 100

    train_imgs_path, train_labels, val_imgs_path, val_labels = read_split_data_img2img(args.img_path,
                                                                                          args.label_path, 0.05)

    data_transform = {
        "train" : transforms.Compose([
            transforms.ToTensor()
        ]),
        "test" : transforms.Compose([
            transforms.ToTensor()
        ])
    }

    train_dataset = MyDataSetimg2img(train_imgs_path,train_labels,data_transform['train'])
    val_dataset = MyDataSetimg2img(val_imgs_path,val_labels,data_transform['test'])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = args.batch_size,
                                                    shuffle = True,
                                                    num_workers = 0)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

    model_save_path = args.save_path
    model =MST_Net().to(device)



    optimizer = Adam(model.parameters(),lr = args.LR)
    for epoch in range(args.EPOCH):
        train_loss = train_one_epoch(model,train_data_loader,epoch,args.EPOCH,optimizer,args.LR,optimizer,device)
        val_loss = valdate(model,val_data_loader,epoch,args.EPOCH,args.LR,optimizer,device)
        avg_train_loss = train_loss/train_num
        avg_val_loss = val_loss/val_num
        print("Avg train loss:{:.10f},Avg val loss:{:.10f}".format(avg_train_loss,avg_val_loss))
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
        if  epoch % 5 == 0:
            torch.save(model.state_dict(), model_save_path.split('.')[0]+'_'+str(epoch)+'.pth')




def parse_args():
    parser = argparse.ArgumentParser(description="MST_Net training")
    parser.add_argument("--img_path",default=r'/home/Dataset/Image',type=str)
    parser.add_argument("--label_path",default=r'/home/Dataset/GT',type=str)
    parser.add_argument("--EPOCH",default = 100,type = int)
    parser.add_argument("--LR", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--save_path", default= '/home/Physics_model/MST_Net.pth', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)


