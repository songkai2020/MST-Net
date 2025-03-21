import os
import random
import torch

#Split the data into training and validation datasets
def read_split_data_img2img(img_path,label_path,rate):
    assert os.path.exists(img_path), "data root:{} does not exist.".format(img_path)
    train_imgs_path = []
    train_labels = []
    val_imgs_path = []
    val_labels = []

    imgs = os.listdir(img_path)
    random.shuffle(imgs)
    val_imgs = random.sample(imgs,k=int(len(imgs)*rate))

    for img in imgs:
        if img in val_imgs:
            val_imgs_path.append(os.path.join(img_path, img))
            val_labels.append(os.path.join(label_path, img.split('_')[0]+'.bmp'))
        else:
            train_imgs_path.append(os.path.join(img_path, img))
            train_labels.append(os.path.join(label_path, img.split('_')[0]+'.bmp'))

    return train_imgs_path,train_labels,val_imgs_path,val_labels



# Learning rate decay strategy
def exponential_decay(initial_lr, global_step, decay_steps, decay_rate):
    return initial_lr * (decay_rate ** (global_step / decay_steps))

def total_variation(image):
    x_diff = torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    y_diff = torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return x_diff + y_diff

def Loss(out,groundtruth,TV_strength = 1e-9):
    loss1 = torch.nn.MSELoss()
    loss2 = total_variation(out)
    loss = loss1(out,groundtruth)+TV_strength*loss2
    return loss


if __name__ == "__main__":
    pass