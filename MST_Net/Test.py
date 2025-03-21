import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
# from Model.MST_Net import MST_Net
from Model.MST_Net_Gate import MST_Net



def test(model,device,test_data,save_path):

        for img_name in os.listdir(test_data):
            test_data_path = os.path.join(test_data,img_name)
            with torch.no_grad():
                img = Image.open(test_data_path).convert("L")
                preprocess = transforms.Compose([
                    transforms.ToTensor(),
                ])
                img = preprocess(img)
                img = img.reshape((1,1,256,256)).to(device)
                output = model(img)

                output = np.reshape(output.cpu().detach().numpy(),(256,256))

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.imsave(os.path.join(save_path,img_name),
                           output, cmap='gray')
                print(test_data_path)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MST_Net().to(device)
    model_path = '/home/MST_Net/Model/MST_Net.pth'
    data_path = '/home/MST_Net/Data/test'
    save_path = '/home/MST_Net/Data/result'

    model.load_state_dict(torch.load(model_path))
    model.eval()

    test(model, device, data_path, save_path)



