import numpy as np
import numpy.random
import os
import pandas as pd
from PIL import Image
import gc
import random


def Forward_model(data_path,patterns,DMD_frame_rate,photon_count_rate,detection_efficiency,possion_noise_level,gauss_mean,
                  gauss_std,data_without_noise_savepath,data_with_possion_noise_savepath,data_with_gauss_noise_savepath,
                  data_with_mix_noise_savepath,noise_level,count):
    '''
    :param data_path: The path of target images.
    :param patterns: Matrix of patterns.
    :param DMD_frame_rate:Pattern projection rates of DMD.
    :param photon_count_rate:Average photon count rate.
    :param detection_efficiency: Detection efficiency of detector.
    :param possion_noise_level: Possion noise level.
    :param gauss_mean: Guassian noise level.
    :param gauss_std: Guassian noise level.
    :param data_without_noise_savepath: The path to save the noiseless data.
    :param data_with_possion_noise_savepath: The path to save the possion noise data.
    :param data_with_gauss_noise_savepath: The path to save the guassian noise data.
    :param data_with_mix_noise_savepath: The path to save the mix noise data.
    :param count:
    :return:
    '''

    Time = 1/DMD_frame_rate
    imgs = os.listdir(data_path)
    imgs_exists = os.listdir(data_with_mix_noise_savepath)

    for img_name in imgs:
        if img_name.split('.')[0] + '.csv' in imgs_exists:
            print(img_name.split('.')[0] + '.csv')
        else:
            # Read target image
            total_path = os.path.join(data_path, img_name)
            img = np.array(Image.open(total_path).convert("L"))
            W, H = img.shape[0], img.shape[1]
            img = np.transpose(img)

            # Calculate the inital photons for each pattern.
            img = np.reshape(img, (W * H, 1))
            intensity = np.matmul(patterns, img)

            # Adjust photon count for each pattern.
            avg_count_per_pattern = np.sum(intensity)/len(intensity)
            rate = ((photon_count_rate*Time)/detection_efficiency)/avg_count_per_pattern
            photon_count_without_noise = []
            photon_count_with_possion_noise = []
            photon_count_with_gauss_noise = []
            photon_count_with_mix_noise = []
            for j in range(len(intensity)):
                Signal_photons_adjust = int(intensity[j]*rate*detection_efficiency)
                Signal_photons_with_possion_noise = Signal_photons_adjust + np.random.poisson(possion_noise_level,1)
                Signal_photons_gauss_noise = Signal_photons_adjust + np.random.normal(gauss_mean, gauss_std)
                Signal_photons_mix_noise = Signal_photons_adjust + (np.random.poisson(possion_noise_level,1) + np.random.normal(gauss_mean, gauss_std))*0.5

                photon_count_without_noise.append(int(Signal_photons_adjust))
                photon_count_with_possion_noise.append(int(Signal_photons_with_possion_noise))
                photon_count_with_gauss_noise.append(int(Signal_photons_gauss_noise))
                photon_count_with_mix_noise.append(int(Signal_photons_mix_noise))

            intensity_without_noise = np.transpose(np.expand_dims(np.array(photon_count_without_noise), axis=-1))
            intensity_with_possion_noise = np.transpose(np.expand_dims(np.array(photon_count_with_possion_noise), axis=-1))
            intensity_with_gauss_noise = np.transpose(np.expand_dims(np.array(photon_count_with_gauss_noise), axis=-1))
            intensity_with_mix_noise = np.transpose(np.expand_dims(np.array(photon_count_with_mix_noise), axis=-1))

            dp1 = pd.DataFrame(intensity_without_noise)
            dp2 = pd.DataFrame(intensity_with_possion_noise)
            dp3 = pd.DataFrame(intensity_with_gauss_noise)
            dp4 = pd.DataFrame(intensity_with_mix_noise)

            save_name1 = os.path.join(data_without_noise_savepath,
                                     img_name.split('.')[0] + '_' + str(photon_count_rate)+ '_' +str(noise_level) + '_' + str(
                                         count) + '_' + '.csv')
            save_name2 = os.path.join(data_with_possion_noise_savepath,
                                     img_name.split('.')[0] + '_' + str(photon_count_rate)+ '_' +str(noise_level) + '_' + str(
                                         count) + '_' + '.csv')
            save_name3 = os.path.join(data_with_gauss_noise_savepath,
                                     img_name.split('.')[0] + '_' + str(photon_count_rate) + '_' +str(noise_level) + '_' +str(
                                         count) + '_' + '.csv')
            save_name4 = os.path.join(data_with_mix_noise_savepath,
                                     img_name.split('.')[0] + '_' + str(photon_count_rate)+ '_' +str(noise_level) + '_' + str(
                                         count) + '_' + '.csv')

            dp1.to_csv(save_name1, index=False, header=True)
            dp2.to_csv(save_name2, index=False, header=True)
            dp3.to_csv(save_name3, index=False, header=True)
            dp4.to_csv(save_name4, index=False, header=True)
            print(img_name.split('.')[0] + '_' + str(photon_count_rate) + '_' + str(
                                         count)+ '_' +str(noise_level) + '_' +'.csv')
    return 0






if __name__ == '__main__':

    data_path = r'C:\Users\ASUS\Desktop\GT_full'
    pattern_path = r'C:\Users\ASUS\Desktop\patterns\m_256_001.csv'
    data_without_noise_savepath = r'C:\Users\ASUS\Desktop\Diff_noise_model\data_wo_noise'
    data_with_possion_noise_savepath = r'C:\Users\ASUS\Desktop\Diff_noise_model\data_w_possion_noise'
    data_with_gauss_noise_savepath = r'C:\Users\ASUS\Desktop\Diff_noise_model\data_w_gauss_noise'
    data_with_mix_noise_savepath = r'C:\Users\ASUS\Desktop\Diff_noise_model\data_w_mix_noise'
    patterns = pd.read_csv(pattern_path, header=None).to_numpy()
    DMD_frame_rate = 655
    detection_efficiency = 0.6

    count = 0
    for possion_noise_level in [800,1500,2500]:
        gauss_mean =possion_noise_level
        gauss_std = int(gauss_mean*random.uniform(0.1,0.3))
        for photon_count_rate in [512000,812000,1512000,3000000]:
            rate = photon_count_rate/3000000
            Enviromental_detection_efficiency = 0.6
            Enviromental_noise = 0
            Forward_model(data_path, patterns, DMD_frame_rate, photon_count_rate, detection_efficiency,
                        int(possion_noise_level*rate), int(gauss_mean*rate),int(gauss_std*rate), data_without_noise_savepath,
                        data_with_possion_noise_savepath,data_with_gauss_noise_savepath,data_with_mix_noise_savepath,possion_noise_level, count)
            count = count+1

