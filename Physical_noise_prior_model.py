import numpy as np
import numpy.random
import os
import pandas as pd
from PIL import Image
import gc
import random


def Physical_noise_prior_model(data_path, patterns, DMD_frame_rate, time_res, illum_intensity, dead_time, amb_noise_modulate,
                               amb_noise_wo_modulate,detection_efficiency, dark_count,post_pulse_prob, data_save_path,resolution:int = None):

    '''
    :param data_path: The path of target images.
    :param patterns: Matrix of patterns.
    :param DMD_frame_rate: Pattern projection rates of DMD.
    :param time_res: Resolution of timeline.
    :param illum_intensity:  Illumination intensity of light source.
    :param dead_time: Dead time of the detector.
    :param amb_noise_modulate: Ambient noise level from modulated path(A  proportion (0-1) of total intensity).
    :param amb_noise_wo_modulate: Ambient noise level direct detected(Expected count per second accounts for detection efficiency).
    :param detection_efficiency: Detection efficiency of detector.
    :param dark_count: Dark count of the detector.
    :param post_pulse_prob: Post_pulse probility of detector.
    :param data_save_path: Save path of simulation data.
    :param resolution: Resolution of target image.
    :return:
    '''

    # Calculate the number of time bin in the timeline corresponding to each pattern.
    Time_single_pattern = 1/DMD_frame_rate
    Time_sq_len = int(Time_single_pattern*(1/time_res))

    # Probability coefficient of ambient event without modulation.
    Amb_wo_mod_prob = amb_noise_wo_modulate / (1 / time_res)

    # Probability coefficient of dark event.
    dark_prob = dark_count/(1/dead_time)

    # Preprocess of dead time noise.
    dead_spot_num = 0
    if dead_time>time_res:
        dead_spot_num = int(dead_time/time_res)

    imgs = os.listdir(data_path)
    imgs_exists = os.listdir(data_save_path)

    for img_name in imgs:
        save_name = img_name.split('.')[0]+'.csv'
        if save_name in imgs_exists:
            print(save_name)
        else:
            # Read target image.
            img_path = os.path.join(data_path, img_name)
            if resolution is not None:
                img = np.array(Image.open(img_path).resize((resolution, resolution)).convert("L"))
            else:
                img = np.array(Image.open(img_path).convert("L"))
            W, H = img.shape[0], img.shape[1]
            img = np.transpose(img)

            # The total intensity of object.
            total_intensity = 255*H*W

            # Calculate the probability of signal photon.
            img = img * illum_intensity # Adjust illumination intensity.

            if np.sum(img)>=total_intensity:
                print("Excessive light intensity leads to detector saturation!")
                break

            img = np.reshape(img, (W * H, 1))
            intensity = np.matmul(patterns, img)
            avg_intensity = sum(intensity)/patterns.shape[0]
            Obj_prob = (intensity+amb_noise_modulate*avg_intensity)/ total_intensity

            Obj_prob = np.clip(Obj_prob, 0, 1)
            Obj_prob = Obj_prob*detection_efficiency

            photon_count = []
            for m in range(len(Obj_prob)):
                # Signal Timeline
                Signal_sq = np.random.choice([0, 1], size=(Time_sq_len), p=[1 - Obj_prob[m][0], Obj_prob[m][0]])

                # Ambient noise without modulation Timeline
                Amb_wo_mod = np.random.choice([0, 1], size=(Time_sq_len), p=[1 - Amb_wo_mod_prob, Amb_wo_mod_prob])

                # Dark count Timeline
                Dark_count = np.random.choice([0, 1], size=(Time_sq_len), p=[1 - dark_prob, dark_prob])


                #Merge timelines (Signal + Dark count + Amb_wo_mod)
                Temp_sq = [min(a + b + c, 1) for a, b, c in zip(Signal_sq, Amb_wo_mod, Dark_count)]

                #Post-pulsing noise
                for n in range(Time_sq_len-1):
                    if Temp_sq[n] == 1 and Temp_sq[n+1] == 0:
                        Temp_sq[n + 1] = 1 if random.random() < post_pulse_prob else 0

                # Dead time noise
                if dead_spot_num!=0:
                    k = 0
                    while k < Time_sq_len:
                        if Temp_sq[k] == 1:
                            Temp_sq[k + 1:k + 1 + dead_spot_num] = [0] * min(dead_spot_num, Time_sq_len - k - 1)
                            k += dead_spot_num + 1
                        else:
                            k += 1

                photon_count_single = np.sum(Temp_sq)  # Photon counting of single pattern.
                photon_count.append(photon_count_single)

            Final_intensity = np.transpose(np.expand_dims(np.array(photon_count), axis=-1))
            dp = pd.DataFrame(Final_intensity)

            save_path = os.path.join(data_save_path, save_name)
            print(save_path)
            dp.to_csv(save_path, index=False,header=False)
            del photon_count
            gc.collect()
    return 0

if __name__ == '__main__':

    data_path = 'Path of target images'
    pattern_path = 'Pattern matrix path'
    patterns = pd.read_csv(pattern_path, header=None).to_numpy()
    DMD_frame_rate = 10
    time_res = 0.00001
    illum_intensity = 1.5
    dead_time = 0.00005
    amb_noise_mod = 0.01
    amb_noise_wo_mod = 500
    detection_efficiency = 0.7
    dark_count= 500
    post_pulse_prob =0.05
    data_save_path = 'Save path of measurements'

    Physical_noise_prior_model(data_path, patterns, DMD_frame_rate, time_res, illum_intensity, dead_time,
                               amb_noise_mod, amb_noise_wo_mod, detection_efficiency, dark_count, post_pulse_prob,
                               data_save_path)
