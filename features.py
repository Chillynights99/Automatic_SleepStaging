import pandas as pd
from scipy import signal
import numpy as np
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

def feature_extraction(stage_no, raw_sig, stages, epochs):
    indices = []
    durations = []

    # Find N1, N2, N3, R, W stages
    stages_marks = [i for i, s in enumerate(stages) if "Stage - {}".format(stage_no) in s]

    # Epochs and durations
    for k in stages_marks:
        # 마지막 스테이지일 경우 pass - epochs[k+1] 없으므로
        if epochs[k] == epochs[-1]:
            pass
        else:
            indices.append(epochs[k])                      # 스테이지 N1 이면 맞는 에폭 넘버 (ex.76
            durations.append((epochs[k+1] - epochs[k]))

    Features_acc = []
    for i in range(len(indices)):                           # 0, 1, 2, 3, 4, 5 ... (인덱스 넘버)
        for k in range(durations[i]):                       # durations[0] = 4 이면 k는 0, 1, 2, 3 이렇게 됨
            start_epoch = indices[i] + k                    # start_epoch = 76 + 0

            # STFT
            sig = raw_sig[6000 * (start_epoch - 1): 6000 * start_epoch - 1]         # 30초에 200 샘플링레이트
            kaiser = signal.windows.kaiser(M=51, beta=0.5)                         # STFT kaiser window = 윈도우 0.5초 overlap 50%
            result = signal.spectrogram(np.squeeze(sig), fs=200, window=kaiser, noverlap=50, nfft=1024)     # spectrogram 결과값
            P = pd.DataFrame(result[2])     # 513 * 118 matrix (가로축 주파수 세로축 시간)

            Features = []
            # Average of lower delta (frequency band 0.5-2Hz)
            f_lowerdelta = P[6:22].mean(axis=0)
            f_lowerdelta_mean = f_lowerdelta.mean()  # lowerdelta mean 118*1
            Features.append(f_lowerdelta_mean)

            # Average of the higher delta (2-4Hz)
            f_higherdelta = P[22:42].mean(axis=0)
            f_higherdelta_mean = f_higherdelta.mean()
            Features.append(f_higherdelta_mean)

            # Maximum delta value (0-4Hz)
            f_delta = P[0:42]
            f_delta_mean = f_delta.mean(axis=0)
            f_delta_max = f_delta_mean.max()
            Features.append(f_delta_max)

            # Average of the sigma band (12-14Hz)
            f_sigma = P[123:154].mean(axis=0)
            f_sigma_mean = f_sigma.mean()
            Features.append(f_sigma_mean)

            # Maximum of sigma band (12-14Hz)
            f_sigma_max = f_sigma_mean.max()
            Features.append(f_sigma_max)

            # Lower theta value 50% (4-6Hz)
            f_theta_lower = P[41:62].mean(axis=0)
            f_theta_lower_mean = f_theta_lower.mean()
            Features.append(f_theta_lower_mean)

            # Upper theta value 50% (6-8Hz)
            f_theta_upper = P[62:83].mean(axis=0)
            f_theta_upper_mean = f_theta_upper.mean()
            Features.append(f_theta_upper_mean)

            # Lower alpha value (8-10Hz)
            f_alpha_lower = P[83:108].mean(axis=0)
            f_alpha_lower_mean = f_alpha_lower.mean()
            Features.append(f_alpha_lower_mean)

            # Upper alpha value (10-12Hz)
            f_alpha_upper = P[108:134].mean(axis=0)
            f_alpha_uper_mean = f_alpha_upper.mean()
            Features.append(f_alpha_uper_mean)

            # Lower Beta value (12-30Hz)
            f_beta_lower = P[135:227].mean(axis=0)
            f_beta_lower_mean = f_beta_lower.mean()
            Features.append(f_beta_lower_mean)

            # Upper Beta value
            f_beta_upper = P[227:318].mean(axis=0)
            f_beta_upper_mean = f_beta_upper.mean()
            Features.append(f_beta_upper_mean)

            # Average of Gamma (30-50Hz)
            f_gamma = P[318:513].mean(axis=0)
            f_gamma_mean = f_gamma.mean()
            Features.append(f_gamma_mean)

            # Index 추가
            Features.append(start_epoch)

            # Accumulation
            if len(Features_acc) == 0:
                Features_acc = Features
            else:
                Features_acc = np.vstack((Features_acc, Features))

    return Features_acc

def save_raw_features(farray, stage_no):
    dataframe = pd.DataFrame(farray)
    dataframe['label'] = "{}".format(stage_no)
    dataframe.to_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Test_{}_features".format(stage_no))











            
            






            
    
    
