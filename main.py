import glob
import os
import numpy as np
import pandas as pd

from mne.io import read_raw_edf
import features
import dhedfreader

def main():
    psg_names = glob.glob(os.path.join("C:/Users/User/Desktop/EWHADATASETS", "*.edf"))
    psg_names.sort()
    psg_fnames = np.asarray(psg_names)

    # PSG info
    sampling_rate = 200
    channel = "F3-A2"


    # features list initialization before loop
    N1_features_all = []
    N2_features_all = []
    N3_features_all = []
    W_features_all = []
    R_features_all = []

    for i in range(len(psg_fnames)):
        print("=====================================================================")
        print("Current file : {} ".format(psg_fnames[i]))

        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)      # use mne to open eeg raw file
        raw_ch_df = raw.to_data_frame()[channel]
        raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        ann = '{}_Annotation.csv'.format(psg_fnames[i][:-4])
        df = pd.read_csv(ann)

        epochs = []
        stages = []
        for i in range(len(df)):
            epoch = df.loc[i, "1"]
            stage = df.loc[i, "Start Recording"]
            if 'Stage' in stage:
                stages.append(stage)
                epochs.append(epoch)

        # 각 스테이지 피쳐 뽑기 ex> shape (94, 13)
        N1_features = features.feature_extraction("R", raw_ch_df , stages, epochs)
        N2_features = features.feature_extraction("N2", raw_ch_df, stages, epochs)
        N3_features = features.feature_extraction("N3", raw_ch_df, stages, epochs)
        W_features = features.feature_extraction("W", raw_ch_df, stages, epochs)
        R_features = features.feature_extraction("R", raw_ch_df, stages, epochs)


        # 모든 사람의 피쳐 합치기
        if len(N1_features_all) == 0:
            N1_features_all = N1_features
        else:
            N1_features_all = np.concatenate((N1_features_all, N1_features), axis=0)

        if len(N2_features_all) == 0:
            N2_features_all = N2_features
        else:
            N2_features_all = np.concatenate((N2_features_all, N2_features), axis=0)

        if len(N3_features_all) == 0:
            N3_features_all = N3_features
        else:
            N3_features_all = np.concatenate((N3_features_all, N3_features), axis=0)

        if len(W_features_all) == 0:
            W_features_all = W_features
        else:
            W_features_all = np.concatenate((W_features_all, W_features), axis=0)

        if len(R_features_all) == 0:
            R_features_all = R_features
        else:
            R_features_all = np.concatenate((R_features_all, R_features), axis=0)

    # 피쳐 파일로 저장하기
    features.save_raw_features(N1_features_all, "N1")
    features.save_raw_features(N2_features_all, "N2")
    features.save_raw_features(N3_features_all, "N3")
    features.save_raw_features(W_features_all, "W")
    features.save_raw_features(R_features_all, "R")


if __name__ == "__main__":
    main()
