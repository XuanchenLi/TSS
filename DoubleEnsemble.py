import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class DoubleEnsembleModel:
    def __init__(
            self,
            loss="mse",
            base_model="CNN",
            num_models=6,
            enable_sr=True,
            enable_fs=True,
            alpha1=1.0,
            alpha2=1.0,
            bins_sr=10,
            bins_fs=5,
            decay=None,  # gamma
            sample_ratios=None,
            sub_weights=None,
            epochs=100,
    ):
        self.num_models = num_models
        self.base_model = base_model
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        self.decay = decay
        if sample_ratios is None:
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if sub_weights is None:
            sub_weights = [1.0, 0.2, 0.2, 0.2, 0.2, 0.2]
        if not len(sample_ratios) == bins_fs:
            raise ValueError("The length of sample_ratios should be equal to bins_fs.")
        self.sample_ratios = sample_ratios
        if not len(sub_weights) == num_models:
            raise ValueError("The length of sub_weights should be equal to num_models.")
        self.sub_weights = sub_weights
        self.epochs = epochs
        self.ensemble = []
        self.sub_features = []
        self.loss = loss

    def sample_reweight(self, loss_curve, loss_values, k_th):
        """
        The SR algorithm
        :param loss_curve: size N*T,
            element(i,t) refers to the error on the i-th sample after the t-th iteration in
            the training of the previous sub-model.
        :param loss_values: size N*1,
            element(i) refers to the error of the current ensemble on the i-th sample
        :param k_th: the index of the current sub_model, starting from 1
        :return: weights
        weights for all samples
        """
        # normalize loss_curve and loss_values via ranking for robustness
        loss_curve_norm = loss_curve.rank(axis=0, pct=True)
        loss_values_norm = -loss_values.rank(pct=True)

        # calculate C_start and C_end
        N, T = loss_curve_norm.shape
        part = np.maximum(1, int(T*0.1))
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mead(axis=1)

        # calculate h_value for each sample
        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})

        # divide sample into bins
        # calculate weights for each bin
        h["bins"] = pd.cut(h["h_values"], self.bins_sr)
        h_avg = h.groupby("bins")["h_values"].mean()
        weights = pd.Series(np.zeros(N, dtype=float))
        for i_b, b in enumerate(h_avg.index):
            weights[h["bins"] == b] = 1.0 / (self.decay**k_th * h_avg[i_b] + 0.1)
        return weights

    def feature_selection(self, df_train, loss_values):
        """
        The FS algorithm
        :param df_train: size N*F
        :param loss_values:size N*1,
            element(i) refers to the error of the current ensemble on the i-th sample
        :return:
        """
        x_train, y_train = df_train["feature"], df_train["label"]
        features = x_train.columns
        N, F = x_train.shape
        g = pd.DataFrame({"g_value": np.zeros(F, dtype=float)})
        M = len(self.ensemble)

        # shuffle columns and calculate g_value for each feature
        x_train_tmp = x_train.copy()
        for i_f, feat in enumerate(features):
            x_train_tmp.iloc[:, feat] = np.random.permutation(x_train_tmp.loc[:, feat].values)
            pred = pd.Series(np.zeros(N), index=x_train_tmp.index)
            for i_s, submodel in enumerate(self.ensemble):
                pred += (
                    pd.Series(
                        submodel.predict(x_train_tmp.loc[:, self.sub_features[i_s]].values), index=x_train_tmp.index
                     )
                    / M
                )
                loss_feat = self.get_loss(y_train.values.squeeze(), pred.values)
                g.loc[i_f, "g_value"] = np.mean(loss_feat - loss_values) / (np.std(loss_feat - loss_values) + 1e-7)
                x_train_tmp.loc[:, feat] = x_train.loc[:, feat].copy()
                # one column in train features is all-nan # if g['g_value'].isna().any()
                g["g_value"].replace(np.nan, 0, inplace=True)

                # divide features into bins_fs bins
                g["bins"] = pd.cut(g["g_value"], self.bins_fs)
                # randomly sample features from bins to construct the new features
                res_feat = []
                sorted_bins = sorted(g["bins"].unique(), reverse=True)
                for i_b, b in enumerate(sorted_bins):
                    b_feat = features[g["bins"] == b]
                    num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
                    res_feat = res_feat + np.random.choice(b_feat, size=num_feat, replace=False).tolist()
                return pd.Index(set(res_feat))

    def get_loss(self, label, pred):
        if self.loss == "mse":
            return (label - pred) ** 2
        else:
            raise ValueError("not implemented yet")

    def fit(self, x_train, y_train, x_valid, y_valid):
        df_train = pd.concat([x_train, y_train], axis=1, keys=["feature", "label"])
        df_valid = pd.concat([x_valid, y_valid], axis=1, keys=["feature", "label"])
        # initialize the sample weights
        N, F = x_train.shape
        weights = pd.Series(np.ones(N, dtype=float))
        # initialize the features
        features = x_train.columns
        pred_sub = pd.DataFrame(np.zeros((N, self.num_models), dtype=float), index=x_train.index)
        # train sub-models
        for k in range(self.num_models):
            self.sub_features.append(features)
            model_k = self.train_submodel(df_train, df_valid, weights, features)
            self.ensemble.append(model_k)
            # no further sample re-weight and feature selection needed for the last sub-model
            if k + 1 == self.num_models:
                break

            loss_curve = self.retrieve_loss_curve(model_k, df_train, features)
            pred_k = self.predict_sub(model_k, df_train, features)
            pred_sub.iloc[:, k] = pred_k
            pred_ensemble = pred_sub.iloc[:, : k + 1].mean(axis=1)
            loss_values = pd.Series(self.get_loss(y_train.values.squeeze(), pred_ensemble.values))

            if self.enable_sr:
                weights = self.sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                features = self.feature_selection(df_train, loss_values)
