import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from BiLSTM import BidirectionalLSTM
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import warnings

class DoubleEnsembleModel:
    def __init__(
            self,
            loss="mse",
            base_model="BiLSTM",
            num_models=3,
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
            hidden_dim=64,
            out_dim=0,
            batch_size=8
    ):
        self.num_models = num_models
        self.base_model = base_model
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        self.decay = 1
        if sample_ratios is None:
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if sub_weights is None:
            sub_weights = [1.0, 0.2, 0.2]
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
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.batch_size = batch_size

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
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)

        # calculate h_value for each sample
        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})

        # divide sample into bins
        # calculate weights for each bin
        h["bins"] = pd.cut(h["h_value"], self.bins_sr)
        h_avg = h.groupby("bins")["h_value"].mean()
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
                with torch.no_grad():
                    xb = torch.from_numpy(x_train_tmp.loc[:, self.sub_features[i_s]].values)
                    pre = submodel(xb.view(len(xb), 1, -1)).numpy().squeeze()
                    pre = pd.DataFrame(pre).max(1)
                    pred_sub = pd.Series(pre, index=x_train_tmp.index)
                pred += (
                    pd.Series(pred_sub, index=x_train_tmp.index)/ M
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
        df_train = pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train)], axis=1, keys=["feature", "label"])
        df_valid = pd.concat([pd.DataFrame(x_valid), pd.DataFrame(y_valid)], axis=1, keys=["feature", "label"])
        # initialize the sample weights
        N, F = x_train.shape
        weights = pd.Series(np.ones(N, dtype=float))
        # initialize the features
        features = df_train["feature"].columns
        pred_sub = pd.DataFrame(np.zeros((N, self.num_models), dtype=float), index=df_train["feature"].index)
        # train sub-models
        for k in range(self.num_models):
            #  print(k)
            self.sub_features.append(features)
            model_k, loss_curve = self.train_submodel(df_train, df_valid, weights, features)
            # print(loss_curve)
            self.ensemble.append(model_k)
            # no further sample re-weight and feature selection needed for the last sub-model
            if k + 1 == self.num_models:
                break

            pred_k = self.predict_sub(model_k, df_train, features)
            pred_sub.iloc[:, k] = pred_k
            pred_ensemble = pred_sub.iloc[:, : k + 1].mean(axis=1)
            loss_values = pd.Series(self.get_loss(df_train["label"].values.squeeze(), pred_ensemble.values))
            if self.enable_sr:
                weights = self.sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                features = self.feature_selection(df_train, loss_values)

    def train_submodel(self, df_train, df_valid, weights, features):
        x_train, y_train = df_train["feature"].loc[:, features], df_train["label"]
        x_valid, y_valid = df_valid["feature"].loc[:, features], df_valid["label"]
        C, _ = x_train.shape
        for i in range(C):
            x_train.loc[i, :] *= weights[i]
        train_ds = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
        valid_ds = TensorDataset(torch.from_numpy(x_valid.values), torch.from_numpy(y_valid.values))
        train_dl, valid_dl = self.get_data(train_ds, valid_ds)
        N, F = x_train.shape
        if self.base_model == "BiLSTM":
            model = BidirectionalLSTM(F, self.hidden_dim, self.out_dim)
            opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        else:
            raise ValueError("not implemented yet")
        loss_curve = pd.DataFrame(np.zeros((N, self.epochs)))
        for epoch in range(self.epochs):
            model.train()
            loss_t = pd.Series(np.zeros(N))
            top = 0
            for xb, yb in train_dl:
                loss, num = self.loss_batch(model, nn.MSELoss(), xb, yb, opt)
                loss_t.iloc[top:num] = loss
                top += num
            loss_curve.iloc[:, epoch] = loss_t

        return model, loss_curve

    def loss_batch(self, model, loss_func, xb, yb, opt=None):
        xb = xb.view(len(xb), 1, -1)
        yb = yb.view(len(yb), 1, -1)
        loss = loss_func(model(xb), yb)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss.item(), len(xb)

    def get_data(self, train_ds, valid_ds):
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=False),
            DataLoader(valid_ds, batch_size=self.batch_size),
        )

    def predict_sub(self, submodel, df_data, features):
        x_data, y_data = df_data["feature"].loc[:, features], df_data["label"]
        tx_data, ty_data = torch.from_numpy(x_data.values),\
                           torch.from_numpy(y_data.values)
        with torch.no_grad():
            pre = submodel(tx_data.view(len(tx_data), 1, -1)).numpy().squeeze()
            pre = pd.DataFrame(pre).max(1)
            pred_sub = pd.Series(pre, index=x_data.index)
        return pred_sub

    def predict(self, inputs):
        if self.ensemble is None:
            raise ValueError("model is not fitted yet!")
        x_test = pd.DataFrame(inputs)
        pred = []
        for i_sub, submodel in enumerate(self.ensemble):
            feat_sub = self.sub_features[i_sub]
            # print(feat_sub.shape)
            xb = torch.from_numpy(x_test.loc[:, feat_sub].values)
            pre = submodel(xb.view(len(xb), 1, -1)).numpy().squeeze()
            pre = pd.DataFrame(pre).idxmax(1).values
            # print(pre)
            pred.append(pre)
        return pred
