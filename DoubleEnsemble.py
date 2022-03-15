import numpy as np
import pandas as pd


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

    