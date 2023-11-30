import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import sklearn.linear_model as linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
# from xgboost import XGBRegressor
from sklearn import svm
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding


def cal_regression_metrics(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    # r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return mae, mse, r2


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3) # 0.3-17.6
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc2(x)
        return x

# class Net(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(Net, self).__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.3) # 0.3-17.6
#         )
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.fc2(x)
#         return x


def plot_points_line(x, y, x_label, y_label, title, save_path):
    plt.clf()
    plt.subplots(figsize=(10, 10))
    plt.scatter(x, y, s=10)
    plt.plot([0.5, 2.5], [0.5, 2.5], color="red")
    plt.xlim(1, 2.5)
    plt.ylim(1, 2.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def plot_training(epoch_list, train_r2_list, valid_r2_list, test_r2_list, title):
    plt.clf()
    plt.subplots(figsize=(10, 5))
    plt.plot(epoch_list, train_r2_list, label="train")
    plt.plot(epoch_list, valid_r2_list, label="valid")
    plt.plot(epoch_list, test_r2_list, label="test")
    plt.xlabel("epoch")
    plt.ylabel("r2")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("r2.jpg")
    plt.close()

class PipeLine:

    def __init__(self, args):
        self.args = args

    def run(self):
        self.init_trainer()
        self.load_data(self.args.train_situations, self.args.test_situations, self.args.dim, self.args.split_ratio, self.args.th_c_range)
        self.train()

    def init_trainer(self):
        self.model = Net(self.args.input_dim, self.args.hidden_dim, self.args.output_dim)
        self.model.to(self.args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.loss_dict = {
                            "f1": F.mse_loss,
                            "f2": F.l1_loss,
                            "combine": lambda x, y: self.args.mse_weight * F.mse_loss(x, y) + F.l1_loss(x, y)
                        }

    def train(self):
        self.model.train()
        self.min_test_mae = 1000
        self.min_train_mae = 1000
        epoch_list = []
        valid_r2_list = []
        train_r2_list = []
        test_r2_list = []
        for epoch in range(self.args.epoch):
            tr_loss, tr_mae, tr_mse, tr_r2, output_list, target_list  = self.train_one_epoch(epoch)
            tr_rmse = np.sqrt(tr_mse)
            if epoch % 100 == 0:
                va_loss, va_mae, va_mse, va_r2 = self.valid_one_epoch(epoch, "valid")
                ts_loss, ts_mae, ts_mse, ts_r2 = self.valid_one_epoch(epoch, "test")

                if self.args.plot_training:
                    epoch_list.append(epoch)
                    valid_r2_list.append(va_r2)
                    train_r2_list.append(tr_r2)
                    test_r2_list.append(ts_r2)
                    title = "Max Train MAE: {:.3f} [ trained on {} ]".format(tr_mae, self.args.train_situations)
                    plot_training(epoch_list, train_r2_list, valid_r2_list, test_r2_list, title)
                    if tr_mae < self.min_train_mae:
                        self.min_train_mae = tr_mae
                        f = open("train_true_vs_predicted.txt", "w")
                        f.write("True\tPredicted\n")
                        for i in range(len(target_list)):
                            f.write("{}\t{}\n".format(target_list[i], output_list[i]))
                        f.close()
                        plot_points_line(target_list, output_list, "True", "Predicted", title, "train_true_vs_predicted.jpg")

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        output_list = []
        target_list = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_dict[self.args.loss_type](output.view(-1), target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            output_list.append(output.cpu().detach().numpy())
            target_list.append(target.cpu().detach().numpy())
        outputs = np.concatenate(output_list, axis=0).reshape(-1)
        targets = np.concatenate(target_list, axis=0).reshape(-1)
        mae, mse, r2 = cal_regression_metrics(targets, outputs)
        # print("Epoch {} train loss: {}".format(epoch, train_loss / len(self.train_loader)))
        return train_loss / len(self.train_loader), mae, mse, r2, outputs, targets
    
    def valid_one_epoch(self, epoch, mode="valid"):
        self.model.eval()
        valid_loss = 0.0
        output_list = []
        target_list = []
        for batch_idx, (data, target) in enumerate(self.valid_loader if mode == "valid" else self.test_loader):
            output = self.model(data)
            loss = self.loss_dict[self.args.loss_type](output.view(-1), target)
            valid_loss += loss.item()

            output_list.append(output.cpu().detach().numpy())
            target_list.append(target.cpu().detach().numpy())

        output_list = np.concatenate(output_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)

        outputs = output_list.reshape(-1)
        targets = target_list.reshape(-1)

        mae, mse, r2 = cal_regression_metrics(targets, outputs)
        print("Epoch {} {} loss: {}".format(epoch, mode, valid_loss / len(self.valid_loader)))
        print("Epoch {} {} mae: {}, mse: {}, r2: {}".format(epoch, mode, mae, mse, r2))

        if self.args.plot_test and mode == "test" and mae < self.min_test_mae:
            self.min_test_mae = mae
            title = "Max Test MAE: {:.3f} [ trained on {} and test on {} ]".format(mae, self.args.train_situations, self.args.test_situations)
            plot_points_line(targets, outputs, "True", "Predicted", title, "test_true_vs_predicted.jpg")
            f = open("test_true_vs_predicted.txt", "w")
            f.write("True\tPredicted\n")
            for i in range(len(target_list)):
                f.write("{}\t{}\n".format(target_list[i], output_list[i][0]))
            f.close()
        return valid_loss / len(self.valid_loader), mae, mse, r2

    def load_data(self, train_situations, test_situations, dim=-1, split_ratio=0.9, th_c_range=[-1, -1]):
        train_input_features_list = []
        train_labels_list = []
        for train_situation in train_situations:
            train_input_file = "../features/train/soap/6_5_5_{}_features.npy".format(train_situation)
            train_label_file = "../features/train/soap/6_5_5_{}_labels.npy".format(train_situation)
            train_input_features = np.load(train_input_file)
            train_labels = np.load(train_label_file)

            train_input_features_list.append(train_input_features)
            train_labels_list.append(train_labels)

        test_input_features_list = []
        test_labels_list = []
        for test_situation in test_situations:
            test_input_file = "../features/test/soap/6_5_5_{}_features.npy".format(test_situation)
            test_label_file = "../features/test/soap/6_5_5_{}_labels.npy".format(test_situation)
            test_input_features = np.load(test_input_file)
            test_labels = np.load(test_label_file)

            test_input_features_list.append(test_input_features)
            test_labels_list.append(test_labels)
        
        train_input_features = np.concatenate(train_input_features_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)

        test_input_features = np.concatenate(test_input_features_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)

        if th_c_range[0] != -1:
            print("here")
            index1 = train_labels > 0
            index2 = train_labels < 4 
            index = index1 & index2
        else:
            index = train_labels > 0

        train_input_features = train_input_features[index]  # select data on this temp
        train_labels = train_labels[index]  # select label on this temp

        train_input_features = train_input_features[:, :-1]  # remove situation temperature attribute
        test_input_features = test_input_features[:, :-1]
        train_input_temp = train_input_features[:, -1] / 1000  # get situation temperature attribute
        test_input_temp = test_input_features[:, -1] / 1000

        # normalize
        # train_labels = (train_labels - min(train_labels)) / (max(train_labels) - min(train_labels))
        # test_labels = (test_labels - min(test_labels)) / (max(test_labels) - min(test_labels))

        # shuffle
        np.random.seed(0)
        shuffle_idx = np.random.permutation(np.arange(len(train_input_features)))

        split_num = int(split_ratio * len(train_input_features))
        # reduce the dimension
        if dim != -1:
            decomp = PCA(dim)
            # decomp = KernelPCA(dim, kernel="rbf", gamma=0.1)
            train_input_features = decomp.fit_transform(train_input_features)
            test_input_features = decomp.transform(test_input_features)

        train_input_features = np.concatenate([train_input_features, train_input_temp.reshape(-1, 1)], axis=1)
        test_input_features = np.concatenate([test_input_features, test_input_temp.reshape(-1, 1)], axis=1)

        # split train to train and valid
        train_input_features = train_input_features[shuffle_idx]
        train_labels = train_labels[shuffle_idx]

        valid_input_features = train_input_features[split_num:, :]
        train_input_features = train_input_features[:split_num, :]

        valid_labels = train_labels[split_num:]
        train_labels = train_labels[:split_num]

        train_input_features = torch.from_numpy(train_input_features).float().to(self.args.device)
        train_labels = torch.from_numpy(train_labels).float().to(self.args.device)
        valid_input_features = torch.from_numpy(valid_input_features).float().to(self.args.device)
        valid_labels = torch.from_numpy(valid_labels).float().to(self.args.device)
        test_input_features = torch.from_numpy(test_input_features).float().to(self.args.device)
        test_labels = torch.from_numpy(test_labels).float().to(self.args.device)

        print("train_input_features.shape: {}".format(train_input_features.shape))
        print("test_input_features.shape: {}".format(test_input_features.shape))

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_input_features, train_labels),
            batch_size=self.args.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(valid_input_features, valid_labels),
            batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_input_features, test_labels),
            batch_size=self.args.batch_size, shuffle=True)


def main(args):
    pipeline = PipeLine(args)
    pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=11)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--loss_type", type=str, default="combine")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--mse_weight", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=100000)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--train_situations", type=list, default=["77K", "150K", "300K", "700K", "900K", "1300K"])
    parser.add_argument("--test_situations", type=list, default=["77K", "150K", "500K"])
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--th_c_range", type=list, default=[-1, -1])
    parser.add_argument("--plot_training", type=bool, default=True)
    parser.add_argument("--plot_test", type=bool, default=True)
    args = parser.parse_args()
    main(args)