import pandas as pd
from sklearn import metrics
from torch.autograd import Variable
import random
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from torch.utils.data import DataLoader, Dataset
from data_prepro import U_data

def segment_exchange(input_x, series_length=10, sub_seq_length=1, device=torch.device('cpu')):
    t = [i for i in range(series_length)]
    i = random.randint(0, series_length - 1 - 2 * sub_seq_length - 1)
    j = random.randint(sub_seq_length + 1, series_length - 1 - sub_seq_length)
    while j - i <= sub_seq_length:
        j = random.randint(sub_seq_length + 1, series_length - 1 - sub_seq_length)
    for v in range(sub_seq_length):
        t[i + v] = j + v
        t[j + v] = i + v
    idx = torch.LongTensor(t).to(device)
    r = input_x.index_select(dim=1, index=idx)
    return r


def segment_remove(input_x, series_length=10, sub_seq_length=1, device=torch.device('cpu')):
    t = [True for _ in range(series_length)]
    i = random.randint(0, series_length - 1 - sub_seq_length - 1)
    for v in range(sub_seq_length):
        t[i + v] = False
    b = torch.masked_select(input_x, torch.tensor(t, device=device))
    c = torch.zeros(input_x.shape[0], sub_seq_length).to(device)
    b = b.view(input_x.shape[0], -1)
    r = torch.cat([b, c], dim=1)
    return r


def segment_shuffle(input_x, series_length=10, sub_seq_length=2, device=torch.device('cpu')):
    t = [i for i in range(series_length)]
    i = random.randint(0, series_length - 1 - sub_seq_length - 1)
    for v in range(int(sub_seq_length * 0.5)):
        t[i + v] = t[i + sub_seq_length - 1 - v]
    idx = torch.LongTensor(t).to(device)
    r = input_x.index_select(dim=1, index=idx)
    return r


def segment_noise(input_x, var=0.1, device=torch.device('cpu')):
    r = torch.randn_like(input_x, requires_grad=False).to(device) * var + input_x
    return r


def data_argumentation(input_x, series_length=10, sub_seq_length=1, device=torch.device('cpu')):
    input_num = input_x.shape[0]
    r = []
    for i in range(input_num):
        t = random.random()
        if t < 0.1:
            r.append(segment_noise(input_x[i], device=device))
        elif t < 0.4:
            r.append(segment_shuffle(input_x[i], series_length, sub_seq_length, device))
        elif t < 0.7:
            r.append(segment_exchange(input_x[i], series_length, sub_seq_length, device))
        else:
            r.append(segment_remove(input_x[i], series_length, sub_seq_length, device))
    return torch.stack(r).to(device)


class CNNEncoder(nn.Module):
    def __init__(self, feature_dim, device):
        super(CNNEncoder, self).__init__()
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.conv_invariant1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=128, kernel_size=(7, ), padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(5,), padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv_invariant2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=(5, ), padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=(3, ), padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=128),
        )

        self.conv_specific1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=128, kernel_size=(7,), padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=(21,), stride=1, padding=10),
        )

        self.conv_specific2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(5,), padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=126)
        )

        self.avg_res = nn.AvgPool1d(kernel_size=126)

        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 128)

        self.fc1 = nn.Linear(128, feature_dim)
        self.fc2 = nn.Linear(128, feature_dim)

    def forward(self, x, with_argumentation=False, with_connection=True):
        x = x.transpose(1, 2)  # --> (n, 3, 128)
        if with_argumentation:
            x = data_argumentation(x.detach(), series_length=128, sub_seq_length=32, device=self.device)
            x = data_argumentation(x.detach(), series_length=128, sub_seq_length=32, device=self.device)
        iv_1 = self.conv_invariant1(x)  # (n, 3, 128) --> (n, 128, 128)
        sp_1 = self.conv_specific1(x)   # (n, 3, 128) --> (n, 128, 128)

        sp_res = self.avg_res(sp_1)

        if with_connection:
            iv_2 = iv_1 + torch.transpose(self.lin1(torch.transpose(sp_1, 1, 2)), 1, 2)
            sp_2 = sp_1 + torch.transpose(self.lin2(torch.transpose(iv_1, 1, 2)), 1, 2)
        else:
            iv_2 = iv_1
            sp_2 = sp_1

        iv_2 = self.conv_invariant2(iv_2)
        iv_2 = iv_2.flatten(1, 2)
        iv_feature = self.fc1(iv_2)

        sp_2 = self.conv_specific2(sp_2)
        sp_2 = sp_2 + sp_res
        sp_2 = sp_2.flatten(1, 2)
        sp_feature = self.fc2(sp_2)

        return iv_feature, sp_feature


class Classifier(nn.Module):
    def __init__(self, input_dim, y_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, y_dim)

    def forward(self, inputs):
        out = self.linear(inputs)
        return out


class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        feature = inputs.view(inputs.size(0), -1)
        validity = self.net(feature)
        return validity


class CATDModel(nn.Module):
    def __init__(self, x_dim, feature_dim, class_num, device=torch.device('cpu')):
        super(CATDModel, self).__init__()
        self.device = device
        self.x_dim = x_dim
        self.feature_dim = feature_dim
        self.class_num = class_num

        self.encoder = CNNEncoder(feature_dim, device)
        self.classifier = Classifier(feature_dim, self.class_num)

        self.discriminator = Discriminator(feature_dim + 1)
        # self.discriminator = Discriminator(feature_dim)
        self.discriminator2 = Discriminator(feature_dim)

        lr = 1e-3
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
        self.classifier_optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr)
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)
        self.discriminator2_optimizer = torch.optim.AdamW(self.discriminator2.parameters(), lr=lr)

        self.ae_loss = torch.nn.MSELoss(reduction='mean')
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.a = torch.eye(self.class_num, device=device)
        self.c = None

        self.gan_target_label = None
        self.gan_source_label = None
        self.target_label = None
        self.source_label = None

    def reset(self):
        self.encoder = CNNEncoder(self.feature_dim, device=self.device).to(self.device)
        self.classifier = Classifier(self.feature_dim, self.class_num).to(self.device)
        # self.discriminator = Discriminator(self.feature_dim).to(self.device)
        self.discriminator = Discriminator(self.feature_dim + 1).to(self.device)  # for gan
        self.discriminator2 = Discriminator(self.feature_dim).to(self.device)

        lr = 1e-3
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
        self.classifier_optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr)
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)
        self.discriminator2_optimizer = torch.optim.AdamW(self.discriminator2.parameters(), lr=lr)

    def init_center_c(self, train_loader):
        n_centers = torch.ones(self.class_num, device=self.device)
        c = torch.zeros((self.class_num, self.feature_dim), device=self.device)
        with torch.no_grad():
            for x_s, y_s in train_loader:
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)
                y_onehot = self.a[y_s]
                iv, sp = self.encoder(x_s)
                n_centers = n_centers + torch.sum(y_onehot, dim=0)
                # sum of all the features from the same class
                c = c + torch.matmul(y_onehot.T, iv)

        # average features of each class
        c = c / torch.unsqueeze(n_centers, 1)
        self.c = c.clone().detach()

    def domain_label(self, batch_size=64):
        self.gan_target_label = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        self.gan_source_label = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

        self.target_label = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
        self.source_label = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)

    def train_step(self, x_source, y_source, x_target, with_argumentation=True, with_kl_loss=True, change_center=False, with_interaction=True, kl_loss_weight=0.0,
                   gan_loss_weight=1.0):
        iv_s, sp_s = self.encoder(x_source, with_argumentation=False)
        iv_t, sp_t = self.encoder(x_target, with_argumentation=False)
        if change_center:
            ce_loss = self.ce_loss(self.classifier(iv_s), y_source)

            domain_cls_loss = self.bce_loss(self.discriminator2(sp_t), self.target_label) + \
                                self.bce_loss(self.discriminator2(sp_s), self.source_label)
            if with_argumentation:
                _, sp_s2 = self.encoder(x_source, with_argumentation=True)
                _, sp_t2 = self.encoder(x_target, with_argumentation=True)
                domain_cls_loss2 = self.bce_loss(self.discriminator2(sp_s2), self.source_label) + \
                                     self.bce_loss(self.discriminator2(sp_t2), self.target_label)
            else:
                domain_cls_loss2 = 0.0
            total_loss = ce_loss + (domain_cls_loss2 + domain_cls_loss) * 1.0

            self.encoder_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            self.discriminator2_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.discriminator2.parameters(), max_norm=0.5)
            self.encoder_optimizer.step()
            self.classifier_optimizer.step()
            self.discriminator2_optimizer.step()

            return total_loss.item()

        y_center = self.c[y_source].detach()

        if with_kl_loss:
            kl_loss = self.ae_loss(iv_s, y_center)
        else:
            kl_loss = 0

        ce_loss = self.ce_loss(self.classifier(iv_s), y_source)

        ivd_s = []
        ivd_t = []
        for i in range(self.class_num):
            ivd_s.append(torch.norm(iv_t - self.c[i].detach(), dim=1, keepdim=True))
            ivd_t.append(torch.norm(iv_s - self.c[i].detach(), dim=1, keepdim=True))
        ivd_s = torch.cat(ivd_s, dim=1)
        ivd_t = torch.cat(ivd_t, dim=1)
        ivd_s = torch.min(ivd_s, dim=1, keepdim=True).values / self.feature_dim
        ivd_t = torch.min(ivd_t, dim=1, keepdim=True).values / self.feature_dim

        domain_cls_loss = self.bce_loss(self.discriminator2(sp_t), self.target_label) + \
                            self.bce_loss(self.discriminator2(sp_s), self.source_label)

        if with_interaction:
            # gan_loss = self.bce_loss(self.discriminator(torch.add(iv_t, ivd_t)), self.gan_target_label) + \
            #            self.bce_loss(self.discriminator(torch.add(iv_s, ivd_s)), self.gan_source_label)
            gan_loss = self.bce_loss(self.discriminator(torch.cat([iv_t, ivd_t], dim=1)), self.gan_target_label) + \
                       self.bce_loss(self.discriminator(torch.cat([iv_s, ivd_s], dim=1)), self.gan_source_label)
        else:
            gan_loss = self.bce_loss(self.discriminator(iv_t), self.gan_target_label) + \
                       self.bce_loss(self.discriminator(iv_s), self.gan_source_label)

        if with_argumentation:
            _, sp_s2 = self.encoder(x_source, with_argumentation=True)
            _, sp_t2 = self.encoder(x_target, with_argumentation=True)
            domain_cls_loss2 = self.bce_loss(self.discriminator2(sp_t2), self.target_label) + \
                                 self.bce_loss(self.discriminator2(sp_s2), self.source_label)
        else:
            domain_cls_loss2 = 0.0

        total_loss = ce_loss * 1.0 + kl_loss * kl_loss_weight + gan_loss * gan_loss_weight + 1.0 * (domain_cls_loss + domain_cls_loss2)

        self.encoder_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()
        self.discriminator2_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.discriminator2.parameters(), max_norm=0.5)
        self.encoder_optimizer.step()
        self.classifier_optimizer.step()
        self.discriminator2_optimizer.step()
        if with_interaction:
            # dis_loss = self.bce_loss(self.discriminator(torch.add(iv_t.detach(), ivd_t.detach())), self.target_label) + \
            #            self.bce_loss(self.discriminator(torch.add(iv_s.detach(), ivd_s.detach())), self.source_label)
            dis_loss = self.bce_loss(self.discriminator(torch.cat([iv_t.detach(), ivd_t.detach()], dim=1)), self.target_label) + \
                       self.bce_loss(self.discriminator(torch.cat([iv_s.detach(), ivd_s.detach()], dim=1)), self.source_label)
        else:
            dis_loss = self.bce_loss(self.discriminator(iv_t.detach()), self.target_label) + \
                       self.bce_loss(self.discriminator(iv_s.detach()), self.source_label)
        self.discriminator_optimizer.zero_grad()
        dis_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
        self.discriminator_optimizer.step()
        return dis_loss.item()

    def compute_loss(self, x, label):
        with torch.no_grad():
            iv, sp = self.encoder(x)
            out = self.classifier(iv)
            _, pred = torch.max(out.data, 1)
            correct = pred.eq(label.data).cpu().sum()

        return self.ce_loss(out, label).item(), correct, label.size(0)

    def compute_loss2(self, x_real, domain):
        # domain dis
        with torch.no_grad():
            label = self.target_label if domain == 1 else self.source_label
            iv, sp = self.encoder(x_real, with_argumentation=True)
            pred_y = self.discriminator2(sp)
            pred = (pred_y < 0.5).int().view(x_real.size(0))
            correct = pred.eq(label).cpu().sum()
        return 0.0, correct, pred_y.shape[0]


def norm(x):
    x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-6)
    return x


if __name__ == '__main__':
    ball_loss_weight = 100
    discriminator_loss = 10
    device = torch.device('cuda:1')

    batch_size = 64
    epochs = 101

    # need change
    user_list = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30,
                 31, 32, 33, 34, 35, 36]

    for v1, v2 in zip([[11], [2], [26], [7], [16], [6], [6], [2], [7], [13],[12], [20], [16], [29], [28], [24], [16], [13], [9], [13]],
                      [[16], [4], [3], [25], [9], [23], [10], [9], [8], [7],[2], [25], [10], [14], [1], [15], [21], [29], [3], [11]]):
        aver_acc = []
        aver_recall = []
        aver_f1 = []
        for rp in range(5):

            data_train = U_data(v1)
            data_target = U_data(v2)
            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
            target_loader = DataLoader(data_target, batch_size=batch_size, shuffle=True, drop_last=True)
            valid_loader = DataLoader(data_target, batch_size=batch_size, shuffle=False, drop_last=True)

            result_total = []
            data_loader = train_loader
            # (self, x_dim, feature_dim, class_num, device=torch.device('cpu')):
            model = CATDModel(x_dim=9, feature_dim=8, class_num=6, device=device)
            model = model.to(device)
            model.domain_label(batch_size=batch_size)
            result, acc_all = [], []
            best_acc, best_recall, best_f1, best_iter = 0, 0, 0, 0
            best_loss = 100.0
            # change_center = False
            for epoch in range(epochs):
                correct, total, loss = 0, 0, 0
                correct2, total2, loss2 = 0, 0, 0
                model.train()
                pre_train_step = 20
                for (xs, ys), (xt, yt) in zip(data_loader, target_loader):
                    xs, xt, ys, yt = xs.to(device), xt.to(device), ys.to(device), yt.to(device)
                    model.train_step(xs, ys, xt, change_center=True if epoch <= pre_train_step else False,
                                     kl_loss_weight=ball_loss_weight if epoch > pre_train_step else 0.0,
                                     gan_loss_weight=discriminator_loss if epoch > pre_train_step else 0.1,
                                     with_argumentation=True, with_kl_loss=True, with_interaction=True)
                if epoch == pre_train_step:
                    model.eval()
                    model.init_center_c(data_loader)
                    model.reset()
                model.eval()
                true_all = []
                pred_all = []
                index_max = 0
                with torch.no_grad():
                    for index, data in enumerate(target_loader):
                        batch_X, batch_y = data
                        batch_X = torch.as_tensor(batch_X, device=device, dtype=torch.float32)
                        batch_y = torch.as_tensor(batch_y, device=device, dtype=torch.long)
                        iv, sp = model.encoder(batch_X)
                        out = model.classifier(iv)
                        _, pred = torch.max(out, 1)
                        pred_all.append(np.array(pred.cpu().detach()))
                        true_all.append(np.array(batch_y.cpu().detach()))
                pred_all = np.concatenate(pred_all)
                true_all = np.concatenate(true_all)
                acc_test = metrics.accuracy_score(y_true=true_all, y_pred=pred_all)
                recall_test = metrics.recall_score(y_true=true_all, y_pred=pred_all, average='macro',zero_division=0)
                f1_test = metrics.f1_score(y_true=true_all, y_pred=pred_all, average='macro',zero_division=0)
                if (acc_test > best_acc):
                    # if (acc_test > best_acc) & (i > 15):
                    best_acc = acc_test
                    best_recall = recall_test
                    best_f1 = f1_test
                    best_iter = epoch + 1
                if epoch % 20 == 0:
                    print(
                        'Epoch: [{}/{}], test acc: {:.5f}, best acc:{:.5f}, best recall:{:.5f}, best f1:{:.5f}, iter:{}'.format(
                            epoch + 1, epochs,
                            acc_test, best_acc, best_recall, best_f1, best_iter))
            aver_acc.append(best_acc)
            aver_recall.append(best_recall)
            aver_f1.append(best_f1)
        data = np.array([aver_acc, aver_recall, aver_f1])
        store = pd.DataFrame(data, index=['acc', 'recall', 'f1'])
        store.to_csv(f'result/ucihar/cadt_1_1/{v1}_{v2}.csv')
        print(v1, v2)
        print('acc')
        print(aver_acc)
        print('recall')
        print(aver_recall)
        print('f1')
        print(aver_f1)