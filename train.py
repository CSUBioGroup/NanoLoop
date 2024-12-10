import pandas as pd
import xgboost as xgb
from sklearn.utils import resample
from torch import nn
import torch.nn.functional as F
import os
import h5py
import numpy as np
import sklearn.metrics as metrics
import time
from six.moves import cPickle
import logging
import sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import joblib
import warnings
warnings.filterwarnings("ignore")

# Sequence module
class ReLUMixin():
    def _relu(self, inp):
        leaky_alpha = 1. / 5.5
        if self.leaky:
            return F.leaky_relu(inp, leaky_alpha)
        else:
            return F.relu(inp)

class NNClassifier(torch.nn.Module, ReLUMixin):
    def __init__(self, input_dim=206, leaky=True, guided=False, legacy=False):
        super(NNClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1024)
        if legacy:
            self.fc = torch.nn.Linear(1024,2)
        else:
            self.fc = torch.nn.Linear(1024, 1)
        self.softmax = torch.nn.Softmax()
        self.drop = torch.nn.Dropout(0.4)

        self.sigmoid = torch.nn.Sigmoid()

        self.leaky = leaky
        self.guided = guided


    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self._relu(x)
        x = self.drop(x)
        x = self.fc(x)
        x = torch.squeeze(x, 1)
        return x

class PartialDeepSeaModel(torch.nn.Module, ReLUMixin):
    def __init__(self, in_filters=4, use_weightsum=False, leaky=False,
                 use_bn=False, use_fc=False, guided=False, use_sigmoid=False):
        super(PartialDeepSeaModel, self).__init__()
        self.use_sigmoid = use_sigmoid
        self._set_filters()
        self._build_model(in_filters, use_weightsum, leaky, use_bn, use_fc, guided)


    def _set_filters(self):
        self.num_filters = [128, 256, 128]

    def _build_model(self, in_filters=4, use_weightsum=False, leaky=False, use_bn=False, use_fc=False, guided=False):
        self.conv1 = nn.Conv1d(in_filters, self.num_filters[0], 8)
        self.conv2 = nn.Conv1d(self.num_filters[0], self.num_filters[1], 8)
        self.conv3 = nn.Conv1d(self.num_filters[1], self.num_filters[2], 8)

        self.weighted_sum = torch.nn.Parameter(torch.randn((self.num_filters[2], 53)))
        torch.nn.init.kaiming_uniform_(self.weighted_sum)

        self.use_weightsum = use_weightsum
        self.use_fc = use_fc

        self.drop2 = nn.Dropout(0.5)

        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.guided = guided
        self.leaky = leaky
        self.use_bn = use_bn


    def forward(self, x):
        x = self._relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.drop2(x)
        x = self._relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.drop2(x)
        x = self._relu(self.conv3(x))
        x = self.drop2(x)

        x = torch.sum(torch.mul(x, self.weighted_sum), dim=2, keepdim=False)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        else:
            x = self.tanh(x)
        return x

    def get_weightsum(self, x):
        x = self._relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.drop2(x)
        x = self._relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.drop2(x)
        x = self._relu(self.conv3(x))
        x = self.drop2(x)
        x = torch.sum(torch.mul(x, self.weighted_sum), dim=2, keepdim=False)
        return x

    def get_conv_activations(self, x, level):
        assert level <= 3
        convs = [self.conv1, self.conv2, self.conv3]
        for i in range(level):
            x = self._relu(convs[i](x))
            if i < level - 1:
                x = self.maxpool(x)
        return x

def load_hdf5_data(fn):
    data = h5py.File(fn, 'r')
    left_data = data['left_data']
    right_data = data['right_data']
    left_edges = data['left_edges'][:]
    right_edges = data['right_edges'][:]
    labels = data['labels'][:]
    pairs = data['pairs'][:]
    methyl = data['methyl'][:]
    dists = [[np.log10(abs(p[5] / 5000 - p[2] / 5000 + p[4] / 5000 - p[1] / 5000) * 0.5) / np.log10(2000001 / 5000),] + list(p)[7:]
             for p in pairs]
    #print('dists:',dists)
    return data,left_data,right_data,left_edges,right_edges,labels,dists,methyl

def get_data_batch(left_data, left_edges, right_data, right_edges, dists, labels, start,
                   max_size=200, limit_to_one=False):
    #用于记录当前数据结束的edge
    end_idx = start + 1
    # 当前数据开始的edge
    i = start + 1
    #true
    if not limit_to_one:
        #遍历完当前数据的edge
        while i < len(left_edges) - 1:
            #如果左右两个anchor的分区数有一个大于300，则break
            if max(left_edges[i] - left_edges[start], right_edges[i] - right_edges[start]) > max_size:
                break
            i += 1
    #取i-1和end_idx中较大者为end_idx
    end_idx = max(end_idx, i - 1)
    #取出左anchor的序列
    curr_left = left_data[left_edges[start]:left_edges[end_idx]]
    #取出左anchor的分区
    curr_left_edges = left_edges[start:(end_idx + 1)]
    # 取出右anchor的序列
    curr_right = right_data[right_edges[start]:right_edges[end_idx]]
    # 取出右anchor的分区
    curr_right_edges = right_edges[start:(end_idx + 1)]

    #取出当前label和dist
    curr_labels = labels[start:end_idx]
    curr_dists = dists[start:end_idx]
    return end_idx, curr_left, curr_left_edges, curr_right, curr_right_edges, curr_dists, curr_labels

def compute_one_side(data, edges, model):
    x = torch.autograd.Variable(torch.from_numpy(data).float()).cuda()
    x_rc = torch.autograd.Variable(torch.from_numpy(np.array(data[:,::-1,::-1])).float()).cuda()
    edges = np.array(edges) - edges[0]
    combined = []
    curr_outputs = torch.cat((model(x), model(x_rc)), dim=1)
    for i in range(len(edges) - 1):
        combined.append(torch.max(curr_outputs[edges[i]:edges[i + 1], :], dim=0, keepdim=True)[0])
    out = torch.cat([x for x in combined], dim=0)
    return out

def compute_factor_output(left_data, left_edges, right_data, right_edges,
                          dists, labels, start, factor_model, max_size=200,
                          limit_to_one=False, legacy=False):
    (end, curr_left_data, curr_left_edges, curr_right_data,
     curr_right_edges, curr_dists, curr_labels) = get_data_batch(left_data,
                                                                 left_edges,
                                                                 right_data,
                                                                 right_edges,
                                                                 dists,
                                                                 labels,
                                                                 start,
                                                                 max_size=max_size,
                                                                 limit_to_one=limit_to_one)
    left_out = compute_one_side(curr_left_data, curr_left_edges, factor_model)
    right_out = compute_one_side(curr_right_data, curr_right_edges, factor_model)
    if legacy:
        curr_labels = torch.autograd.Variable(torch.from_numpy(curr_labels).long()).cuda()
    else:
        curr_labels = torch.autograd.Variable(torch.from_numpy(curr_labels).float()).cuda()
    curr_dists = torch.autograd.Variable(torch.from_numpy(np.array(curr_dists, dtype='float32'))).cuda()
    return end, left_out, right_out, curr_dists, curr_labels

def apply_classifier(classifier, left_out, right_out, curr_dists, input_grad=False, use_distance=True):

    if use_distance:
        if len(curr_dists.size()) == 1:
            curr_dists = curr_dists.view(-1, 1)
        combined1 = torch.cat((left_out, right_out, curr_dists), dim=1)
    else:
        combined1 = torch.cat((left_out, right_out), dim=1)

    if input_grad:
        combined1 = torch.nn.Parameter(combined1.data)
        out1 = classifier(combined1)
        return out1, combined1
    else:
        out1 = classifier(combined1)
        return out1

def predict(model, classifier, loss_fn,
            valid_left_data, valid_left_edges,
            valid_right_data, valid_right_edges,
            valid_dists, valid_labels, return_prob=False, use_distance=True,
            use_metrics=True, max_size=300, verbose=0, same=False, legacy=False):

    model.eval()
    classifier.eval()
    val_err = 0.
    val_samples = 0
    all_probs = []
    last_print = 0
    edge = 0
    # torch.no_grad()表示不会构建计算图，这样运算起来比较省空间，但也无法进行反向传播
    with torch.no_grad():
        while edge < len(valid_left_edges) - 1:
            end, left_out, right_out, curr_dists, curr_labels = compute_factor_output(valid_left_data, valid_left_edges,
                                                                                      valid_right_data, valid_right_edges,
                                                                                      valid_dists, valid_labels, edge,
                                                                                      model, max_size=max_size,legacy=legacy)
            if verbose > 0:
                logging.info(str(curr_dists.size()))
            curr_outputs = apply_classifier(classifier, left_out, right_out, curr_dists, use_distance=use_distance)

            #print('curr_outputs:',curr_outputs)
            #print('curr_labels:', curr_labels)
            #print('loss_fn type:', type(loss_fn))
            loss = loss_fn(curr_outputs, curr_labels)
            if legacy:
                if int(torch.__version__.split('.')[1]) > 2:
                    val_predictions = F.softmax(curr_outputs, dim=1).data.cpu().numpy()
                else:
                    val_predictions = F.softmax(curr_outputs).data.cpu().numpy()
                #print('curr_outputs:',curr_outputs)
                #print('val_predictions:',val_predictions)
                val_predictions = val_predictions[:,1]
            else:
                val_predictions = torch.sigmoid(curr_outputs).data.cpu().numpy()
            all_probs.append(val_predictions)
            val_err += loss.data.item() * (end - edge)

            val_samples += end - edge
            if verbose > 0 and end - last_print > 10000:
                logging.info(str(end))
            edge = end
    #print('all_probs:',all_probs)
    all_probs = np.concatenate(all_probs)

    if use_metrics:
        c_auprc = [metrics.average_precision_score(valid_labels, all_probs), ]
        c_roc = [metrics.roc_auc_score(valid_labels, all_probs), ]

        logging.info("  validation loss:\t\t{:.6f}".format(val_err / val_samples))
        logging.info("  auPRCs: {}".format("\t".join(map(str, c_auprc))))
        logging.info("  auROC: {}".format("\t".join(map(str, c_roc))))
        all_preds = np.zeros(all_probs.shape[0])
        all_preds[all_probs > 0.5] = 1
        logging.info("  f1: {}".format(str(metrics.f1_score(valid_labels, all_preds))))
        logging.info("  precision: {}".format(str(metrics.precision_score(valid_labels, all_preds))))
        logging.info("  recall: {}".format(str(metrics.recall_score(valid_labels, all_preds))))
        logging.info("  accuracy: {}".format(str(metrics.accuracy_score(valid_labels, all_preds))))
        logging.info("  ratio: {}".format(np.sum(valid_labels) / len(valid_labels)))
        one_prec = metrics.precision_score(valid_labels, np.ones(len(valid_labels)))
        precision, recall, _ = metrics.precision_recall_curve(valid_labels, all_probs, pos_label=1)
        fpr, tpr, _ = metrics.roc_curve(valid_labels, all_probs, pos_label=1)
    if return_prob:
        return val_err / val_samples, all_probs
    return val_err / val_samples

def train_estimator(train_data, train_label, val_data, val_label,
                    n_estimators=1000, threads=20, max_depth=6, verbose_eval=True):
    dtrain = xgb.DMatrix(train_data, label=train_label)
    dval = xgb.DMatrix(val_data, label=val_label)
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}
    params = {'max_depth': max_depth, 'objective': 'binary:logistic',
              'eta': 0.1, 'nthread': threads, 'eval_metric': ['aucpr', 'map', 'logloss'],'lambda': 1.0,'subsample': 0.8,'colsample_bytree':0.5}
    model = xgb.train(params, dtrain, n_estimators, evallist, early_stopping_rounds=40,
                      verbose_eval=verbose_eval, evals_result=evals_result)
    # 获取最佳性能
    best_epoch = max(range(len(evals_result['eval']['aucpr'])),
                     key=lambda i: evals_result['eval']['aucpr'][i])
    best_aucpr = evals_result['eval']['aucpr'][best_epoch]
    best_map = evals_result['eval']['map'][best_epoch]
    best_logloss = evals_result['eval']['logloss'][best_epoch]

    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Best AUC-PR: {best_aucpr}")
    print(f"Best MAP: {best_map}")
    print(f"Best Logloss: {best_logloss}")

    return model

# 重新调整正负样本比例
# 将数据按照 0 类和 1 类分开
def methyl_resample(train_methyl,train_labels, n):
    train_methyl_df = pd.DataFrame(train_methyl)
    train_labels_df = pd.DataFrame(train_labels)
    train_labels_df.columns = ['labels']
    df = pd.concat([train_labels_df, train_methyl_df], axis=1)
    df_class_0 = df[df['labels'] == 0]
    df_class_1 = df[df['labels'] == 1]

    # 对 0 类进行下采样，使其数量与 1 类相同
    df_class_0_downsampled = resample(df_class_0,
                                      replace=False,   # 不重复采样
                                      n_samples=len(df_class_1) * n,  # 与 1 类数量相同
                                      random_state=42)  # 保持结果可复现

    # 将下采样的 0 类数据和原始的 1 类数据合并
    df_balanced = pd.concat([df_class_0_downsampled, df_class_1])

    # 打乱新数据集的顺序
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(df_balanced['labels'].value_counts())

    return df_balanced

def extract_seq_feature(train_left_data, train_left_edges,train_right_data,train_right_edges,train_dists, train_labels,model):
    i = 0
    is_begin = True
    last_print = 0
    with torch.no_grad():
        while i < len(train_left_edges) - 1:
            #true参数表示用模型进行评估，而不是训练
            end, left_out, right_out, _, _ = compute_factor_output(train_left_data, train_left_edges,
                                                                   train_right_data,
                                                                   train_right_edges,
                                                                   train_dists, train_labels, i,
                                                                   factor_model=model, max_size=2000)
            if is_begin:
                train_left_all = left_out.data.cpu().numpy()
                train_right_all = right_out.data.cpu().numpy()
                is_begin = False
            else:
                train_left_all = np.vstack((train_left_all, left_out.data.cpu().numpy()))
                train_right_all = np.vstack((train_right_all, right_out.data.cpu().numpy()))
            if end - last_print > 5000:
                last_print = end
                print('generating input : %d / %d' % (end, len(train_labels)))
            i = end
    return train_left_all, train_right_all

def train(model_name,model_dir,train_list,valid_list,sigmoid = False, legacy = True, init_lr=0.0002,eps=1e-8,
          use_distance = False,verbose=0,same=False,epochs=40,interval=5000):
    model = PartialDeepSeaModel(4, use_weightsum=True, leaky=True, use_sigmoid=sigmoid)
    n_filters = model.num_filters[-1]*4
    classifier = NNClassifier(n_filters, legacy=legacy)

    model.cuda()
    classifier.cuda()

    (train_data, train_left_data, train_right_data,
     train_left_edges, train_right_edges,
     train_labels, train_dists, train_methyl) = train_list
    (valid_data, valid_left_data, valid_right_data,
     valid_left_edges, valid_right_edges,
     valid_labels, valid_dists, valid_methyl) = valid_list

    # 创建一个logger日志对象
    rootLogger = logging.getLogger()
    for handler in rootLogger.handlers:
        rootLogger.removeHandler(handler)
    # TimedRotatingFileHandler对象自定义日志级别
    rootLogger.setLevel(logging.INFO)
    # TimedRotatingFileHandler对象自定义日志格式
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler('logs/' + model_name + ".log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logging.info('learning rate: %f, eps: %f' % (init_lr, eps))


    weights = torch.FloatTensor([1, 1]).cuda()

    logging.info(str(weights))

    #交叉熵损失
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(list(classifier.parameters()) + list(model.parameters()),lr=init_lr, eps=eps,weight_decay=init_lr * 0.1)
    best_val_loss = predict(model, classifier, loss_fn,
                            valid_left_data, valid_left_edges,
                            valid_right_data, valid_right_edges,
                            valid_dists, valid_labels, return_prob=False,
                            use_distance=use_distance, verbose=verbose, same=same,
                            legacy=legacy)

    last_update = 0
    for epoch in range(0, epochs):
        start_time = time.time()
        i = 0
        train_loss = 0.
        num_samples = 0

        model.train()
        classifier.train()

        last_print = 0
        curr_loss = 0.
        curr_pos = 0
        while i < len(train_labels):
            # 获取 classifier 中 fc1 层的初始权重
            param_before = classifier.fc1.weight.clone().detach().cpu().numpy()

            end, left_out, right_out, curr_dists, curr_labels = compute_factor_output(
                train_left_data, train_left_edges, train_right_data, train_right_edges,
                train_dists, train_labels, i, model, legacy=legacy
            )

            if verbose > 0:
                logging.info(str(curr_dists.size()))

            curr_outputs = apply_classifier(classifier, left_out, right_out, curr_dists, use_distance=use_distance)

            loss = loss_fn(curr_outputs, curr_labels)

            optimizer.zero_grad()
            loss.backward()

            # 打印优化前的参数
            #print(f"优化前的参数: \n{param_before}")

            optimizer.step()

            # 获取更新后的参数
            param_after = classifier.fc1.weight.clone().detach().cpu().numpy()

            # 打印优化后的参数
            #print(f"优化后的参数: \n{param_after}")

            # 比较差异
            #print(f"参数变化量: \n{param_after - param_before}")

            num_samples += end - i
            curr_loss += loss.data.item() * (end - i)
            train_loss += loss.data.item() * (end - i)
            curr_pos += torch.sum(curr_labels).data.item()
            i = end
            if num_samples < 1000 or num_samples - last_print > interval:
                logging.info("%d  %f  %f  %f  %f", i, time.time() - start_time,
                             train_loss / num_samples, curr_loss / (num_samples - last_print),
                             curr_pos * 1.0 / (num_samples - last_print))
                curr_pos = 0
                curr_loss = 0
                last_print = num_samples

        # for epoch in range(0, epochs):
    #     start_time = time.time()
    #     i = 0
    #     train_loss = 0.
    #     num_samples = 0
    #
    #     model.train()
    #     classifier.train()
    #
    #     last_print = 0
    #     curr_loss = 0.
    #     curr_pos = 0
    #     while i < len(train_labels):
    #         end, left_out, right_out, curr_dists, curr_labels = compute_factor_output(train_left_data,train_left_edges,train_right_data,train_right_edges,
    #                                                                                   train_dists, train_labels, i, model, legacy=legacy)
    #
    #
    #         if verbose > 0:
    #             logging.info(str(curr_dists.size()))
    #         curr_outputs = apply_classifier(classifier, left_out, right_out, curr_dists, use_distance=use_distance)
    #
    #         loss = loss_fn(curr_outputs, curr_labels)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         num_samples += end - i
    #         curr_loss += loss.data.item() * (end - i)
    #         train_loss += loss.data.item() * (end - i)
    #         curr_pos += torch.sum(curr_labels).data.item()
    #         i = end
    #         if num_samples < 1000 or num_samples - last_print > interval:
    #             logging.info("%d  %f  %f  %f  %f", i, time.time() - start_time,
    #                          train_loss / num_samples, curr_loss / (num_samples - last_print),
    #                          curr_pos*1.0 / (num_samples - last_print))
    #             curr_pos = 0
    #             curr_loss = 0
    #             last_print = num_samples

        logging.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, epochs, time.time() - start_time))
        logging.info("Train loss: %f", train_loss / num_samples)

        val_err = predict(model, classifier, loss_fn,
                          valid_left_data, valid_left_edges,
                          valid_right_data, valid_right_edges,
                          valid_dists, valid_labels, return_prob=False,
                          use_distance=use_distance, verbose=verbose,
                          same=same, legacy=legacy)

        if val_err < best_val_loss or epoch == 0:
            best_val_loss = val_err
            last_update = epoch
            logging.info("current best val: %f", best_val_loss)
            torch.save(model.state_dict(),
                       "{}/{}_sequence.model.pt".format(model_dir, model_name),
                       pickle_protocol=cPickle.HIGHEST_PROTOCOL)
            # #torch.save(classifier.state_dict(),
            #            #"{}/{}.classifier.pt".format(model_dir, model_name),
            #            pickle_protocol=cPickle.HIGHEST_PROTOCOL)
        if epoch - last_update >= 10:
            break
    fileHandler.close()
    consoleHandler.close()
    rootLogger.removeHandler(fileHandler)
    rootLogger.removeHandler(consoleHandler)
    return model

def get_args():
    parser = argparse.ArgumentParser(description="Train distance matched models")
    parser.add_argument('--data_pre', help='The name of the data')
    parser.add_argument('--model_name', help="The prefix of the model.")
    parser.add_argument('--model_dir', help='Directory for storing the models.')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs for training. Default: 40')
    return parser.parse_args()

def main():
    args = get_args()
    # Sequence module
    print('Training sequence model...')
    model_name = args.model_name
    model_dir = args.model_dir
    data_pre = args.data_pre
    epochs = args.epochs

    (train_data, train_left_data, train_right_data,
     train_left_edges, train_right_edges,
     train_labels, train_dists, train_methyl) = load_hdf5_data("%s_train.hdf5" % data_pre)

    (valid_data, valid_left_data, valid_right_data,
     valid_left_edges, valid_right_edges,
     valid_labels, valid_dists, valid_methyl) = load_hdf5_data("%s_valid.hdf5" % data_pre)

    train_list = [train_data, train_left_data, train_right_data,
                 train_left_edges, train_right_edges,
                 train_labels, train_dists, train_methyl]

    valid_list = [valid_data, valid_left_data, valid_right_data,
                  valid_left_edges, valid_right_edges,
                  valid_labels, valid_dists, valid_methyl]

    seq_model = train(model_name,model_dir,train_list,valid_list,epochs = epochs)
    
    # Methylation module
    print('Training methylation model...')
    #train_data_resampled = methyl_resample(train_methyl,train_labels, 1)
    #valid_data_resampled = methyl_resample(valid_methyl,valid_labels, 1)
    train_data_resampled = pd.DataFrame(train_methyl)
    valid_data_resampled = pd.DataFrame(valid_methyl)
    methyl_model = train_estimator(train_data_resampled, train_labels, valid_data_resampled, valid_labels, max_depth=4, threads=20, verbose_eval=True)
    joblib.dump(methyl_model, os.path.join(model_dir, f"{model_name}_methylation.gbt.pkl"))

    # Clf
    # 加载序列模型
    # seq_model_path= "{}/{}_sequence.model.pt".format(model_dir, model_name)
    # seq_model = PartialDeepSeaModel(4, use_weightsum=True, leaky=True, use_sigmoid=False)
    # seq_model.load_state_dict(torch.load(seq_model_path))
    print('Training classifier...')
    # 提取序列特征
    seq_model.cuda()
    seq_model.eval()
    train_left_all, train_right_all = extract_seq_feature(train_left_data, train_left_edges,train_right_data,train_right_edges,train_dists, train_labels,seq_model)
    valid_left_all, valid_right_all = extract_seq_feature(valid_left_data, valid_left_edges,valid_right_data,valid_right_edges,valid_dists, valid_labels,seq_model)

    # 提取甲基化特征
    train_methyl_probs = methyl_model.predict(xgb.DMatrix(pd.DataFrame(train_methyl)))
    valid_methyl_probs = methyl_model.predict(xgb.DMatrix(pd.DataFrame(valid_methyl)))
    ## 将序列特征拼接甲基化特征
    train_for_classifier = np.concatenate([train_left_all, train_right_all, np.array(train_dists).reshape(-1, 1),train_methyl,np.array(train_methyl_probs).reshape(-1, 1)], axis=1)
    train_for_classifier = pd.DataFrame(train_for_classifier)
    valid_for_classifier = np.concatenate([valid_left_all, valid_right_all, np.array(valid_dists).reshape(-1, 1),valid_methyl,np.array(valid_methyl_probs).reshape(-1, 1)], axis=1)
    valid_for_classifier = pd.DataFrame(valid_for_classifier)
    clf_model = train_estimator(train_for_classifier, train_labels, valid_for_classifier, valid_labels, max_depth=4, threads=20, verbose_eval=True)
    joblib.dump(clf_model, os.path.join(model_dir, f"{model_name}_clf.gbt.pkl"))

if __name__ == "__main__":
    main()