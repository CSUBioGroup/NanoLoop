import pandas as pd
import xgboost as xgb
from torch import nn
import torch.nn.functional as F
import os
import h5py
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import joblib
import warnings
import argparse
warnings.filterwarnings("ignore")

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
    return data,left_data,right_data,left_edges,right_edges,labels,dists,methyl,pairs

# model
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

def get_data_batch(left_data, left_edges, right_data, right_edges, dists, labels, start,
                   max_size=300, limit_to_one=False):
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
                          dists, labels, start, evaluation, factor_model, max_size=100,
                          limit_to_one=False, same=False, legacy=False):
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
                                                                   True, factor_model=model, max_size=2000)
            if is_begin:
                train_left_all = left_out.data.cpu().numpy()
                train_right_all = right_out.data.cpu().numpy()
                is_begin = False
            else:
                train_left_all = np.vstack((train_left_all, left_out.data.cpu().numpy()))
                train_right_all = np.vstack((train_right_all, right_out.data.cpu().numpy()))
            if end - last_print > 2000:
                last_print = end
                print('generating input : %d / %d' % (end, len(train_labels)))
            i = end
    return train_left_all, train_right_all

def get_args():
    parser = argparse.ArgumentParser(description="Predict probability of loops.")
    parser.add_argument('--data_pre', help='The name of the data')
    parser.add_argument('--model_name', help="The prefix of the model.")
    parser.add_argument('--model_dir', help='Directory for storing the models.')
    parser.add_argument('--result_path', help='Directory for storing the result.')
    parser.add_argument('--save_feature', type=bool, default=False,help='Save feature or not.')
    return parser.parse_args()

def main():
    args = get_args()
    data_pre = args.data_pre
    #data_pre = 'out_dir/HG004_5k_noBalance_singleton_tf_with_random_neg_seq_data_length_filtered'
    (test_data, test_left_data, test_right_data,test_left_edges, test_right_edges,
     test_labels, test_dists, test_methyl,test_pairs) = load_hdf5_data("%s_test.hdf5" % data_pre)

    # 读取模型
    model_dir = args.model_dir
    model_name = args.model_name
    seq_model_path= "{}/{}_sequence.model.pt".format(model_dir, model_name)
    seq_model = PartialDeepSeaModel(4, use_weightsum=True, leaky=True, use_sigmoid=False)
    seq_model.load_state_dict(torch.load(seq_model_path))
    seq_model.cuda()
    seq_model.eval()

    methyl_model_path= "{}/{}_methylation.gbt.pkl".format(model_dir, model_name)
    methyl_model = joblib.load(methyl_model_path)

    clf_path= "{}/{}_clf.gbt.pkl".format(model_dir, model_name)
    clf_model = joblib.load(clf_path)

    # 提取序列特征
    test_left_all, test_right_all = extract_seq_feature(test_left_data, test_left_edges,test_right_data,test_right_edges,test_dists, test_labels,seq_model)
    # 提取甲基化特征
    test_methyl_probs = methyl_model.predict(xgb.DMatrix(pd.DataFrame(test_methyl)))
    # 特征拼接
    test_for_classifier = np.concatenate([test_left_all, test_right_all, np.array(test_dists).reshape(-1, 1),test_methyl,np.array(test_methyl_probs).reshape(-1, 1)], axis=1)
    # 特征转换成df
    test_for_classifier = pd.DataFrame(test_for_classifier)
    # 如果要保存特征
    save_feature = args.save_feature
    if save_feature:
        test_for_classifier.to_csv(model_dir+'/'+model_name+'_test_feature.csv',index=None,header=None,sep='\t')
    # 使用clf进行预测
    # 使用clf进行预测
    test_probs = clf_model.predict(xgb.DMatrix(test_for_classifier))
    result = pd.concat([pd.DataFrame(test_pairs),pd.DataFrame(test_probs)], axis=1)
    result_path = args.result_path
    result.to_csv(result_path,index=None,header=None,sep='\t')

if __name__ == "__main__":
    main()