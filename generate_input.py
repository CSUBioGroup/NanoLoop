import pylab as pl
import bisect
from Bio import SeqIO
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import h5py
import argparse
from functools import partial
import os
import warnings

warnings.filterwarnings('ignore')

# 读取assembly文件
def load_assembly(gfile):
    # 将1-23+x号染色体的所有碱基提取出来，放入hg19的列表中，例如hg19=['aacc```','cctt```',```]
    print('Loading assembly data...')
    global assembly
    assembly = ['' for i in range(24)]
    for o in SeqIO.parse(open(gfile), 'fasta'):
        if o.name.replace('chr', '') not in list(map(str, range(1,23))) + ['X']:
            continue
        if o.name == 'chrX':
            temp_key = 23
        else:
            temp_key = int(o.name.replace('chr', ''))
        assembly[temp_key] = str(o.seq.upper())
        print(o.name,':',len(assembly[temp_key]))


CHANNELS_FIRST = "channels_first"
CHANNELS_LAST = "channels_last"


def load_pairs_as_dict(files, min_length=1000, min_dist=5000, max_dist=2000000, max_length=None):
    scores = {}
    t_dists = {}
    for inter_file in files:
        with open(inter_file, 'r') as f:
            for r in f:
                tokens = r.strip().split()
                if len(tokens) < 7:
                    tokens.append(0)
                for i in [1, 2, 4, 5, 6]:
                    try:
                        tokens[i] = int(tokens[i])
                    except:
                        tokens[i] = int(float(tokens[i]))
                if tokens[0] == tokens[3] and tokens[1] > tokens[4]:
                    temp = tokens[1]
                    tokens[1] = tokens[4]
                    tokens[4] = temp
                    temp = tokens[2]
                    tokens[2] = tokens[5]
                    tokens[5] = temp
                if max_length is not None and (tokens[2]-tokens[1] > max_length or tokens[5] - tokens[4] > max_length):
                    continue
                if min_length > 0:
                    for i,j in zip([1,4],[2,5]):
                        if tokens[j] - tokens[i] < min_length:
                            diff = min_length - (tokens[j]-tokens[i])
                            half_diff = diff // 2
                            tokens[i] -= half_diff
                            tokens[j] += diff - half_diff
                curr_dist = 0.5 * (tokens[4] + tokens[5] - tokens[1] - tokens[2])
                if min_dist <= curr_dist <= max_dist:
                    scores[tuple(tokens[:6])] = tokens[6]
                    if tokens[0] not in t_dists:
                        t_dists[tokens[0]] = []
                    t_dists[tokens[0]].append(curr_dist)

    return scores, t_dists


#将碱基序列转换成四维向量
def get_seq_matrix(seq, seq_len: int, data_format: str, one_d: bool, rc=False):
    channels = 4
    # seq长度*4列
    mat = np.zeros((seq_len, channels), dtype="float32")

    for i, a in enumerate(seq):
        idx = i
        if idx >= seq_len:
            break
        a = a.lower()

        if a == 'a':
            mat[idx, 0] = 1
        elif a == 'g':
            mat[idx, 1] = 1
        elif a == 'c':
            mat[idx, 2] = 1
        elif a == 't':
            mat[idx, 3] = 1
        else:
            mat[idx, 0:4] = 0

    if rc:
        #行全部倒序，列也全部倒序
        mat = mat[::-1, ::-1]
    # true
    if not one_d:
        # 压缩成1维向量
        mat = mat.reshape((1, seq_len, channels))
    # true
    if data_format == CHANNELS_FIRST:
        axes_order = [len(mat.shape)-1,] + [i for i in range(len(mat.shape)-1)]
        #做一个转置，mat变成4列若干行
        mat = mat.transpose(axes_order)

    return mat


def get_annotation_matrix(chrom, start, end, annotations, seq_len, data_format, rc=False):
    out = np.zeros((len(annotations), seq_len), dtype="float32")
    end = min(start + seq_len, end)
    for i, anno in enumerate(annotations):
        if chrom in anno[0]:
            recs = anno[0][chrom]
            rec_ends = anno[1][chrom]
            si = bisect.bisect(rec_ends, start)
            while si < len(recs) and recs[si][1] < end:
                r = recs[si]
                if end > r[1] and start < r[2]:
                    r_start = max(r[1], start) - start
                    r_end = min(r[2], end) - start
                    out[i][r_start:r_end] = r[6]
                si += 1
    if rc:
        out = out[:, ::-1]
    if data_format != CHANNELS_FIRST:
        out = out.transpose()
    return out

# 如果anchor长度<min_size，则需要扩展长度，最后返回从start到end的碱基seq
def _get_sequence(chrom, start, end, min_size=1000, crispred=None):
    # assumes the CRISPRed regions were not overlapping
    # assumes the CRISPRed regions were sorted
    #is None
    if crispred is not None:
        #print('crispred is not None')
        seq = ''
        curr_start = start
        for cc, cs, ce in crispred:
            # overlapping
            #print('check', chrom, start, end, cc, cs, ce)
            if chrom == cc and min(end, ce) > max(cs, curr_start):
                #print('over', curr_start, end, cs, ce)
                if curr_start > cs:
                    seq += assembly[chrom][curr_start:cs]
                curr_start = ce
        if curr_start < end:
            seq += assembly[chrom][curr_start:end]
        #print(start, end, end-start, len(seq))

    else:
        # 从hg19的对应染色体上，取出anchor从开始到结束的碱基seq
        seq = assembly[chrom][start:end]
    # 如果取出的seq比min_size还要短
    if len(seq) < min_size:
        # diff = seq比min_size短多少
        diff = min_size - (end - start)
        # 将diff除于2，左边和右边各扩展ext_left的长度
        ext_left = diff // 2
        # 如果左边扩展以后越界了，则不扩展
        if start - ext_left < 0:
            ext_left = start
        # 如果右边扩展以后越界，则ext_left=diff-右边能扩展的最大长度
        elif diff - ext_left + end > len(assembly[chrom]):
            ext_left = diff - (len(assembly[chrom]) - end)
        # 经过扩展后的起始位点
        curr_start = start - ext_left
        curr_end = end + diff - ext_left

        # 如果起始位点扩展了，则seq也要跟着扩展
        if curr_start < start:
            seq = assembly[chrom][curr_start:start] + seq
        if curr_end > end:
            seq = seq + assembly[chrom][end:curr_end]
    if start < 0 or end > len(assembly[chrom]):
        return None
    return seq


def encode_seq(chrom, start, end, min_size=1000, crispred=None):
    #_get_sequence：如果anchor长度<min_size，则需要扩展长度，最后返回从start到end的碱基seq
    seq = _get_sequence(chrom, start, end, min_size, crispred)
    if seq is None:
        return None
    mat = get_seq_matrix(seq, len(seq), 'channels_first', one_d=True, rc=False)
    parts = []
    for i in range(0, len(seq), 500):
        if i + 1000 >= len(seq):
            break
        parts.append(mat[:, i:i + 1000])
    parts.append(mat[:, -1000:])
    parts = np.array(parts, dtype='float32')
    return parts


def print_feature_importances(estimator):
    f = open("/data/protein/gm12878_files.txt")
    ann_files = [r.strip().split('/')[-1] for r in f]
    f.close()
    feature_importances = [(idx, n, i) for idx, (n, i) in
                           enumerate(zip(['distance', 'correlation'] + ann_files * 2, estimator.feature_importances_))]
    feature_importances.sort(key=lambda k: -k[2])
    for idx, n, i in feature_importances:
        print(idx, '\t', n, "\t", i)


def plot_dist_distr(pos_dists_dict, neg_dists_dict, normed=True, savefig=None,
                    num_bins=50, dist_range=(np.log10(5000), np.log10(2000000))):
    pos_dists = []
    for c in pos_dists_dict:
        pos_dists += pos_dists_dict[c]
    neg_dists = []
    for c in neg_dists_dict:
        neg_dists += neg_dists_dict[c]
    n_counts, n_edges = np.histogram(np.log10(neg_dists), normed=normed, bins=num_bins, range=dist_range)
    p_counts, p_edges = np.histogram(np.log10(pos_dists), bins=num_bins, normed=normed, range=dist_range)
    n_centers = 0.5*(n_edges[:-1] + n_edges[1:])
    p_centers = 0.5*(p_edges[:-1] + p_edges[1:])
    fig = pl.figure()
    pl.plot(n_centers, n_counts, label="negative\n(%d)"%len(neg_dists))
    pl.plot(p_centers, p_counts, label="positive\n(%d)"%len(pos_dists))
    pl.legend(loc='upper left')
    pl.xlabel("Distance between centers of anchors (log10)", fontsize=14)
    if normed:
        pl.ylabel("Density", fontsize=14)
    else:
        pl.ylabel("Frequency", fontsize=14)
    if savefig is not None:
        fig.savefig(savefig + ".pdf", dpi=600)
        fig.savefig(savefig + ".jpg", dpi=300)
    #print("\n".join(["\t".join(map(str, i)) for i in zip(n_centers, n_counts, p_centers, p_counts)]))


def generate_features(a, annotations):
    temp_dist = np.log10(0.5*(a[4] + a[5] - a[1] - a[2])) / 7.0
    temp_mat1 = get_annotation_matrix(a[0],a[1],a[2], annotations, a[2]-a[1], 'channels_first')
    temp_mat2 = get_annotation_matrix(a[3],a[4],a[5], annotations, a[5]-a[4], 'channels_first')
    temp_mean1 = list(np.mean(temp_mat1, axis=1))
    temp_mean2 = list(np.mean(temp_mat2, axis=1))

    return [temp_dist] + temp_mean1 + temp_mean2


def get_matrix_binary(chrom, start, end, annotations, seq_len, data_format):
    curr_features = [0 for _ in range(len(annotations))]
    end = min(start + seq_len, end)
    for i, anno in enumerate(annotations):
        if chrom in anno[0]:
            recs = anno[0][chrom]
            rec_ends = anno[1][chrom]
            si = bisect.bisect(rec_ends, start)
            while si < len(recs) and recs[si][1] < end:
                curr_overlap = min(recs[si][2], end) - max(recs[si][1], start)
                if curr_overlap >= 100 or curr_overlap / (end - start) >= 0.5 or curr_overlap / (recs[si][2] - recs[si][1]) >= 0.5:
                    curr_features[i] += 1
                si += 1
    return curr_features


def generate_features_binary(a, annotations):
    chrom_map = {}
    for i, c in enumerate(list(range(1, 23)) + ['X']):
        chrom_map[i + 1] = 'chr' + str(c)
    a = list(a)
    if type(a[0]) == int:
        a[0] = chrom_map[a[0]]
    if type(a[3]) == int:
        a[3] = chrom_map[a[3]]

    temp_mean1 = get_matrix_binary(a[0], a[1], a[2], annotations, a[2] - a[1], 'channels_first')
    temp_mean2 = get_matrix_binary(a[3], a[4], a[5], annotations, a[5] - a[4], 'channels_first')
    temp_dist = np.log10(0.5 * (a[4] + a[5] - a[1] - a[2])) / 7.0
    return [temp_dist, ] + temp_mean1 + temp_mean2


def generate_data(pos_pairs, neg_pairs, annotations, binary=False, min_size=1000):
    if binary:
        gen_fn = generate_features_binary
    else:
        gen_fn = generate_features

    chrom_sizes = {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hg19.len'), 'r') as f:
        for r in f:
            tokens = r.strip().split()
            chrom_sizes[tokens[0]] = int(tokens[1])
    print(len(pos_pairs), len(neg_pairs))
    train_data = []
    val_data = []
    test_data = []
    train_label = []
    val_label = []
    test_label = []
    train_pairs = []
    val_pairs = []
    test_pairs = []
    val_chroms = ['chr5', 'chr14']
    test_chroms = ['chr' + str(i) for i in [4, 7, 8, 11]]
    for a in pos_pairs:
        if a[2] - a[1] < min_size:
            temp = (a[1] + a[2]) // 2
            a[1] = temp - min_size // 2
            a[2] = temp + (min_size - min_size // 2)
        if a[5] - a[4] < min_size:
            temp = (a[4] + a[5]) // 2
            a[4] = temp - min_size // 2
            a[5] = temp + (min_size - min_size // 2)
        if (a[1] < 0 or a[4] < 0 or
                a[2] >= chrom_sizes[a[0]] or a[5] >= chrom_sizes[a[3]]):
            print('skipping', a)
            continue
        curr_features = gen_fn(a, annotations)
        if a[0] in val_chroms:
            val_data.append(curr_features)
            val_label.append(1)
            val_pairs.append(a)
        elif a[0] in test_chroms:
            test_data.append(curr_features)
            test_label.append(1)
            test_pairs.append(a)
        else:
            train_data.append(curr_features)
            train_label.append(1)
            train_pairs.append(a)
    print("finished positive")

    for a in neg_pairs:
        curr_features = gen_fn(a, annotations)
        if a[0] in val_chroms:
            val_data.append(curr_features)
            val_label.append(0)
            val_pairs.append(a)
        elif a[0] in test_chroms:
            test_data.append(curr_features)
            test_label.append(0)
            test_pairs.append(a)
        else:
            train_data.append(curr_features)
            train_label.append(0)
            train_pairs.append(a)
    train_data, train_label, train_pairs = shuffle(train_data, train_label, train_pairs)
    val_data, val_label, val_pairs = shuffle(val_data, val_label, val_pairs)
    test_data, test_label, test_pairs = shuffle(test_data, test_label, test_pairs)
    train_data = np.array(train_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)
    return (train_data, train_label, train_pairs), (val_data, val_label, val_pairs), (test_data, test_label, test_pairs)


def save_data_to_hdf5(data, labels, pairs, name, dset, type_str):
    chrom_map = {}
    for i, x in enumerate(list(range(1, 23)) + ['X']):
        chrom_map['chr' + str(x)] = i + 1
    new_pairs = []
    for p in pairs:
        p = list(p)
        p[0] = chrom_map[p[0]]
        p[3] = chrom_map[p[3]]
        new_pairs.append(p)
    new_pairs = np.array(new_pairs)
    with h5py.File('%s_%s_%s.hdf5' % (name, type_str, dset), 'w') as a:
        a.create_dataset('data', data=data, chunks=True, compression='gzip')
        a.create_dataset('pairs', data=new_pairs, compression='gzip')
        a.create_dataset('labels', data=np.array(labels, dtype='int8'), compression='gzip')


def load_data_from_hdf5(name, dset, type_str):
    a = h5py.File('%s_%s_%s.hdf5' % (name, type_str, dset), 'r')
    data = a['data'][:]
    labels = a['labels'][:]
    pairs = a['pairs'][:]
    a.close()

    return data, labels, pairs

def check_chrom(chrom):
    import re
    return re.match('^chr(\d+|X)$', chrom)


def chrom_to_int(chrom):
    if chrom == 'chrX':
        chrom = 23
    else:
        chrom = int(chrom.replace('chr', ''))
    return chrom

def load_peaks(fn):
    peaks = {}
    with open(fn) as f:
        for r in f:
            tokens = r.strip().split()
            for i in [1,2]:
                tokens[i] = int(tokens[i])
            if tokens[0] not in peaks:
                peaks[tokens[0]] = []
            peaks[tokens[0]].append(tokens)
    for c in peaks:
        peaks[c].sort(key=lambda k:(k[1], k[2]))
    return peaks


def check_peaks(peaks, chrom, start, end):
    if chrom not in peaks:
        return False
    for p in peaks[chrom]:
        if min(end, p[2]) - max(start, p[1]) > 0:
            return True
    return False


def check_all_peaks(peaks_list, chrom, start, end):
    return [1 if check_peaks(peaks, chrom,start,end) else 0 for peaks in peaks_list]



def _load_data(fn, assembly, label,
               train_pairs, train_labels, train_mCG, val_pairs, val_labels,val_mCG,
               test_pairs, test_labels, test_mCG, peaks_list, allow_inter=False, breakpoints={}):
    int_cols = [1, 2, 4, 5]
    chrom_cols = [0, 3]
    val_chroms = [5, 14]
    test_chroms = [4, 7, 8, 11]
    with open(fn) as f:
        for r in f:
            tokens = r.strip().split()
            # 提取出mCG的数据
            mCG = tokens[-21:]
            tokens = tokens[:6]
            if not check_chrom(tokens[0]):
                continue

            for i in chrom_cols:
                tokens[i] = chrom_to_int(tokens[i])

            for i in int_cols:
                tokens[i] = int(tokens[i])

            if tokens[0] >= len(assembly) or tokens[3] >= len(assembly):
                continue

            if not allow_inter and tokens[0] != tokens[3]:
                print('skipping different chrom ', tokens)
                continue
            # 如果允许跨染色体的loop，那么计算距离
            elif allow_inter and tokens[0] != tokens[3]:
                #check distances
                if not (tokens[0] in breakpoints and tokens[3] in breakpoints):
                    continue
                temp_dl = 0.5 * (tokens[1] + tokens[2]) - breakpoints[tokens[0]]
                temp_dr = 0.5 * (tokens[4] + tokens[5]) - breakpoints[tokens[3]]
                #proper translocation, different sides of the translocation breakpoints
                if temp_dl * temp_dr > 0 or not (5000<=abs(temp_dl) + abs(temp_dr)<=2000000):
                    print('distance issues for different chromosome')
                    continue

                #change chromosome order
                if tokens[0] > tokens[3]:
                    temp = tokens[3:6]
                    tokens[3:6] = tokens[:3]
                    tokens[:3] = temp
            # 调整两个anchor的顺序
            if tokens[1] > tokens[4]:
                temp1,temp2,temp3 = tokens[3:6]
                tokens[3], tokens[4], tokens[5] = tokens[0], tokens[1], tokens[2]
                tokens[0], tokens[1], tokens[2] = temp1, temp2, temp3
            # 如果anchor范围有误，则skip
            if (tokens[1] < 0 or tokens[4] < 0 or
                    tokens[2] >= len(assembly[tokens[0]]) or
                    tokens[5] > len(assembly[tokens[3]]) or
                    (tokens[0] == tokens[3] and tokens[4] < tokens[2])):
                print('skipping', tokens)
                continue

            if (tokens[0] != tokens[3]) or (tokens[0] == tokens[3]
                                            and 5000. <= 0.5 * (tokens[4] - tokens[1] + tokens[5] - tokens[2]) <= 2000000):
                if len(tokens) < 7:
                    tokens.append(label)
                else:
                    tokens[6] = int(float(tokens[6]))
                if peaks_list is not None:
                    temp_peaks = check_all_peaks(peaks_list, *tokens[:3]) + check_all_peaks(peaks_list, *tokens[3:6])
                    tokens += temp_peaks

                tokens = tuple(tokens)

                if tokens[0] in val_chroms:
                    val_pairs.append(tokens)
                    val_labels.append(label)
                    val_mCG.append(mCG)
                elif tokens[0] in test_chroms:
                    test_pairs.append(tokens)
                    test_labels.append(label)
                    test_mCG.append(mCG)
                else:
                    train_pairs.append(tokens)
                    train_labels.append(label)
                    train_mCG.append(mCG)
        #print('tokens:',tokens)
        #print('mCG:',mCG)



def load_pairs(pos_files, neg_files, assembly, peaks_list=None, allow_inter=False, breakpoints={}):
    train_pairs = []
    train_methyl = []
    train_labels = []
    val_pairs = []
    val_methyl = []
    val_labels = []
    test_pairs = []
    test_methyl = []
    test_labels = []

    #for fn in pos_files:
    #print('Loading positive file...')
    _load_data(pos_files, assembly,1,
               train_pairs, train_labels,train_methyl,
               val_pairs, val_labels, val_methyl,
               test_pairs, test_labels, test_methyl, peaks_list, allow_inter, breakpoints)
    #print('positive mCG[:5]:',train_methyl[:5])

    #for fn in neg_files:
    #print('Loading negative file...')
    _load_data(neg_files, assembly, 0,
               train_pairs, train_labels, train_methyl,
               val_pairs, val_labels, val_methyl,
               test_pairs, test_labels, test_methyl, peaks_list, allow_inter, breakpoints)
    # 打乱顺序，但pairs和labels还是对应的
    train_pairs, train_labels, train_methyl = shuffle(train_pairs, train_labels, train_methyl)
    val_pairs, val_labels, val_methyl = shuffle(val_pairs, val_labels, val_methyl)
    test_pairs, test_labels, test_methyl = shuffle(test_pairs, test_labels, test_methyl)
    return train_pairs, train_labels, train_methyl, val_pairs, val_labels, val_methyl, test_pairs, test_labels, test_methyl


def __get_mat(p, left, min_size, ext_size, crispred=None):
    if left:
        chrom, start, end = (0, 1, 2)
    else:
        chrom, start, end = (3, 4, 5)
    curr_chrom = p[chrom]

    if ext_size is not None:
        min_size = p[end]-p[start] + 2*ext_size
    temp = encode_seq(curr_chrom, p[start], p[end], min_size=min_size, crispred=crispred)
    if temp is None:
        raise ValueError('Nong value for matrix')
    return temp


def get_one_side_data_parallel(pairs, pool, left=True, out=None, verbose=False,
                               min_size=1000, ext_size=None, crispred=None):
    edges = [0]
    data = pool.map(partial(__get_mat, left=left, min_size=min_size, ext_size=ext_size, crispred=crispred), pairs)
    for d in data:
        edges.append(d.shape[0] + edges[-1])

    return np.concatenate(data, axis=0), edges


def get_one_side_data(pairs, left=True, out=None, verbose=False, min_size=1000, ext_size=None, crispred=None):
    """
        根据一组基因对（pairs）获取单侧数据，并返回一个数组和一个列表，或者将数据存储在指定输出文件中。
        Args:
            pairs (list): 一个元素为元组的列表，每个元组包含 6 个值，分别是两个基因的名称、每个基因的起始位置和终止位置。
            left (bool): 一个布尔值，指定函数获取哪个基因的哪一侧的数据。如果为 True，则获取第一个基因的左侧区间的数据；如果为 False，则获取第二个基因的右侧区间的数据。
            out (Optional): 一个可选的输出文件，指定函数将输出数据存储在哪个 HDF5 文件中。如果为 None，则函数将返回 data 数组和 edges 列表。
            verbose (Optional): 一个可选的布尔值，如果为 True，则在处理数据时输出一些调试信息。
            min_size (Optional): 一个可选的整数，指定要获取的数据矩阵的最小大小。
            ext_size (Optional): 一个可选的整数，指定在每个基因的起始和终止位置外围添加的附加长度。
            crispred (Optional): 一个可选的布尔值，如果为 True，则使用 CRISPR-DNA 算法对 DNA 序列进行编码。
        Returns:
            如果 out 参数为 None，则返回一个数组 data 和一个列表 edges；否则，将数据存储在指定的 HDF5 文件中。
        """
    if out is not None:
        # 如果 out 参数被传递，创建 HDF5 数据集用于存储数据。
        data_name = "left_data" if left else "right_data"
        data_store = out.create_dataset(data_name, (50000, 4, 1000), dtype='uint8', maxshape=(None, 4, 1000),
                                        chunks=True, compression='gzip')
    # 根据 left 参数设置基因名称、起始位置和终止位置的索引。
    if left:
        chrom, start, end = (0, 1, 2)
    else:
        chrom, start, end = (3, 4, 5)
    edges = [0]  # 记录每个数据矩阵的末尾边缘位置。
    data = []  # 存储所有数据矩阵。
    last_cut = 0  # 记录上一个数据矩阵的末尾边缘位置。

    for p in pairs:
        curr_chrom = p[chrom] # 获取当前基因的名称。
        if type(curr_chrom) == int:
            if curr_chrom == 23:
                curr_chrom = 'chrX'
            else:
                curr_chrom = 'chr%d' % curr_chrom
        if ext_size is not None:
            # 计算要获取的数据矩阵的最小大小。
            min_size = p[end]-p[start] + 2*ext_size
        temp = encode_seq(p[chrom], p[start], p[end], min_size=min_size, crispred=crispred)
        if temp is None:
            raise ValueError('Nong value for matrix')
        new_cut = edges[-1] + temp.shape[0]  # 计算当前数据矩阵的末尾边缘位置。
        data.append(temp)  # 将当前数据矩阵添加到 data 列表中。
        edges.append(new_cut)  # 将当前数据矩阵的末尾边缘位置添加到 edges 列表中。
        # 如果 out 参数不为 None，并且当前数据矩阵的大小超过 50000，则将当前的 data 列表中的所有数据存储到 HDF5 文件中。
        if out is not None and new_cut - last_cut > 50000:
            data_store.resize((edges[-1], 4, 1000))
            data_store[last_cut:edges[-1]] = np.concatenate(data, axis=0)
            data = []
            last_cut = edges[-1]
            if verbose:
                print(last_cut, len(edges))
    # 将 data 列表中剩余的所有数据存储到 HDF5 文件中。
    if out is not None:
        data_store.resize((edges[-1], 4, 1000))
        data_store[last_cut:edges[-1]] = np.concatenate(data, axis=0)
        edge_name = 'left_edges' if left else 'right_edges'
        out.create_dataset(edge_name, data=edges, dtype='long', chunks=True, compression='gzip')
    else:
        # 如果 out 参数为 None，则返回 data 数组和 edges 列表。
        return np.concatenate(data, axis=0), edges


def get_and_save_data(pairs, labels, methyl, filename, min_size=1000, ext_size=None, crispred=None):
    print('using ext_size: ', ext_size)
    with h5py.File(filename, 'w') as out:
        pair_dtype = ','.join('uint8,u8,u8,uint8,u8,u8,u8'.split(',') + ['uint8'] * (len(pairs[0]) - 7))
        out.create_dataset('labels', data=np.array(labels, dtype='uint8'), chunks=True, compression='gzip')
        out.create_dataset('pairs', data=np.array(pairs, dtype=pair_dtype), chunks=True,compression='gzip')
        out.create_dataset('methyl', data=np.array(methyl, dtype='float32'), chunks=True, compression='gzip')
        print('Generating left anchor features...')
        get_one_side_data(pairs, left=True, out=out, verbose=True,min_size=min_size, ext_size=ext_size, crispred=crispred)
        print('Generating right anchor features...')
        get_one_side_data(pairs, left=False, out=out, verbose=True,
                          min_size=min_size, ext_size=ext_size, crispred=crispred)

def main():
    parser = argparse.ArgumentParser(description='generate hdf5 files for prediction.'
                                                 + " The validation chromosomes are 5, 14."
                                                 + " The test chromosomes are 4, 7, 8, 11. Rest will be training.")
    parser.add_argument('--pos_loop', type=str, required=True, help='The path of true loop.')
    parser.add_argument('--neg_loop', type=str, required=True, help='The path of false loop.')
    parser.add_argument('--assembly_data', type=str, required=True, help='The path of assembly data.')
    parser.add_argument('--methylation_data', type=str, required=True, help='The path of methylation data.')
    parser.add_argument('--name', type=str, required=True, help='The name of the cell.')
    parser.add_argument('--min_size', type=int, required=False, help='Minimum length of loop.')
    parser.add_argument('--ext_size', type=int, required=False, help='Extension length of loop.')
    parser.add_argument('--save_dir', type=str, required=True, help='The path of result.')
    parser.add_argument('--test_balance', type=bool, default=False ,help='Balance of test data.')
    args = parser.parse_args()

    load_assembly(args.assembly_data)

    true_loop = pd.read_csv(args.pos_loop,header=None,sep='\t')
    false_loop = pd.read_csv(args.neg_loop,header=None,sep='\t')
    true_loop.columns = ['chrom1','start1','end1','chrom2','start2','end2']
    false_loop.columns = ['chrom1','start1','end1','chrom2','start2','end2']
    true_loop['labels'] = 1
    false_loop['labels'] = 0
    test_chroms = ['chr4', 'chr7', 'chr8', 'chr11']
    # 如果测试集也要过滤成1：1
    #if args.test_balance:
    # 从 DataFrame 随机采样正样本相同的数量
    #false_loop = false_loop.sample(n=true_loop.shape[0], random_state=42)
    # else:
    #     # 非测试集设置为1：1
    #     false_noTest = false_loop[~false_loop['chrom1'].isin(test_chroms)]
    #     true_noTest = true_loop[~true_loop['chrom1'].isin(test_chroms)]
    #     false_noTest = false_noTest.sample(n=true_noTest.shape[0], random_state=42)
    #     # 测试集不动
    #     false_Test = false_loop[false_loop['chrom1'].isin(test_chroms)]
    #     false_loop = pd.concat([false_noTest, false_Test], axis=0, ignore_index=True)
    merge_loop = pd.concat([true_loop, false_loop], axis=0, ignore_index=True)

    methylation = pd.read_csv(args.methylation_data,header=None,sep='\t')
    methylation.columns = ['chrom','start','end','methyl_ratio','methyl_count','unmethyl_count']

    # 提取甲基化信息
    # 添加前10列
    print('Extracting methylation')
    for i in range(1, 11):
        merge_loop[f'left_methyl{i}'] = 0  # 或者初始化为您需要的默认值

    # 添加后10列
    for i in range(1, 11):
        merge_loop[f'right_methyl{i}'] = 0  # 或者初始化为您需要的默认值

    # 对甲基化和loop数据按照染色体排序
    merge_loop.sort_values(by='chrom1', ascending=True, inplace=True)
    methylation.sort_values(by='chrom', ascending=True, inplace=True)
    methylation['pos'] = methylation['start'] + 1
    methylation = methylation[['chrom','pos','methyl_ratio']]
    methylation = methylation[methylation['methyl_ratio'] >= 50]

    # 提取出甲基化计数
    merge_loop['corr'] = 0
    curr_chrom = ''
    for i in range(merge_loop.shape[0]):
        # 如果染色体发生改变
        if curr_chrom != merge_loop.iloc[i,0]:
            curr_chrom = merge_loop.iloc[i,0]
            curr_methyl = methylation[methylation['chrom'] == curr_chrom]
        # 处在左anchor范围内的甲基化
        left_methyl = curr_methyl[(curr_methyl['pos'] >= merge_loop.iloc[i,1]) & (curr_methyl['pos'] <= merge_loop.iloc[i,2])]
        # 统计左anchor10个区间内的甲基化位点计数
        bin_len = (merge_loop.iloc[i,2] - merge_loop.iloc[i,1]) // 10
        for j in range(1,11):
            merge_loop.iloc[i,6+j] = left_methyl[(left_methyl['pos'] >= merge_loop.iloc[i,1] + bin_len * (j-1)) & (left_methyl['pos'] <= merge_loop.iloc[i,1] + bin_len * j)].shape[0]
        # 处在右anchor范围内的甲基化
        right_methyl = curr_methyl[(curr_methyl['pos'] >= merge_loop.iloc[i,4]) & (curr_methyl['pos'] <= merge_loop.iloc[i,5])]
        # 统计左anchor10个区间内的甲基化位点计数
        bin_len = (merge_loop.iloc[i,5] - merge_loop.iloc[i,4]) // 10
        for j in range(1,11):
            merge_loop.iloc[i,16+j] = right_methyl[(right_methyl['pos'] >= merge_loop.iloc[i,4] + bin_len * (j-1)) & (right_methyl['pos'] <= merge_loop.iloc[i,4] + bin_len * j)].shape[0]
        merge_loop.iloc[i,27] = np.corrcoef(list(merge_loop.iloc[i,7:17]), list(merge_loop.iloc[i,17:27]))[0, 1]
        if i % 1000 == 0:
            print('Extracted methylation:',i,'/',merge_loop.shape[0])
    # 去除包含nan的列
    merge_loop_cleaned = merge_loop.dropna()
    # 改造成pos 和 neg file
    merge_loop_cleaned[merge_loop_cleaned['labels'] == 1].to_csv(args.save_dir+args.name+'_pos_loop_methyl.csv',sep='\t',index=None)
    merge_loop_cleaned[merge_loop_cleaned['labels'] == 0].to_csv(args.save_dir+args.name+'_neg_loop_methyl.csv',sep='\t',index=None)

    name = args.name
    pos_files = args.save_dir+args.name+'_pos_loop_methyl.csv'
    neg_files = args.save_dir+args.name+'_neg_loop_methyl.csv'
    dataset_names = ['train', 'valid', 'test']
    out_dir = args.save_dir

    train_pairs, train_labels, train_methyl,val_pairs, val_labels, val_methyl,test_pairs, test_labels,test_methyl = load_pairs(pos_files,neg_files,assembly)

    #data_pairs将训练、验证和测试集的数据拼接成一个列表，data_labels同理
    data_pairs = [train_pairs, val_pairs, test_pairs]
    data_labels = [train_labels, val_labels, test_labels]
    data_methyl = [train_methyl, val_methyl, test_methyl]
    out_idxes = [0, 1, 2]

    #0存储为训练集，1存储为验证集，2存储为测试集
    for idx in out_idxes:
        #pairs[(2, 38207803, 38213410, 2, 38500151, 38503080, 106), (19, 47011724, 47022032, 19, 47032049, 47036050, 18)]
        pairs = data_pairs[idx]
        print('idx:',idx)
        if len(pairs) == 0:
            print('continue')
            continue
        #labels[0,0]
        labels = data_labels[idx]
        methyl = data_methyl[idx]
        #dset=['train','valid','test']
        dset = dataset_names[idx]
        fn = os.path.join(out_dir,
                          "{}_singleton_tf_with_random_neg_seq_data_length_filtered_{}.hdf5".format(name, dset))
        print(fn)
        get_and_save_data(pairs, labels, methyl, fn)

    
if __name__ == "__main__":
    main()