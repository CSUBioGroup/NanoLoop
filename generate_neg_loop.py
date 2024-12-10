# 用滑动窗口随机生成负样本loop，并且按距离匹配生成一定比例的正负样本，合并后成为train,test,valid数据集
import pandas as pd
import random
import argparse

# 读取正样本loop
def read_pos_loop(path):
    columns = ['chrom1','start1','end1','chrom2','start2','end2']
    pos_loop = pd.read_csv(path, sep='\t',header=None, names=columns)
    return pos_loop

# 2.读取染色体长度文件
def read_chrom_sizes_file(file_path):
    chrom_sizes = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 使用制表符或空格分隔每一行的数据，并去除两端的空白字符
            chrom, size = line.strip().split()
            # 将染色体名和长度存储到字典中
            chrom_sizes[chrom] = int(size)
    return chrom_sizes

# 输入染色体号数，返回染色体长度
def get_chrom_length(chrom_sizes_dict, chrom):
    # 获取指定染色体的长度
    chrom_len = chrom_sizes_dict.get(chrom, None)
    if chrom_len is not None:
        print(f"The length of {chrom}: {chrom_len}")
    else:
        print(f"Chromosome {chrom} not found in the file.")

    return chrom_len

# 将正样本loop拆分成anchor
def loop_to_anchor(chr_pos):
    # 先将loop拆分成anchor
    chrom1_list = chr_pos['chrom1'].tolist()
    chrom2_list = chr_pos['chrom2'].tolist()
    chrom_list = chrom1_list + chrom2_list

    start1_list = chr_pos['start1'].tolist()
    start2_list = chr_pos['start2'].tolist()
    start_list = start1_list + start2_list

    end1_list = chr_pos['end1'].tolist()
    end2_list = chr_pos['end2'].tolist()
    end_list = end1_list + end2_list

    # 使用DataFrame构造函数生成DataFrame
    chr_anchors = pd.DataFrame({'chrom': chrom_list, 'start': start_list, 'end': end_list})
    # 对新的DataFrame按照 start1 列进行排序
    chr_anchors.sort_values(by=['chrom', 'start'], inplace=True)
    # 基于所有列的去重
    chr_anchors = chr_anchors.drop_duplicates()
    # 重置索引
    chr_anchors.reset_index(drop=True, inplace=True)

    return chr_anchors

# 用窗口滑动函数生成随机的anchor
# 输入一个长度为2000的范围，即可生成在此区间内的负anchor
def generate_neg_anchor(chrom, last_anchor_end, neg_anchor_df, anchor_len, extend_len):

    # anchor的长度为10000
    # anchor_len = 10000
    # anchor间要有500bp以上的间隔，避免重叠
    start = last_anchor_end + extend_len
    end = start + anchor_len
    # 将数据添加到df
    new_data = {'chrom': chrom, 'start': start, 'end': end}
    new_row = pd.DataFrame(new_data, index=[0])
    neg_anchor_df = pd.concat([neg_anchor_df, new_row], ignore_index=True)

    # 返回新的end和neg_anchor_df
    return neg_anchor_df, end

# 遍历真实anchor，如果两个loop之间的间距大于阈值，则把这个间距传入generate_neg_anchor函数，并生成负anchor
def slide_window(true_anchors, anchor_len, extend_len, neg_anchor):
    # 如果两个anchor间的距离大于window_len，那么可以在这两个anchor间生成一个负anchor
    window_len = anchor_len + 2*extend_len
    # 获取当前chrom的anchors
    chrom_list = ["chr" + str(i) for i in range(1, 23)] + ['chrX']
    for curr_chrom in chrom_list:
        curr_true_anchor = true_anchors[true_anchors['chrom'] == curr_chrom]
        for i in range(curr_true_anchor.shape[0]-1):
            # 下一行anchor的start - 上一行anchor的end，如果这个区间大小大于window_len，那么可以在这个区间内生成一个负anchor
            if curr_true_anchor.iloc[i+1, 1] - curr_true_anchor.iloc[i, 2] >= window_len:
                last_anchor_end = curr_true_anchor.iloc[i, 2]
                while curr_true_anchor.iloc[i+1, 1] - last_anchor_end >= window_len:
                    # 每生成一个anchor，end都会向chr_anchors.iloc[i+1, 1]逼近，直到他们之间的距离不能再生成一个新的anchor
                    neg_anchor, last_anchor_end = generate_neg_anchor(curr_chrom, last_anchor_end, neg_anchor, anchor_len, extend_len)
            if i % 1000 == 0:
                print(curr_chrom,':',i,'/',curr_true_anchor.shape[0])

    return neg_anchor

# 计算loop距离
def loop_distance_distribution(true_loops):
    true_loops['loop_distance'] = (true_loops['start2'] + true_loops['end2']) / 2 - (true_loops['start1'] + true_loops['end1']) / 2
    return true_loops

# 获取正样本loop在每个距离段中的数量分布
def get_ture_loop_distribution(true_loops):
    loop_distance = loop_distance_distribution(true_loops)
    # 千，万，十万，百万
    each_bin_count = [0,0,0,0,0,0]
    each_bin_count[0] = true_loops[true_loops['loop_distance'] < 1000].shape[0]
    each_bin_count[1] = true_loops[(true_loops['loop_distance'] >= 1000) & (true_loops['loop_distance'] < 10000)].shape[0]
    each_bin_count[2] = true_loops[(true_loops['loop_distance'] >= 10000) & (true_loops['loop_distance'] < 100000)].shape[0]
    each_bin_count[3] = true_loops[(true_loops['loop_distance'] >= 100000) & (true_loops['loop_distance'] < 1000000)].shape[0]

    return each_bin_count

# 负anchor生成负loop
# 从x，y中生成n个不相等的实数
def generate_unique_integers(x, y, n):
    if n > (y - x + 1):
        n = y - x + 1
        #raise ValueError("Cannot generate n unique integers in the given range.")

    unique_integers = set()
    while len(unique_integers) < n:
        unique_integers.add(random.randint(x, y))

    return list(unique_integers)

# (start1,end1)是第一个负anchor，该函数可以找到第二个负anchor，然后把这两个负anchor组合成负loop
# def sample_from_interval(x, y, chrom, start1, end1, neg_anchor_df, neg_loop):
#     # 从区间中随机选取1个anchor
#     index_list = generate_unique_integers(x, y, 1)
#     for random_index in index_list:
#         # 计算两anchor的距离
#         distance = int((neg_anchor_df.iloc[random_index, 2] + neg_anchor_df.iloc[random_index, 1]) / 2 - (start1 + end1) / 2)
#         if distance >= 10000:
#             # 生成一个loop，然后添加到neg_loop
#             new_data = {'chrom1': chrom, 'start1': start1, 'end1': end1, 'chrom2': chrom, 'start2': neg_anchor_df.iloc[random_index, 1], 'end2': neg_anchor_df.iloc[random_index, 2], 'bin': int("{:.2e}".format(distance).split('e')[1])-2}
#             new_row = pd.DataFrame(new_data, index=[0])
#             neg_loop = pd.concat([neg_loop, new_row], ignore_index=True)
#     return neg_loop

def sample_from_interval(x, y, chrom, start1, end1, neg_anchor_df, neg_loop):
    # 从区间中随机选取3个anchor
    index_list = generate_unique_integers(x, y, 3)
    for random_index in index_list:
        # 计算两anchor的距离
        distance = int((neg_anchor_df.iloc[random_index, 2] + neg_anchor_df.iloc[random_index, 1]) / 2 - (start1 + end1) / 2)
        if distance >= 10000:
            # 生成一个loop，然后添加到neg_loop
            new_data = {'chrom1': chrom, 'start1': start1, 'end1': end1, 'chrom2': chrom, 'start2': neg_anchor_df.iloc[random_index, 1], 'end2': neg_anchor_df.iloc[random_index, 2], 'bin': int("{:.2e}".format(distance).split('e')[1])-2}
            new_row = pd.DataFrame(new_data, index=[0])
            neg_loop = pd.concat([neg_loop, new_row], ignore_index=True)
    return neg_loop

# 利用负anchor生成距离介于1w-100w之间的负loop
# def generate_neg_loop(neg_anchors, neg_loop):
#     chrom_list = ["chr" + str(i) for i in range(1, 23)] + ['chrX']
#     #chrom_list = ["chr1"]
#     for curr_chrom in chrom_list:
#         curr_neg_anchor = neg_anchors[neg_anchors['chrom'] == curr_chrom]
#         print(curr_chrom)
#         a = 1
#         for i in range(curr_neg_anchor.shape[0]-1):
#             # a表示距离当前anchor距离不超过100w的anchor的索引
#             while curr_neg_anchor.iloc[a, 2] - curr_neg_anchor.iloc[i, 2] < 1000000:
#                 if a+1 < curr_neg_anchor.shape[0]:
#                     a += 1
#                 # 如果a已经移动到最后一个anchor，那么跳出循环
#                 else:
#                     break
#             neg_loop = sample_from_interval(i+1, a, curr_chrom, curr_neg_anchor.iloc[i,1], curr_neg_anchor.iloc[i,2], curr_neg_anchor, neg_loop)
#             if i % 1000 == 0 and i != 0:
#                 print(curr_chrom,'processed:',i,'/',curr_neg_anchor.shape[0])
#
#         	# 如果a已经移动到最后一个anchor，那么a不动，只移动i
#             i += 1
#             while i < a:
#                 neg_loop = sample_from_interval(i, a, curr_chrom, curr_neg_anchor.iloc[i,1], curr_neg_anchor.iloc[i,2], curr_neg_anchor, neg_loop)
#                 i += 1
#                 if i >= a:
#                     break
#                 if i % 1000 == 0:
#                     print(curr_chrom,'processed:',i,'/',curr_neg_anchor.shape[0])
#             print('Generated',neg_loop.shape[0],' negative loops')
#     return neg_loop

def generate_neg_loop(neg_anchors, neg_loop):
    chrom_list = ["chr" + str(i) for i in range(1, 23)] + ['chrX']
    for curr_chrom in chrom_list:
        curr_neg_anchor = neg_anchors[neg_anchors['chrom'] == curr_chrom]
        print(curr_chrom)
        for i in range(curr_neg_anchor.shape[0]-3):
            neg_loop = sample_from_interval(i+1, curr_neg_anchor.shape[0]-3, curr_chrom, curr_neg_anchor.iloc[i,1], curr_neg_anchor.iloc[i,2], curr_neg_anchor, neg_loop)
            if i % 1000 == 0 and i != 0:
                print(curr_chrom,'processed:',i,'/',curr_neg_anchor.shape[0],'anchors')
                print(curr_chrom,':generated',neg_loop.shape[0],' negative loops')
    return neg_loop

#按比例采样负样本，neg_ratio表示负样本数量是正样本的neg_ratio倍
def sample_neg_loop(neg_loop, sample_neg_chrx_loop, neg_ratio, each_bin_count):
    # 千，万，十万，百万
    #  0  1   2    3
    # loop_distance_count
    # 只选10w-100w的loop即可
    for i in range(2, 4):
        # 先筛选出某个距离区间内的所有neg loop
        bin_neg_loop = neg_loop[neg_loop['bin'] == i]
        # 删除bin列
        del bin_neg_loop['bin']
        # 正样本loop在该区间内的数量
        pos_num = each_bin_count[i]
        print(10 ** (i+3),'的true Loop有:', pos_num)
        print(10 ** (i+3),'的false Loop有:',bin_neg_loop.shape[0])
        # 比较两者的大小，如果负样本的数量达不到所需要求，那就把save（保存）列全部置为1，意为全部负样本都使用上
        if int(pos_num) * int(neg_ratio) >= bin_neg_loop.shape[0]:
            print('负Loop数量不足，所有负Loop都将被使用')
            bin_neg_loop['save'] = 1
        # 如果负样本样本够，那么就按照比例筛选需要的数量
        else:
            print('负Loop数量充足，将从中随机采样倍数为:',neg_ratio)
            save_list = [0] * bin_neg_loop.shape[0]
            # 随机选择pos_num * neg_ratio个索引，并将这些索引对应的元素设置为1
            save_indices = random.sample(range(bin_neg_loop.shape[0]), pos_num * neg_ratio)
            for index in save_indices:
                save_list[index] = 1
            # 将list付给负样本loop的save列
            bin_neg_loop['save'] = save_list
        # 提取save列为1的loop
        save_neg_loop = bin_neg_loop[bin_neg_loop['save'] == 1]
        print(10 ** (i+3),'的采样false Loop有:',bin_neg_loop.shape[0])
        # 删除save列
        del save_neg_loop['save']
        # 将要保留的负loop存入sample_neg_loop
        sample_neg_chrx_loop = pd.concat([sample_neg_chrx_loop, save_neg_loop], ignore_index=True)
    return sample_neg_chrx_loop


# 将正负样本合并
def merge_pos_neg_loop(sample_neg_chrx_loop, chrx_pos):
    # 添加label列
    sample_neg_chrx_loop['label'] = 0
    chrx_pos['label'] = 1
    pos_loop = chrx_pos[['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'label']]
    # 合并
    merge_loop = pd.concat([sample_neg_chrx_loop, pos_loop], ignore_index=True)

    # 对数据进行乱序
    shuffled_merge_loop = merge_loop.sample(frac=1, random_state=42).reset_index(drop=True)

    return shuffled_merge_loop

    # 在循环中将训练、测试、验证集都处理好
def process_data(true_loops, output_path, anchor_len, extend_len, sample_proportion):
    # 将正样本loop拆分成anchor
    true_anchors = loop_to_anchor(true_loops)
    # 生成负anchor
    print('generate neg anchor')
    # 生成用于存储负anchor的空dataframe
    neg_anchors = pd.DataFrame(columns=['chrom', 'start', 'end'])
    neg_anchors = slide_window(true_anchors, anchor_len, extend_len, neg_anchors)
    # 生成负样本loop,bin代表该loop的距离属于千（1），万（2），十万（3），百万（4）
    neg_loops = pd.DataFrame(columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'bin'])
    print('generate neg loop')
    neg_loops = generate_neg_loop(neg_anchors, neg_loops)
    # 按照正样本距离匹配采样
    sampled_neg_loops =  pd.DataFrame(columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])
    # 获取正样本loop在每个距离的数量
    true_loop_each_bin_count = get_ture_loop_distribution(true_loops)
    # 正负样本比是1：5
    sampled_neg_loops = sample_neg_loop(neg_loops, sampled_neg_loops, sample_proportion, true_loop_each_bin_count)
    # 保存负样本
    sampled_neg_loops.to_csv(output_path, sep='\t', index=None, header=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate hdf5 files for prediction.'
                                                 + " The validation chromosomes are 5, 14."
                                                 + " The test chromosomes are 4, 7, 8, 11. Rest will be training.")
    parser.add_argument('--input_path', type=str, required=True, help='the path of train loop')
    parser.add_argument('--chrom_sizes_path', type=str, required=True, help='the path of chrom size file')
    parser.add_argument('--anchor_len', type=int, required=False, default=10000, help='The length of anchor.')
    parser.add_argument('--extend_len', type=int, required=False, default=500, help='The extend length of anchor.')
    parser.add_argument('--sample_proportion', type=int, required=False, default=5, help='The proportion of positive and negative samples.')
    parser.add_argument('--output_path', type=str, required=True, help='The save path of train loop.')
    args = parser.parse_args()

    # 读取真实的loop
    input_path = args.input_path
    true_loops = read_pos_loop(input_path)

    # 读取染色体size
    chrom_sizes_path = args.chrom_sizes_path
    chrom_sizes_dict = read_chrom_sizes_file(chrom_sizes_path)

    anchor_len = args.anchor_len
    extend_len = args.extend_len
    sample_proportion = args.sample_proportion

    output_path = args.output_path

    # 生成负loop
    process_data(true_loops, output_path, anchor_len, extend_len, sample_proportion)








