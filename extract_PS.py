import itertools
import os
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from scipy.integrate import quad, dblquad, tplquad, nquad
from sklearn.tree import export_text
from scipy.interpolate import interp1d,interp2d



# ETC-PS只是一种特征，他用的是机器学习方法做分类
# 再梳理一下，一个flow产生四个路径特征序列 + 一个时间维度，对应一个标签
# 然后对每一个路径按等级窗口切分，把特征序列转化为积分值序列，也就是一个路径积分特征
# 所以最终的特征行数 = 样本数的四倍，因为只选择了数据包长度作为特征，因此不同的长度序列试做不同的特征维度
# 每个特征样本产生的积分值数量 = C43 * (A31 + A32 + A33) = 4 * 15 = 60个，也就是路径积分特征的长度 = 60
# 因此，每个flow对应的特征矩阵 行数： flow_len / sequence_length , 列数：60

hierarchical_dyadic_window = 2 # path切分到subpath的粒度有这么多种，也就是窗口有四层

ISCX_sample_num = 445
# ISCX_sample_num = 40
ISCX_window_size = 40
# 读取文件夹中的所有文件，并把文件名字符串作为标签，文件中的每一列是整个文件的数据包的特征向量，需要按照最大sequence长度进行切分
def read_files_and_labels():
    all_files = []
    features = []
    labels = []

    files = os.listdir('/home/lxc/ETC-PS/data/vpn')
    for filename in files:
        file_path = os.path.join('/home/lxc/ETC-PS/data/vpn', filename)
        all_files.append(file_path)

    files = os.listdir('/home/lxc/ETC-PS/data/nonvpn')
    for filename in files:
        file_path = os.path.join('/home/lxc/ETC-PS/data/nonvpn', filename)
        all_files.append(file_path)

    
    for filename in all_files:
        print("cur file:",filename)
        # 解析文件名作为标签
        label = filename.split("/")[-1]
        file_path = filename

        # 计数器，用于跟踪样本数量
        sequence_sample_count = 0
        sequence_max = ISCX_window_size
        cur_sequence_features = [[], [], [], []] # U0,D0,U0CS,D0CS,相当于四个积分维度了

        with open(file_path, 'r') as file:
            file.readline() # 读取第一列
            cur_class_num = 0
            for line in file:
                cur_pkt_features = line.strip().split(',')
                # 当前数据包特征写入对应的sequence位置
                for i in range(4):
                    cur_sequence_features[i].append(float(cur_pkt_features[i]))
                sequence_sample_count += 1
                # 达到指定长度，组成一个sequence，写入全局特征和全局label
                if sequence_sample_count == sequence_max:
                    features.append(cur_sequence_features)
                    labels.append(label)
                    cur_sequence_features = [[], [], [], []]
                    sequence_sample_count = 0
                    
                    cur_class_num += 1 # 本类样本多了一个sequence
                if cur_class_num >= ISCX_sample_num:
                    break

    return np.array(features), np.array(labels)

# 对于非递增序列，无法做积分，因此需要额外的时间维度，之前所有版本都没考虑特征的时间维度
def path_signature(path):
    # print(path)
    # [[[    0.   -76.     0.   -52.     0.   -52.     0.   -52.     0.   -52.]]
    #  [[    0.   -76.   -76.  -128.  -128.  -180.  -180.  -232.  -232.  -284.]]]

    integrated_sequence = []
    # print(integrated_sequence)

    # 输入：combined组合之后的sequence或者sequence的一部分

    # 计算一重路径积分
    for j in range(path.shape[0]):  # 假设 sub_feature.shape[1] 是维度数
        # 按照逻辑，sequence的每一列都是对应一个sequence，都是递增的
        # x_lower = path[0, j]  # 第一个元素
        # x_upper = path[-1, j]  # 最后一个元素
        
        x_lower = np.min(path[j,:])
        x_upper = np.max(path[j,:])
        def integrand_1(x):
            return 1
        print("x_lower:",x_lower,"x_upper:",x_upper)
        result_single, _ = quad(integrand_1, x_lower, x_upper, epsabs=0.1, epsrel=0.1)

        integrated_sequence.append([result_single])
        # print("path.shape:",path.shape)
        if path.shape[0] == 1:
            return integrated_sequence
        
    # 计算二重路径积分
    # 即使之前已经计算过所有组合的，组合内部的组合还是要算，例如某个三维特征内的二维积分还是要遍历，而且是在三维积分维度的基础上遍历
    two_dims = list(itertools.combinations(range(path.shape[0]), 2))
    print(two_dims)
    two_dim_integrates = []
    # 二维列组合,混合积分
    for two_dim in two_dims:
        dim1, dim2 = two_dim
        # 获取积分范围，假设路径的每一列都是递增的
        dim1_lower = np.min(path[dim1,:])
        dim1_upper = np.max(path[dim1,:])
        dim2_lower = np.min(path[dim2,:])
        dim2_upper = np.max(path[dim2,:])
        print("x_lower:",dim1_lower,"x_upper:",dim1_upper,"y_lower:",dim2_lower,"y_upper:",dim2_upper)
        # 数值错误，太大了
        # def integrand_1(x):
        #     return x - dim1_lower
        # def integrand_2(y):
        #     return y - dim2_lower

        # 数值是对,但这样计算的其实是矩形面积
        # def integrand_1(x):
        #     return 1
        # def integrand_2(y):
        #     return 1
        # [[939.0, 295.0, [277005.0, 277005.0]]

        # 函数拟合y与x的关系
        f = interp1d(path[dim1,:].flatten(), path[dim2,:].flatten(), kind='linear', bounds_error=False, \
                     fill_value=(path[dim2,:].min(), path[dim2,:].max()))
        def integrand_12(x,y):
            if y < f(x):
                return 1
            else:
                return 0
        def integrand_21(x,y):
            if y > f(x):
                return 1
            else:
                return 0
        
        # result_double_12, _ = quad(lambda x: quad(lambda y: integrand_12(x,y), dim1_lower, dim1_upper)[0], dim2_lower, dim2_upper)
        # result_double_21, _ = quad(lambda y: quad(lambda x: integrand_21(x,y), dim2_lower, dim2_upper)[0], dim1_lower, dim1_upper)
        # # [[1360.0, 234.0, [0.0, 318240.0]], [1360.0, 234.0, [0.0, 318240.0]]]

        # result_double_12, _ = dblquad(integrand_12, dim1_lower, dim1_upper, lambda x: dim2_lower, lambda x: dim2_upper)
        # result_double_21, _ = dblquad(integrand_21, dim2_lower, dim2_upper, lambda y: dim1_lower, lambda y: dim1_upper)
        # # [[1360.0, 234.0, [318240.0, 316270.3732226095]], [1360.0, 234.0, [318240.0, 316270.3732226095]]]
        
        # 尚未理解，但是确实是曲线上下两部分的面积了
        # 5.22,降低积分精度，提速,我对结果要求只保留一位小数，精度直接选0.1
        result_double_12, _ = dblquad(integrand_12, dim2_lower, dim2_upper, lambda x: dim1_lower, lambda x: dim1_upper, epsabs=0.1, epsrel=0.1)
        result_double_21, _ = dblquad(integrand_21, dim2_lower, dim2_upper, lambda y: dim1_lower, lambda y: dim1_upper, epsabs=0.1, epsrel=0.1)
        # [[1360.0, 234.0, [1969.6198188620278, 316270.3732226095]], [1360.0, 234.0, [1969.6198188620278, 316270.3732226095]]]
        # [[1360.0, 5095.0, [2660657.5949453716, 4268542.431821445]], [1360.0, 5095.0, [2660657.5949453716, 4268542.431821445]]]
        result_double_12 = round(result_double_12,1)
        result_double_21 = round(result_double_21,1)

        two_dim_integrates.extend([result_double_12,result_double_21])
    
    # 单维度的二维积分，即1/2对应一重积分的平方
    for j in range(path.shape[0]):
        x_lower = np.min(path[j,:])
        x_upper = np.max(path[j,:])
        result_single, _ = quad(integrand_1, x_lower, x_upper, epsabs=0.1, epsrel=0.1)

        two_dim_integrates.append((result_single / 2) ** 2)
    
    # 整体加入进来，没考虑积分值的位置顺序，应该不重要
    integrated_sequence.append(two_dim_integrates)
    if path.shape[0] == 2:
        return integrated_sequence
    return
    # 计算三重路径积分
    triple_dims = list(itertools.combinations(range(path.shape[1]), 3))
    # 因为这样排列组合，会出现例如123和132的组合
    # 这样就不需要在每一个triple_dim组合内部考虑维度的顺序问题，可以统一看做第三个维度z是前两个维度的函数
    triple_dim_integrates = []
    for triple_dim in triple_dims:
        dim1,dim2,dim3 = triple_dim
        x = path[:, dim1, 0]  # 第一个维度的数据
        y = path[:, dim2, 1]  # 第二个维度的数据
        z = path[dim3, :, 2]  # 第三个维度的数据
        values = z  # 目标值
        # 假设积分区间
        x_lower, x_upper = np.min(x), np.max(x)
        y_lower, y_upper = np.min(y), np.max(y)
        z_lower, z_upper = np.min(z), np.max(z)

        # 创建插值函数
        
        f = interp2d(x, y,values, kind='linear', bounds_error=False, \
                     fill_value=(z_lower, z_upper))

        # 定义积分函数
        def integrand_below(x, y, z):
            return 1 if z < f((x, y)) else 0

        def integrand_above(x, y, z):
            return 1 if z > f((x, y)) else 0

        

        # 计算不同顺序的三重积分
        # 曲面下的体积
        result_below, _ = tplquad(integrand_below, x_lower, x_upper, 
                                lambda x: y_lower, lambda x: y_upper, 
                                lambda x, y: z_lower, lambda x, y: z_upper)

        # 曲面上的体积
        result_above, _ = tplquad(integrand_above, x_lower, x_upper, 
                                lambda x: y_lower, lambda x: y_upper, 
                                lambda x, y: z_lower, lambda x, y: z_upper)

        triple_dim_integrates.extend([result_below,result_above])
    integrated_sequence.append(triple_dim_integrates)

    if path.shape[1] == 3:
        return integrated_sequence
    return integrated_sequence



def split_and_integrate_hierarchical_dyadic_window(max_window_depth, features, labels):
    """
        根据分层二进制窗口的描述将特征序列切分成子路径，并在每个子路径内进行路径积分
        参数:
        q (int): 深度
        features (list): 特征列表，每个特征是一个序列
        labels (list): 标签列表，与特征对应
        返回:
        list: 子路径的路径积分结果列表
        list: 对应的标签列表
        特征格式,有多少个长度为40的数据包子序列就有多少个sequence，每个sequence是4*40的数组
        [  sequence1
            [[     0.      0.      0.      0.   -160.   -235.   -235.      0. 0.       0.      0.]
            [   382.     63.     63.     63.      0.      0.      0.     60. 52. 40.     93.]
            [-17384. -17384. -17384. -17384. -17544. -17779. -18014. -18014. -23007. -23665. -23665.]
            [ 10467.  10530.  10593.  10656.  10656.  10656.  10656.  10716. 13141. 13261.  13354.]],
           sequence2
            [[     0.      0.      0.      0.   -160.   -235.   -235.      0. 0.       0.      0.]
            [   382.     63.     63.     63.      0.      0.      0.     60. 52. 40.     93.]
            [-17384. -17384. -17384. -17384. -17544. -17779. -18014. -18014. -23007. -23665. -23665.]
            [ 10467.  10530.  10593.  10656.  10656.  10656.  10656.  10716. 13141. 13261.  13354.]],
        
        ]
    """
    assert len(features) == len(labels), "特征和标签数量不匹配"
    
    # 存储所有的路径积分结果和对应的标签
    integrated_pathSig_features = []
    integrated_labels = []

    # 遍历每个序列，得到对应的路径签名
    for sequence_feature, label in zip(features, labels):
        # print(sequence_feature.shape) # (67, 4, 40)
        # print(sequence_feature) # 嵌套列表
        sub_pathSig_feature = []
        sub_label = []
        """
            根据深度q,将特征和标签分割成子路径,共2^(q-1)个子路径，窗口彼此不重叠
            第i个窗口长度分别为n, n/2, n/4, . . . , n/2^(q-1)
            对应的窗口个数分别为所有满足长度的，连续的取值区间，分别为1，2，4，...，2^(q-1)个
            等比数列求和，一共2^(q)-1个窗口，每个窗口视为路径进行积分

            1.先切分窗口，然后对每个窗口分别进行一维到三维积分
            2.或者先遍历积分维度的数量，然后对每个维度的积分再切分窗口分别积分，用的这个
            循环的嵌套顺序没那么重要，最后只要是积分路径就可以
        """

        # 计算最大积分深度,我只有四个特征序列,所以最大积分深度只能是4，和给定的积分要求取最小。
        upper_integrate_depth = len(sequence_feature) + 1
        print(upper_integrate_depth)
        for integrate_depth in range(1,upper_integrate_depth - 2):
            # 用于跳过某个维度的积分
            # if integrate_depth != 1:
            #     continue

            # 将特征列表转换为特征矩阵，这样每一列是特征维度，分别是U0,D0,U0CS,D0CS，每一行是sequence的一个数据包
            feature_matrix = np.array(sequence_feature)
            # print(feature_matrix.shape) # (4, 40)，对的，一个sequence有四个路径的特征，sequence长度为40
            # print(feature_matrix) # 嵌套列表，有sample_num列，4行对应U0,D0,U0CS,D0CS
            sequence_length_40 = feature_matrix.shape[1]

            # 生成所有可能的列组合,用于计算多维积分
            combinations = []
            for i in range(1,integrate_depth+1):
                combinations.append(list(itertools.combinations(range(feature_matrix.shape[0]), i)))
            # print(combinations)

            # 遍历同一级分深度下，不同大小的窗口，也就是从n, n/2, n/4, . . . , n/2^(q-1)开始切分窗口
            for window_depth in range(max_window_depth):
                length = sequence_length_40 // (2 ** window_depth)
                step = length
                print("label, integrate dimension num, cur window size:",label, integrate_depth, length)
                if step == 0:
                    continue
                
                # 选择窗口遍历当前大小的每一个窗口
                for k in range(0, sequence_length_40 - length + 1, step):
                    # 选择特征列，对当前积分深度下，当前窗口的每一种维度组合
                    for combination in combinations[ integrate_depth-1 ]:
                        # 选择当前窗口下的，当前特征列
                        print("combination,k:k+length:",combination,k,k+length)
                        # combination,k:k+length: (2,) 20 40
                        if integrate_depth == 1:
                            sub_feature = feature_matrix[combination][k:k+length]
                        else:
                            sub_feature = feature_matrix[list(combination),k:k+length]
                        """
                            sub_feature:
                            [    0.   -76.     0.   -52.     0.   -52.     0.   -52.     0.   -52.
                            0.   -52.     0.   -52.     0.   -52.     0.   -52.     0.   -52.
                            0.     0.     0.     0.  -256.  -256.  -268.     0.   -60.     0.
                            0.   -52. -1360.     0. -1360.     0.  -939.     0.     0.     0.]
                            (40,)
                        """
                        # print(sub_feature.ndim)
                        print("shape before:",sub_feature.shape)
                        if sub_feature.ndim == 1:
                            sub_feature = np.reshape(sub_feature, (integrate_depth, sub_feature.shape[0]))
                        else:
                            # sub_feature = np.reshape(sub_feature, (integrate_depth, int(sub_feature.shape[1]/(window_depth+1))))
                            # 将 sub_feature 分成了 integrate_depth 个子数组，并且每个子数组都具有相同的长度
                            sub_feature = np.reshape(sub_feature, (integrate_depth, -1, sub_feature.shape[1]))
                            sub_feature = sub_feature[:, :length, :]
                        print("shape after:",sub_feature.shape)
                        # print("feature after:",sub_feature)
                        signatures = path_signature(sub_feature)
                        print("signatures:",signatures)

                        flattened_signatures = [value for sublist in signatures for value in sublist]
                        print("flattened_signatures:",flattened_signatures)

                        sub_pathSig_feature.append(flattened_signatures)
                        # 每一个特征签名序列都添加一个label
                        for i in range(len(flattened_signatures)):
                            sub_label.append(label)
        
        # 将当前窗口的特征和标签写入全局
        integrated_pathSig_features.append(sub_pathSig_feature)
        integrated_labels.append(sub_label)
        # print(integrated_pathSig_features[:10])
    
    with open('vpn_metrics_dynamic_window_simple.txt', "w") as file:
        # 循环遍历列表中的每个元素
        for item,label in zip(integrated_pathSig_features, integrated_labels):
            # 将每个元素写入文件，并在末尾添加换行符
            file.write("%s:%s\n" % (item,label))

    return integrated_pathSig_features, integrated_labels


def classify():
    output_file = 'vpn_metrics_dynamic_window.txt'
    result_file = 'vpn_metrics_result.txt'
    features,labels = read_files_and_labels()

    if not os.path.exists(output_file):
        integrated_pathSig_features, integrated_labels = \
                split_and_integrate_hierarchical_dyadic_window(hierarchical_dyadic_window, features, labels)
    else:
        integrated_pathSig_features = []
        integrated_labels = []
        with open('vpn_metrics_dynamic_window.txt', "r") as file:
            for line in file:
                item, label = line.strip().split(':')
                # print(item, label)
                item = eval(item)  # 将字符串转换回原来的数据类型（列表）
                integrated_pathSig_features.append(item)
                integrated_labels.append(label)
    
    # 打印特征的初始状态
    # print("integrated_pathSig_features:",integrated_pathSig_features[:1])
    # print("len(integrated_pathSig_features):",len(integrated_pathSig_features)) # 样本数，5194
    # print("integrated_labels:",integrated_labels)
    # print("len(integrated_labels):",len(integrated_labels)) # 样本数

    # 汇总打印一次
    # print("integrated_pathSig_features[0]:",integrated_pathSig_features[0])
    # print("integrated_labels[0]:",integrated_labels[0])

    """
        预处理结束，开始分类
    """

    shapes = [len(arr) for arr in integrated_pathSig_features]
    # 使用集合获取所有不同的形状
    unique_shapes = set(shapes)
    print("unique_shapes:",unique_shapes) # {1, 12}

    # 训练不同决策树
    # 对于列表中的每个数组，根据其形状将其添加到对应的字典键中
    label_encoder = LabelEncoder()

    # 适配标签并进行转换
    integrated_labels_encoded = label_encoder.fit_transform(integrated_labels)
    # 创建训练数据的形状字典
    shape_dict = {}
    for path_group, label in zip(integrated_pathSig_features, integrated_labels_encoded):
        for path_i in path_group:
            if type(path_i) is list:
                shape = len(path_i)
            else:
                shape = path_i.shape[0] if isinstance(path_i, np.ndarray) else 1
            
            if shape not in shape_dict:
                shape_dict[shape] = {"X": [path_i], "y": [label]}
            else:
                shape_dict[shape]["X"].append(path_i)
                shape_dict[shape]["y"].append(label)


    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42,min_samples_leaf=1)
    
    for shape, data in shape_dict.items():
        X = data["X"]
        y = data["y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # print("len(X_train):",len(X_train)) 
        # print("len(y_train):",len(y_train)) 
        """
            对于1维路径:拆分前4155，拆分后49862
            对于2维路径:拆分后74793
        """
        # print("len(X_train[0]):",len(X_train[0])) # 30
        print("X_train[0]:",X_train[0]) # 30
        # print("y_train[0]:",y_train[0]) # 标识标签的数
        rf_classifier.fit(X_train, y_train)
        details = []
        # 显示每棵树的内部结构
        print("details:")
        for i, tree in enumerate(rf_classifier.estimators_):
            # print(f"Tree {i + 1}:")
            tree_rules = export_text(tree)
            details.append(tree_rules)
            # print(tree_rules)
        print("predict:")
        y_pred = rf_classifier.predict(X_test)
        # 评估模型性能
        # print(y_test)
        # print(y_pred)

        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(conf_matrix)

        with open(result_file, 'a') as file:
            file.write("Accuracy: {:.2f}\n".format(accuracy))
            file.write("Confusion Matrix:\n")
            file.write(np.array2string(conf_matrix, separator=', '))
            # 先把准确率提上去再用这个
            # i=0
            # for tree_detail in details:
            #     file.write("tree_detail{}:{}\n".format(i,tree_detail))
            #     i += 1


def my_test():
    features, labels = read_files_and_labels()
    print(features[:6])
    print(labels[:6])
if __name__ == "__main__":
    classify()
    # my_test()

"""
    对于某一个样本,采样出来一维积分值有12个,二维积分值有18个,对应30个标签
    对每个子区间,有四个特征,每一个特征产生一个一维积分值/积分路径,路径长度为1,一共4个;
                          每两个特征产生一个二维积分路径,路径长度为4个一维积分+6个二维积分+4个二重积分=14,一共6个;
                          为什么文件只有2个二重积分，可能是统计路径特征值的二重积分平方之后越界了，那就先不用
    划分了两层子区间，第一层不划分就一个区间，第二个对半分两个
    (4+6) * 3 = 30 ,所以是对的

    那就是分类器的问题
"""