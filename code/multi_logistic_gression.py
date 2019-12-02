#%%
import numpy as np
#%%
class Multi_logistic_regression:
    '模型所使用的数据集格式：x是属性值集合,x.shape=(N, D)。y是标记{0，1}, y.shape=(N)：'
    def __init__(self, class_num, attr_num):
        '给出预测类数量以及属性数量来初始化参数'
        #最后一类概率可以由K-1类给出，故class_num-1。偏置由最后一维给出，故attr+1
        self.w = np.random.random((class_num-1, attr_num+1))
        self.class_num = class_num
    
    def reset_model(self):
        '重置模型参数'
        self.w = np.random.random(self.w.shape)

    def train(self, x, y, learning_rate, stop_interval=0.001):
        '给定数据（x是属性值集合,x.shape=(N, D)。y是标记{0，1}, y.shape=(N)）训练到暂停间隙值stop_interval，学习率为learning_rate'
        #last_ml = self.__cal_maximum_likehood(x, y)
        N = x.shape[0]
        train_step = 0 
        while True:
            train_step += 1
            if train_step%50==0:
                right = self.evalue_model(x, y)
                acc = right/N
                print("step:{} right:{}/{}={:.3f}".format(train_step, right, N, acc))
                # print('step:{} w_shape:{}'.format(train_step, self.w.shape))
            # last_ml =cur_ml
            #更新系数
            for q in range(self.class_num-1):
                self.w[q] = self.w[q] + learning_rate*self.__cal_derivative(x,y,q)
            if train_step>3000:
                print("train_finish!")
                break 
        
    def __cal_derivative(self, x, y, q):
        '计算似然函数对于第q维权重的导数值，'
        deri = np.zeros(self.w[q].shape)
        for attr, label in zip(x,y):
            #示性函数值
            indication = int(label==q)
            #公式第二部分的分母值
            denominator = 1+np.sum(np.exp(self.w@attr))
            deri += indication*attr - np.exp(self.w[q]@attr)*attr/ denominator
        return deri

    def predict(self, attr):
        '给出属性预测所属的类'
        #分母都一样，故比较分子就可以了，k-1 个分子值
        numberator = np.exp(self.w@attr)
        #K个分子值
        K_numberator = np.append(numberator,1)
        pre_class = np.argmax(K_numberator)
        return pre_class
    
    def evalue_model(self, x, y):
        '评价模型，预测对了多少（x是属性值集合,x.shape=(N, D)。y是标记{0，1}, y.shape=(N)）'
        N = y.shape[0]
        right_count = 0
        for attr,label in zip(x, y):
            pre_class = self.predict(attr)
            if label==pre_class:
                right_count +=1
        return right_count
        
#%%

def split_data(data, unit=10):
    '将数据按类比例分割为unit份，默认十份，用十折验证法'
    labels = set(data[:,-1])
    label_data = []
    #按类别分出数据
    for label in labels:
        t_data = data[np.where(data[:,-1]==label)]
        label_data.append(t_data)
    data_splited =[[] for i in range(unit)]
    for onedata in label_data:
        total = onedata.shape[0]
        part_num = total//unit
        for i in range(unit):
            #最后一次全部放在一起
            if i==unit-1:
                data_splited[i].append(onedata[i*part_num:])
            else:
                data_splited[i].append(onedata[i*part_num:(i+1)*part_num])
    for i in range(unit):
        #行合并
        data_splited[i] = np.row_stack(data_splited[i])
    return data_splited

def prepare_data_x_y(data):
    '将数据和标签分割开，并且将x尾部填充1，标签要都-1'
    x = data[:,:-1]
    y = data[:,-1]-1
    ones = np.ones(x.shape[0])
    #尾部填充0
    x = np.column_stack([x, ones])
    return x,y

def ten_times_test(lr_model, data_splited, learning_rate=0.001):
    '十折测试法对模型进行测试'
    #准确率列表
    acc_list = []
    for i in range(len(data_splited)):
        print("----------",i,"------------")
        temp_data = data_splited.copy()
        test_data = temp_data.pop(i)
        train_data = np.row_stack(temp_data)
        train_x, train_y = prepare_data_x_y(train_data)
        test_x, test_y = prepare_data_x_y(test_data)
        lr_model.train(train_x, train_y, learning_rate, stop_interval=0.002)
        test_right = lr_model.evalue_model(test_x, test_y)
        test_len = test_x.shape[0]
        #重置模型进行下一次试验
        lr_model.reset_model()
        print("test:{}/{}= {:.3f}".format(test_right,test_len, test_right/test_len))
        acc_list.append(test_right/test_len)
    print("********** avg-accuracy: %.3f ************"%np.average(acc_list))

#%%
if __name__ == "__main__":

    iris_data = r"data\ecoli-3c.txt"
    data = np.loadtxt(iris_data)
    #种类数量
    class_num = len(set(data[:,-1]))
    #属性数量
    attr_num = data.shape[1]-1
    lr = Multi_logistic_regression(class_num, attr_num)
    #数据分隔为9份
    data_splited = split_data(data)
    ten_times_test(lr, data_splited)
    
#能达到的训练精度和学习率有关
# %%
