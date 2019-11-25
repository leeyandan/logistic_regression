#%%
import numpy as np
#%%
class Logistic_regression:
    '模型所使用的数据集格式：x是属性值集合,x.shape=(N, D)。y是标记{0，1}, y.shape=(N)：'
    def __init__(self, attr_num):
        self.w = np.random.random((attr_num))
    
    def reset_model(self):
        '重置模型参数'
        self.w = np.random.random(self.w.shape)

    def train(self, x, y, learning_rate, stop_interval=0.1):
        '给定数据（x是属性值集合,x.shape=(N, D)。y是标记{0，1}, y.shape=(N)）训练到暂停间隙值stop_interval，学习率为learning_rate'
        last_ml = self.__cal_maximum_likehood(x, y)
        N = x.shape[0]
        train_step = last_ml = 0 
        while True:
            train_step += 1
            cur_ml = self.__cal_maximum_likehood(x, y)
            interval = abs(last_ml-cur_ml)
            right = self.evalue_model(x, y)
            #每十步打印一次信息
            if train_step%10==0:
                print("step:{} cur_max_likehood:{:.3f} up_abs:{:.3f} right:{}/{}={:.3f}".format(train_step, cur_ml, cur_ml-last_ml, right, N, right/N))
            last_ml =cur_ml
            #更新系数
            self.w = self.w + learning_rate*self.__cal_derivative(x,y)
            if interval<stop_interval or train_step>5000:
                print("train_finish!")
                break 

    def __cal_maximum_likehood(self, x, y):
        '计算对数最大似然估计值'
        t = (x@self.w)
        temp = y*t-np.log(1+np.exp(t))
        return np.sum(temp)
        
    def __cal_derivative(self, x, y):
        '计算导数值'
        temp =(y-1.0/(1+np.exp(-(x@self.w))))
        repeat_times = x.shape[1]
        #重复填充使得维度相同可以相乘
        re_temp = np.repeat(temp.reshape((temp.shape[0],1)), repeat_times, axis=1)
        ans_temp = re_temp*x
        ans = np.sum(ans_temp, axis=0)
        return ans

    def evalue_model(self, x, y):
        '评价模型，预测对了多少（x是属性值集合,x.shape=(N, D)。y是标记{0，1}, y.shape=(N)）'
        N = y.shape[0]
        p1 = 1.0/(1.0+np.exp(-(x@self.w)))
        y_predict = np.zeros(y.shape)
        y_predict[np.where(p1>=0.5)] = 1
        #np.where 返回回一个元组，包裹着下标对象
        # ((array([0, 1, 2], dtype=int64),)
        right_list = np.where(y==y_predict)[0]
        return len(right_list)
        
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
    '将数据和标签分割开，并且将x尾部填充1'
    x = data[:,:-1]
    y = data[:,-1]
    ones = np.ones(x.shape[0])
    #尾部填充0
    x = np.column_stack([x, ones])
    #y标签2改为0
    y[np.where(y==2)] = 0
    return x,y

def ten_times_test(lr_model, data_splited, learning_rate=0.0002):
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
    magic_data = r"data\Magic Dataset.txt"
    data = np.loadtxt(magic_data)
    data_splited = split_data(data)
    lr = Logistic_regression(11)
    ten_times_test(lr, data_splited)
    
#能达到的训练精度和学习率有关
# %%
