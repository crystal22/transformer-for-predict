import numpy as np
test_final='/Users/guojianzou/Downloads/byte_cup/test/final.txt'
train_final='/Users/guojianzou/Downloads/byte_cup/train/final.txt'

save_train='/Users/guojianzou/Downloads/byte_cup/save_train.txt'
save_test='/Users/guojianzou/Downloads/byte_cup/save_test.txt'

each_feature_num=[73974, 396, 4122689, 850308, 461, 5, 89778, 75085, 8735, 10000, 641]

time={'max': 53087342732.0, 'min': 53015151211.0}

dictionary={'uid': 73974, 'user_city':396 , 'item_id': 4122689, 'author_id': 850308,
 'item_city': 461, 'channel': 5, 'music_id': 89778, 'device': 75085, 'duration_time': 641}

feature_start=[0, 73974, 74370, 4197059, 5047367, 5047828, 5047833, 5137611, 5212696, 5221431, 5231431, 5232072]

feature_length=5232072

feature_num=11

def train_set(train_final,save_train):
    length = 12
    read_file=open(train_final,'r',encoding='utf-8')
    line=read_file.readline()
    step=1
    with open(save_train, 'w', encoding='utf-8') as write:
        while(line):
            data = []
            data_dict = {}
            Line=line.strip().split('\t')
            if Line!='':
                data_dict['uid']=Line[0]
                data_dict['item_id']=Line[2]
                data_dict['finish']=Line[6]
                data_dict['like']=Line[7]
                for i in range(length):
                    if i==6 or i==7:
                        continue
                    else:
                        if i==10:
                            data.append(Line[i][3:7])
                            data.append(Line[i][7:11])
                        else:
                            data.append(Line[i])
            data_dict['feature']=data
            write.write(str(data_dict)+'\n')
            print('Has been read :'+str(step)+' lines')
            step+=1
            line=read_file.readline()
#train_set(train_final,save_train)
# read_file=open(save_train,'r',encoding='utf-8')
# line=read_file.readlines()
# print(len(line))

class DataParser(object):
  """
  Detailed operator foe line input
  """
  @staticmethod
  def data_parser(content):
    """ parser line content and generate idx, features, and gts """
    features = content[:feature_num]
    features = list(map(lambda feature: np.float32(feature), features))
    idx = [0 if feature < 0 else feature for feature in features]
    features = [np.float32(0) if feature < 0 else np.float32(1) for feature in features]

    shifts = feature_start
    idx = [idx[i] + shifts[i] for i in range(len(idx))]

    idx= list(map(lambda one_id: np.int32(one_id), idx))
    return idx, features

def sparse_one_hot(index):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param array_length: length of one-hot representation
    :return: one-hot representation, type of np.array
    """
    array = np.zeros([feature_length])
    for field in range(feature_num):
        # get index of array
        array[index] = 1
    return array

def train_next(batch_size=64):
    file=open(save_train,'r',encoding='utf-8')
    line=file.readline()
    sparse_one=[]
    label = []
    feature_index=[]
    feature=[]
    while line:
        Line = eval(line.strip('\n'))
        if Line != '':
            if len(Line['feature']) != 11:
                line = file.readline()
                continue
            else:
                f_index,feat=DataParser.data_parser(Line['feature'])
                sparse_one.append(sparse_one_hot(f_index))
                feature_index.append(f_index)
                feature.append(feat)
                # inputs.append(Line['feature'])
                label.append(int(Line['finish'].strip()))
        if len(label) == batch_size:
            sparse_one=np.array(sparse_one)
            feature_index=np.array(feature_index)
            feature=np.array(feature)
            label = np.reshape(np.array(label),[-1,1])
            yield (sparse_one,label,feature_index,feature)
            sparse_one = []
            label = []
            feature_index = []
            feature = []
        line = file.readline()

def test_next(batch_size=64):
    file=open(save_test,'r',encoding='utf-8')
    line=file.readline()
    sparse_one=[]
    label = []
    feature_index=[]
    feature=[]
    while line:
        Line = eval(line.strip('\n'))
        if Line != '':
            if len(Line['feature']) != 11:
                line = file.readline()
                continue
            else:
                f_index,feat=DataParser.data_parser(Line['feature'])
                sparse_one.append(sparse_one_hot(f_index))
                feature_index.append(f_index)
                feature.append(feat)
                # inputs.append(Line['feature'])
                label.append(int(Line['finish'].strip()))
        if len(label) == batch_size:
            sparse_one=np.array(sparse_one)
            feature_index=np.array(feature_index)
            feature=np.array(feature)
            yield (sparse_one,feature_index,feature)
            sparse_one = []
            label = []
            feature_index = []
            feature = []
        line = file.readline()
# for sparse_ones,labels,feature_index,features in train_next(64):
#     print(sparse_ones[0])