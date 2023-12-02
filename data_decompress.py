"""
    CASIA-AHCDB 数据集解压缩
"""
import os
import numpy as np
from PIL import Image
# 乱码对应的unicode码相对应的中文
Json_list = {
    '2d503':'𭔃', '2a9d0':'𪧐', '2b74a':'𫝊', '2b755':'𫝕', '2bfa3':'𫾣',
    '2c4b3':'𬒳', '2d0e1':'𭃡', '2d4a1':'𭒡', '2d8e3':'𭣣', '2d9c2':'𭧂', '2d081':'𭂁',
    '2d503':'𭔃', '2d539':'𭔹', '2d818':'𭠘', '2dc17':'𭰗', '2de1f':'𭸟', '2e2b5':'𮊵',
    '2e4e1':'𮓡', '2e5a0':'𮖠','2e7b8':'𮞸', '2e7c3':'𮟃', '2e509':'𮔉', '2ea8d':'𮪍',
    '2f9b5':'虧', '2f91b':'𠔥', '2f963':'築', '20a64':'𠩤', '20ba5':'𠮥','20bb7':'𠮷',
    '20d4a':'𠵊', '21a56':'𡩖', '21e01':'𡸁', '21ed5':'𡻕', '22fbe':'𢾾', '23a75':'𣩵',
    '23d11':'𣴑', '23f1b':'𣼛', '23f43':'𣽃', '24c1e':'𤰞', '24f32':'𤼲', '25a25':'𥨥',
    '25b07':'𥬇', '26f94':'𦾔', '28d92':'𨶒', '28ffd':'𨿽', '29a59':'𩩙', '30a6c':'𰩬',
    '30a76':'𰩶', '202e3':'𠋣', '212ae':'𡊮', '216b1':'𡚱', '219f1':'𡧱', '235f3':'𣗳',
    '248aa':'𤢪', '254d3':'𥓓', '256d1':'𥛑', '260b6':'𦂶', '261b5':'𦆵',
    '284dc':'𨓜', '286ab':'𨚫', '2004a':'𠁊', '2315c':'𣅜', '2505e':'𥁞', '2899f':'𨦟',
    '20158':'𠅘', '21596':'𡖖', '22544':'𢕄', '22662':'𢙢', '23780':'𣞀', '24360':'𤍠',
    '25609':'𥘉', '25677':'𥙷', '25807':'𥠇', '28427':'𨐧', '29516':'𩔖', '2b826': '𫠦'
}

# 路径为存放数据集解压后的.gntx文件
dataset_dir = '../data/datasets/CASIA-AHCDB/'

# CASIA数据集 将.gtnx后缀格式文件转化为png
# 保存unicode和中文的字典文件
def save_file(data, name):
    import pickle
    f = open(os.path.join(dataset_dir,name), 'wb')
    pickle.dump(data, f)
    f.close()

# 读取gntx目录
def read_from_gntx_dir(gntx_dir):
    def one_file(f):
        #头大小为12
        header_size = 12
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            # print('header:',header)
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            #Unicode = header[7] + (header[6] << 24)+ (header[5] << 16)+ (header[4] << 8)
            # Unicode =  hex(header[5]).strip('0x') + hex(header[4]).strip('0x')
            # Unicode = '\\u' + Unicode
            Unicode = header[4] + (header[5] << 8)+ (header[6] << 16) + (header[7] << 24)
            width = header[8] + (header[9] << 8)
            height = header[10] + (header[11] << 8)

            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            yield image, Unicode

    for file_name in os.listdir(gntx_dir):
        if file_name.endswith('.gntx'):
            file_path = os.path.join(gntx_dir, file_name)
            print("正在加载：{}".format(file_name))
            with open(file_path, 'rb') as f:
                for image, Unicode in one_file(f):
                    yield image, Unicode

# CASIA数据集 将.gtnx后缀格式文件转化为png
def gntx_to_png():
    char_undict = {}
    char_set = set()
    files = os.listdir(dataset_dir)
    for path in files:
        classname = 'train' if 'train' in path else 'test'
        classdir = os.path.join(dataset_dir, classname)
        print(classdir)
        data_dir = os.path.join(dataset_dir, path)
        counter = 0
        for image, Unicode in read_from_gntx_dir(gntx_dir=data_dir):
            # 转化为16进制再转化为unicode字符
            # Unicode在这里是int型
            # if(Unicode<0x1000 or Unicode>0xffff):
            #     continue
            temp = "\\u" + hex(Unicode)[2:]
            unicode = hex(Unicode)[2:]  # 十六进制
            if len(unicode) == 4:
                Unicode_unicode = temp.encode('utf-8').decode('unicode_escape')  # ‘中文字符’
            else:
                Unicode_unicode = Json_list[unicode]
            #print(temp, unicode, Unicode_unicode)
            char_set.add(Unicode_unicode)
            if unicode not in char_undict:
                char_undict[unicode] = Unicode_unicode
            
            im = Image.fromarray(image)
            # 存放图片的路径
            img_dir = classdir + '/' + unicode

            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            im.convert('L').save(img_dir + '/' + str(counter) + '.png')
            if (counter % 5000 == 0):
                print("counter=", counter)
            counter += 1
        
        print(classname + " counter = " + str(counter))
        print(classname + ' transformation finished ...')
        
    char_list = list(char_set)
    char_dict = dict(zip(sorted(char_list), range(len(char_list))))
    print("char_dict=", char_dict)
    print("char_undict = ", char_undict)
    save_file(char_dict, 'char_dict')
    save_file(char_undict, 'char_undict')


gntx_to_png()    
    
# 修改文件名，有些文件是乱码，所以设置了unicode码，现在根据修改文件名，把unicode改成汉字
def change_dir_name(data_dir):
    files = os.listdir(data_dir)
    for names in files:
        if len(names) > 1:
            new_names = Json_list[names]
        old_file_name = os.path.join(data_dir, names)
        new_file_name = os.path.join(data_dir, new_names)
        print(names, ' to ', new_names)
        os.rename(old_file_name, new_file_name)

# change_dir_name(os.path.join(data_dir,'style1_basic_train_part1'))
# change_dir_name(os.path.join(data_dir,'style1_basic_test'))
