############
# Model
############
from util._1_FC_based_class import FCClass
from model_basic.resnet import ResNet
E_lr, FC_lr = 1e-4, 1e-4
E_beta, FC_beta = (0.5,0.9), (0.5,0.9)
E_weight_decay, FC_weight_decay = 0.01, 0.01

#################
# Data
#################

y_col_name = "판매량"

# ("category_embedding", embedding_size, min_freq(포함x)) ("category_onehot", min_freq(포함x))
# ("numerical_minmax",) ("numerical", ) ("numerical_embedding", embedding_size)
info = \
{"월": ("category_embedding",20,2), "시간": ("category_embedding",20,2), "방송요일": ("category_embedding",20,2),
 "연속 휴일": ("category_embedding",20,2), "노출(분)_all": ("numerical_minmax",), "상품군": ("category_embedding",20,2),
 "마더코드": ("category_embedding",20,2), "브랜드": ("category_embedding",20,2), "상품코드": ("category_embedding",20,2),
 "판매단가": ("numerical_embedding", 20), "광고 사람": ("category_embedding",20 ,2),
 "국내생산": ("category_embedding",20 ,2), "기온": ("numerical_embedding", 20), "강수량": ("numerical_embedding", 20),
 "vs":  ("numerical_embedding", 20), "lcsch": ("numerical_embedding", 20), "dc10tca": ("numerical_embedding", 20),
 "icsr": ("numerical_embedding", 20), "ss": ("numerical_embedding", 20), "pa": ("numerical_embedding", 20),
 "pv": ("numerical_embedding", 20), "hm": ("numerical_embedding", 20), "ws": ("numerical_embedding", 20),
 "morning_drama": ("category_embedding",20,2), "popular_program": ("category_embedding",20,2), "동시방송 상품개수": ("category_embedding",20,2),
 "Seq 방송 개수": ("category_embedding",20,2), "성별": ("category_embedding",20,2), "일시불/무이자": ("category_embedding",20,2),
 "하루방송 수" : ("category_embedding",20,2)}
drop = ["노출(분)_all"]

embedding_set = {"날씨": ["기온", "강수량", "vs", "lcsch", "dc10tca", "icsr", "ss", "pa", "pv", "hm", "ws"],
                 "상품": ['마더코드', '상품군', '브랜드', '상품코드'],
                 "상품기타": ['성별', '국내생산','광고 사람','판매단가', '일시불/무이자'],
                 "시간": ['방송요일', '연속 휴일', 'popular_program', 'Seq 방송 개수', 'morning_drama', '시간', '월', '동시방송 상품개수', "하루방송 수"]}

reg_coef = {'월': 0.01, '시간': 0.01, '방송요일': 0.01, '연속 휴일': 0.01, '상품군': 0.01, '마더코드': 0.01, '브랜드': 0.01,
            '상품코드': 0.01, '판매단가': 0.01, '광고 사람': 0.01, '국내생산': 0.01, '기온': 0.01, '강수량': 0.01, 'vs': 0.01,
            'lcsch': 0.01, 'dc10tca': 0.01, 'icsr': 0.01, 'ss': 0.01, 'pa': 0.01, 'pv': 0.01, 'hm': 0.01, 'ws': 0.01,
            'morning_drama': 0.01, 'popular_program': 0.01, '동시방송 상품개수': 0.01, 'Seq 방송 개수': 0.01, '성별': 0.01,
            '일시불/무이자': 0.01, "하루방송 수" : 0.01}

coef = {'월': 1, '시간': 1, '방송요일': 1, '연속 휴일': 1, '상품군': 1, '마더코드': 1, '브랜드': 1, '상품코드': 1, '판매단가': 1, '광고 사람': 1,
        '국내생산': 1, '기온': 1, '강수량': 1, 'vs': 1, 'lcsch': 1, 'dc10tca': 1, 'icsr': 1, 'ss': 1, 'pa': 1, 'pv': 1, 'hm': 1, 'ws': 1,
        'morning_drama': 1, 'popular_program': 1, '동시방송 상품개수': 1, 'Seq 방송 개수': 1, '성별': 1, '일시불/무이자': 1, "하루방송 수" : 1}
concat = True
num = "80_50"
lower_sim = 0.7
rank_sim_train = 2# 상위 몇개 까지 train할 것
max_y = 10000

# Fix
GPU_NUM = 2 # 원하는 GPU 번호 입력
random_state = 777
save_loc = "/home/bigdyl/jayoung/tmp/result/last"
save_file_name = "original.txt"
validation_ratio = 0.01 / 350
EPOCH = 10000
BATCH_SIZE = 10000

#################
# Init
#################
import torch
import warnings, random, os, datetime
from torch.utils.tensorboard import SummaryWriter
if not(os.path.isdir(save_loc)):
    os.makedirs(save_loc)
save_file = open(save_loc + "/" + save_file_name,"w")

print(str(datetime.datetime.now()))
save_file.write(str(datetime.datetime.now()) + "\n")
writer = SummaryWriter()
warnings.filterwarnings(action='ignore')
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print ('Current cuda device ', torch.cuda.current_device())
random.seed(random_state); torch.manual_seed(random_state); torch.cuda.manual_seed_all(random_state)


model = FCClass(num, y_col_name, info, max_y, embedding_set, concat, validation_ratio, drop, lower_sim,
                rank_sim_train, BATCH_SIZE, device)
model.fit(ResNet, EPOCH, E_lr, FC_lr, E_beta, FC_beta,E_weight_decay, FC_weight_decay, reg_coef, coef, random_state, writer, save_file,save_loc)
save_file.close()