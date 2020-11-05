#############
# IMPORT
#############
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def make_data_set(info, y_col_name, max_y, device, num, lower_sim, train_validation_ratio):
    name_code = pd.read_csv("data/extra_data/name_code.csv", index_col=0)
    sim_matrix = pd.read_csv("data/extra_data/name_sim.csv", index_col=0).values
    sim_matrix[sim_matrix < lower_sim] = 0
    sim_matrix[list(range(len(sim_matrix))),list(range(len(sim_matrix)))] = 0
    train_data = pd.read_csv("data/preprocessed_data/preprocess2_train{}.csv".format(num), index_col=0)
    test_data = pd.read_csv("data/preprocessed_data/preprocess2_test{}.csv".format(num), index_col=0)

    final_json = []
    train_price = torch.Tensor(train_data["판매단가"]).view(-1, 1).to(device)
    test_price = torch.Tensor(test_data["판매단가"]).view(-1, 1).to(device)
    train_revenue = torch.Tensor(train_data["취급액"]).view(-1, 1).to(device)
    train_name = torch.LongTensor(train_data["상품명_띄어쓰기_set"]).view(-1, 1).to(device)
    test_name = torch.LongTensor(test_data["상품명_띄어쓰기_set"]).view(-1, 1).to(device)

    name = y_col_name
    train_data[name] = train_data[name].astype(float)
    y_Min, y_Max = train_data[name].min(), train_data[name].max()
    over_y_max = list((train_data.index[train_data[name] > max_y]))
    y_Max = np.min(np.array([max_y, y_Max]))
    train_data[name] = (train_data[name] - y_Min) / (y_Max - y_Min)
    train_data[name] = 2 * train_data[name] - 1
    final_json.append({"name": name, "info": {"min": y_Min, "max": y_Max}, "type": "y_value"})

    real_train_index, valid_index = _train_validation_index(train_data, train_validation_ratio, over_y_max)

    for name in info:
        if info[name][0] == "numerical":
            train_data[name] = train_data[name].astype(float)
            test_data[name] = test_data[name].astype(float)
            final_json.append({"name": name, "type": "numerical"})

        elif info[name][0] == "numerical_minmax":
            train_data[name] = train_data[name].astype(float)
            test_data[name] = test_data[name].astype(float)
            Min, Max = min(train_data[name].min(), test_data[name].min()), max(train_data[name].max(), test_data[name].max())
            train_data[name] = (train_data[name] - Min) / (Max - Min)
            train_data[name] = 2 * train_data[name] - 1
            final_json.append({"name" : name, "info" : {"min": Min, "max": Max}, "type" : "numerical_minmax"})

        elif info[name][0] == "numerical_embedding":
            train_data[name] = train_data[name].astype(float)
            test_data[name] = test_data[name].astype(float)
            tmp = pd.concat([train_data[name], test_data[name]])
            mean, std = tmp.mean(), tmp.std()
            train_data[name] = (train_data[name] - mean) / std
            test_data[name] = (test_data[name] - mean) / std
            final_json.append({"name": name, "embedding_size" : info[name][1], "type": "numerical_embedding"})

        elif info[name][0] == "category_onehot":
            train_data[name] = train_data[name].astype(str)
            test_data[name] = test_data[name].astype(str)
            tmp = train_data.loc[real_train_index,name].value_counts()
            all_category = list(set(tmp.index[tmp > info[name][1]]) - {"nan"})
            dic = dict(zip(all_category, range(1, len(all_category) + 1)))
            train_data[name] = train_data[name].map(dic).fillna(0).astype(int)
            test_data[name] = test_data[name].map(dic).fillna(0).astype(int)
            if 0 not in (set(train_data[name].unique()) | set(test_data[name].unique())):
                train_data[name] = train_data[name] - 1
                test_data[name] = test_data[name] - 1
            final_json.append({"name": name, "item": sorted(list(set(train_data[name].unique()) | set(test_data[name].unique()))),
                               "type": "category_onehot"})

        elif info[name][0] == "category_embedding":
            train_data[name] = train_data[name].astype(str)
            test_data[name] = test_data[name].astype(str)
            tmp = train_data.loc[real_train_index,name].value_counts()
            all_category = list(set(tmp.index[tmp > info[name][2]]) - {"nan"})
            dic = dict(zip(all_category, range(1, len(all_category) + 1)))
            train_data[name] = train_data[name].map(dic).fillna(0).astype(int)
            test_data[name] = test_data[name].map(dic).fillna(0).astype(int)
            final_json.append({"name": name, "item": sorted(list(range(len(all_category) + 1))), "type": "category_embedding",
                               "embedding_size" : info[name][1]})
            if name in ["상품군", "마더코드", "상품코드"]:
                name_code[name] = name_code[name].astype(str)
                name_code[name] = name_code[name].map(dic).fillna(0).astype(int)
            if name == "상품코드":
                name_code = name_code[name_code["상품코드"] != 0]
                sim_matrix = sim_matrix[:,list(name_code.index)]
        else:
            raise Exception('type을 잘못 입력했습니다. "category_embedding", "category_onehot", "numerical_minmax", '
                            '"numerical_embedding", "numerical", ""중 하나를 입력해야합니다.')
    return (train_data, test_data, name_code.reset_index(drop = True), sim_matrix, final_json, train_price, test_price, train_revenue,
            train_name, test_name, real_train_index, valid_index, y_Max, y_Min)

def make_dataset_embedding(data, meta_data, info, embedding_set, drop, device, concat, data_type, before_x_embedding = None):
    if data_type in ["valid", "test"] and before_x_embedding == None:
        assert 0 , "valid, test 데이터 일때는 x_embedding 인자 넣어야 합니다."
    _check_category_set(info, embedding_set, concat)
    total_input_size = 0
    x_val_pos = 0
    x_embedding = {}
    x_val = []
    check = dict(zip(list(embedding_set.keys()) + ["concat"], [True] * (len(embedding_set) + 1)))
    all_column_info = {}

    for meta in meta_data:
        if meta["name"] in drop:
            continue
        if  (data_type != "test") and (meta["type"] == "y_value"):
            tmp = data[meta["name"]].values
            y_val = torch.Tensor(tmp).view(-1, 1)
        elif meta["type"] in ["numerical", "numerical_minmax"]:
            tmp = data[meta["name"]].values
            x_val.append(torch.Tensor(tmp).view(-1, 1))
            total_input_size += 1
            all_column_info[meta["name"]] = list(range(x_val_pos, x_val_pos + 1)) ; x_val_pos += 1
        elif meta["type"] == "category_onehot":
            tmp = data[meta["name"]].astype(int).values
            tmp = np.eye(len(meta['item']), dtype=int)[tmp, 1:]
            x_val.append(torch.Tensor(tmp))
            total_input_size += len(meta['item']) - 1
            all_column_info[meta["name"]] = list(range(x_val_pos, x_val_pos + len(meta['item']) - 1)); x_val_pos += len(meta['item']) - 1
        elif meta["type"] == "numerical_embedding":
            tmp = data[meta["name"]].values
            if data_type in ["valid", "test"]:
                x_embedding[meta["name"]] = {"model_basic": before_x_embedding[meta["name"]]["model_basic"], "data": torch.Tensor(tmp).view(-1, 1).to(device)}
            else:
                x_embedding[meta["name"]] = {"model_basic": nn.Linear(1, meta['embedding_size'], bias=False).to(device),
                                             "data": torch.Tensor(tmp).view(-1, 1).to(device)}

            if _check_set(meta["name"],embedding_set) == "concat" or check[_check_set(meta["name"],embedding_set)] == True or concat:
                total_input_size += meta['embedding_size']
                check[_check_set(meta["name"], embedding_set)] = False

        elif meta["type"] == "category_embedding":
            tmp = data[meta["name"]].astype(int).values
            tmp = np.eye(len(meta['item']), dtype=int)[tmp, 1:]
            if data_type in ["valid", "test"]:

                x_embedding[meta["name"]] = {"model_basic": before_x_embedding[meta["name"]]["model_basic"], "data": torch.Tensor(tmp).to(device)}
            else:
                x_embedding[meta["name"]] = {"model_basic": nn.Linear(len(meta['item']) - 1, meta['embedding_size'], bias=False).to(device),
                                             "data": torch.Tensor(tmp).to(device)}
            if _check_set(meta["name"],embedding_set) == "concat" or check[_check_set(meta["name"], embedding_set)] == True or concat:
                total_input_size += meta['embedding_size']
                check[_check_set(meta["name"], embedding_set)] = False

    if len(x_val) != 0:
        x_val = torch.cat(x_val, axis=1)
    if data_type == "test":
        print("total_input_size:", total_input_size)
        return (all_column_info, total_input_size, x_embedding, x_val)
    else:
        return (all_column_info, total_input_size, x_embedding, x_val, y_val)



def _check_set(name, embedding_set):
    for set_name in embedding_set:
        if name in embedding_set[set_name]:
            return set_name
    return "concat"

def _check_category_set(info, embedding_set, concat):
    if concat:
        return
    for set_name in embedding_set:
        tmp = []
        for name in embedding_set[set_name]:
            if "embedding" not in info[name][0]:
                raise Exception("embedding_set이 잘못 지정돼 있습니다.")
            else:
                tmp.append(info[name][1])
        if len(set(tmp)) != 1:
            raise Exception("embedding_set이 잘못 지정돼 있습니다.")

def _train_validation_index(data, valid_ratio, over_y_max):
    import random
    total_length = len(data)
    total_index = list(set(range(total_length)) - set(over_y_max))
    random.shuffle(total_index)
    test_index = total_index[: int(total_length * valid_ratio)] + list(over_y_max)
    train_index = total_index[int(total_length * valid_ratio):]
    return train_index, test_index

