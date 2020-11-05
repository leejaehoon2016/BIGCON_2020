import torch
import numpy as np
import pandas as pd
from data.data_preprocessing import _check_set

def cal_MAPE(hat_y, revenue, price, y_max, y_min, revenue_bool):
    y_real = revenue
    if revenue_bool:
        hat_y_real = (((hat_y + 1) / 2) * (y_max - y_min) + y_min)
    else:
        hat_y_real = (((hat_y + 1) / 2) * (y_max - y_min) + y_min) * price
    result_loss = torch.mean(torch.abs((y_real - hat_y_real)) / y_real)
    return y_real, hat_y_real, result_loss

def prepare_sim_matrix(sim_matrix, rank_sim_train, meta_data, name_code, device):
    sim_matrix_ratio, sim_matrix_num = torch.tensor(sim_matrix).sort(dim=1, descending=True)
    sim_matrix_ratio, sim_matrix_num = np.array(sim_matrix_ratio[:, :rank_sim_train]), np.array(sim_matrix_num[:,:rank_sim_train])
    for meta in meta_data:
        if meta["name"] == "상품군":
            meta1 = len(meta["item"])
        elif meta["name"] == "마더코드":
            meta2 = len(meta["item"])
        elif meta["name"] == "상품코드":
            meta3 = len(meta["item"])
    last_result = []
    for i in range(rank_sim_train):
        ratio, num = sim_matrix_ratio[:,[i]], sim_matrix_num[:,i]
        dic1, dic2, dic3 = dict(name_code["상품군"]), dict(name_code["마더코드"]), dict(name_code["상품코드"])
        func1,func2,func3 = np.vectorize(lambda x: dic1[x]), np.vectorize(lambda x: dic2[x]), np.vectorize(lambda x: dic3[x])
        tmp1, tmp2, tmp3 = func1(num), func2(num), func3(num)
        last_result.append({"상품군": torch.Tensor(np.eye(meta1)[tmp1, 1:]).to(device),
                            "마더코드": torch.Tensor(np.eye(meta2)[tmp2, 1:]).to(device),
                            "상품코드": torch.Tensor(np.eye(meta3)[tmp3, 1:]).to(device),
                            "유사도": torch.Tensor(ratio).to(device)})
    return last_result

def make_embedding(embedding_set, x_embedding, index, device, reg_coef, coef, concat, name_column, sim_good, num, train):
    embedded_values = []
    reg_values = []
    for name, val in x_embedding.items():
        if name in ["상품군", "마더코드", "상품코드"] and num >= 0:
            tmp_index = name_column[index].view(-1)
            value = val["model_basic"](sim_good[num][name][tmp_index]) * coef[name] * (sim_good[num]["유사도"][tmp_index])
            embedded_values.append(value)
            reg_values.append((value ** 2) * reg_coef[name] )
        elif name == "브랜드" and train == True:
            input_val = val["data"][index].to(device)
            input_val[torch.randperm(len(input_val))[:len(input_val) // 2]] = 0
            value = val["model_basic"](input_val) * coef[name]
            embedded_values.append(value)
            reg_values.append((value ** 2) * reg_coef[name])
        else:
            value = val["model_basic"](val["data"][index].to(device)) * coef[name]
            embedded_values.append(value)
            reg_values.append((value ** 2) * reg_coef[name])



    reg_values = torch.cat(reg_values, dim=1)
    regularizer = reg_values.mean()
    if concat == True:
        embedded_value = torch.cat(embedded_values, dim=1)
    else:
        tmp = {}
        concat = []
        for i, name in enumerate(x_embedding.keys()):
            set_name = _check_set(name, embedding_set)
            if set_name == "concat":
                concat.append(embedded_values[i])
            elif set_name in tmp:
                tmp[set_name].append(embedded_values[i])
            else:
                tmp[set_name] = [embedded_values[i]]
        embedded_value = torch.cat([sum(val) / len(val) for val in tmp.values()] + concat,dim = 1)
    return embedded_value, regularizer

def make_embedding_for_test(embedding_set, x_embedding, device, coef, concat, name_column, sim_good):
    embedded_values = []
    for name, val in x_embedding.items():
        if name in ["상품군", "마더코드", "상품코드"]:
            input_value = val["data"][:]
            index0 = (input_value.sum(dim = 1) == 0)
            input_value[index0] = sim_good[0][name][name_column[index0].view(-1)]
            value = val["model_basic"](input_value) * coef[name]
            value[index0] = value[index0] *(sim_good[0]["유사도"][name_column[index0].view(-1)])
            embedded_values.append(value)
        else:
            value = val["model_basic"](val["data"].to(device)) * coef[name]
            embedded_values.append(value)

    if concat == True:
        embedded_value = torch.cat(embedded_values, dim=1)
    else:
        tmp = {}
        concat = []
        for i, name in enumerate(x_embedding.keys()):
            set_name = _check_set(name, embedding_set)
            if set_name == "concat":
                concat.append(embedded_values[i])
            elif set_name in tmp:
                tmp[set_name].append(embedded_values[i])
            else:
                tmp[set_name] = [embedded_values[i]]
        embedded_value = torch.cat([sum(val) / len(val) for val in tmp.values()] + concat,dim = 1)
    return embedded_value



def prepare_x_embedding_for_opt(x_embedding, device, name_column, sim_good, good_id):
    id_index = torch.LongTensor([int(i) for i in good_id["index"].str.cat(sep = ",").split(",")])
    opt_x_embedding = {}
    for name, val in x_embedding.items():
        if name in ["상품군", "마더코드", "상품코드"]:
            input_value = val["data"][:]
            index0 = (input_value.sum(dim = 1) == 0)
            input_value[index0] = sim_good[0][name][name_column[index0].view(-1)]

            opt_x_embedding[name] = {"model_basic": val["model_basic"].to("cpu"), "data": input_value[id_index].cpu()}
        elif name in ["브랜드", "판매단가", "광고 사람", "국내생산", "동시방송 상품개수", "성별", "일시불/무이자"]:
            opt_x_embedding[name] = {"model_basic": val["model_basic"].to("cpu"), "data": val["data"][id_index].cpu()}
        elif name in ["월", "시간", "방송요일", "연속 휴일", "기온", "강수량", "vs", "lcsch", "dc10tca", "icsr", "ss", "pa", "pv", "hm",
                      "ws", 'popular_program', 'morning_drama']:
            opt_x_embedding[name] = {"model_basic": val["model_basic"].to("cpu"), "data": val["data"].cpu()}
        else:
            opt_x_embedding[name] = {"model_basic": val["model_basic"].to("cpu"), "data": val["data"][id_index].cpu()}
    return opt_x_embedding

def cal_y_for_opt(opt_x_embedding, device, coef, rep_num, day_id, good_ids_ori, result, model, y_max, y_min):
    id_index = torch.LongTensor([int(i) for i in good_ids_ori["index"].str.cat(sep=",").split(",")])
    day_index = result.loc[day_id,"index"]
    good_ids = good_ids_ori[good_ids_ori["반복"] >= rep_num]
    good_num = good_ids["index"].apply(lambda x: x.count(",") + 1)
    embedded_values = []
    for name, val in opt_x_embedding.items():
        if name in ["상품군", "마더코드", "상품코드", "브랜드", "판매단가", "광고 사람", "국내생산", "동시방송 상품개수", "성별", "일시불/무이자"]:
            value = val["model_basic"](val["data"]) * coef[name]
            embedded_values.append(value)
        elif name in ["월", "시간", "방송요일", "연속 휴일", "기온", "강수량", "vs", "lcsch", "dc10tca", "icsr", "ss", "pa", "pv", "hm", "ws", 'popular_program', 'morning_drama']:
            input_val = torch.zeros_like(val["data"][id_index])
            input_val[:] = val["data"][[day_index]]
            value = val["model_basic"](input_val) * coef[name]
            embedded_values.append(value)
        elif name == "Seq 방송 개수":
            input_val = torch.zeros_like(val["data"]).to("cpu")
            input_val[:, 0] = 1
            if str(result.loc[day_id,"상품"]) != "nan" and result.loc[day_index, "연결여부"] == True:
                input_val[int(result.loc[day_id,"상품"]), min(int(result.loc[day_id,"seq"] + 1 - 1), 6 - 1)] = 1
                input_val[int(result.loc[day_id,"상품"]),0] = 0
            value = val["model_basic"](input_val) * coef[name]
            embedded_values.append(value)
        elif name == "하루방송 수":
            input_val = torch.zeros_like(val["data"]).to("cpu")
            for i in range(len(input_val)):
                if i not in good_ids.index:
                    continue
                tmp = result[result["동일날짜"] == result.loc[day_id,"동일날짜"]] ; tmp = tmp[tmp["상품"] == i]
                if len(tmp) == 0:
                    input_val[i,0] = 1
                else:
                    input_val[i, min(int(tmp["하루"].iloc[-1] + 1 - 1), 6 - 1)] = 1
            value = val["model_basic"](input_val) * coef[name]
            embedded_values.append(value)
    embedded_value = torch.cat(embedded_values, dim=1)


    hat_y = model(embedded_value)
    y = (((hat_y + 1) / 2) * (y_max - y_min) + y_min)
    tmp = pd.Series([0] + list(good_num.values)).cumsum()
    result = []
    for i in range(len(good_num)):
        result.append(y[tmp[i]: tmp[i+1]].sum().item())

    return np.array(result), good_num

def assign_result(good_num, hugarian_result, result):
    day_index = result[result["상품"].astype(str) == "nan"].index
    good_index = good_num.index
    for i,j in enumerate(hugarian_result):
        result.loc[day_index[j],"상품"] = good_index[i]
    return result



def arrange_result_matrix(result):
    for i in range(len(result)):
        good = result.loc[i,"상품"]
        if str(good) == "nan":
            continue
        if result.loc[i, "연결여부"] and (result.loc[i-1,"상품"] == good):
            result.loc[i, "seq"] = result.loc[i - 1, "seq"] + 1
        else:
            result.loc[i, "seq"] = 1
        tmp = result[(result["동일날짜"] == result.loc[i,"동일날짜"]) & (result.index < i)]
        tmp = tmp[tmp["상품"] == good]
        if len(tmp) == 0:
            result.loc[i, "하루"] = 1
        else:
            tmp = tmp["하루"]
            result.loc[i, "하루"] = tmp.iloc[-1] + 1
    return result





def write_loss1(loss, i, lst, writer, plot_name):
    lst.append(loss.item())
    writer.add_scalar(plot_name, loss, i)

def write_loss2(loss1, loss2, name1, name2, i, lst1, lst2, writer, plot_name):
    lst1.append(loss1.item()) ; lst2.append(loss2.item())
    dic = {name1 : loss1, name2:loss2}
    writer.add_scalars(plot_name, dic, i)

def FC_embedding_model_save(fc, x_embedding, save_loc):
    for name in x_embedding:
        if name == "일시불/무이자":
            name2 = "일시불,무이자"
        else:
            name2 = name
        torch.save(x_embedding[name]["model_basic"].state_dict(), save_loc + "/{}.pth".format(name2))
    torch.save(fc.state_dict(), save_loc + "/fc.pth")

def FC_embedding_model_load(fc, x_embedding, save_loc, device):
    for name in x_embedding:
        if name == "일시불/무이자":
            name2 = "일시불,무이자"
        else:
            name2 = name
        x_embedding[name]["model_basic"].load_state_dict(torch.load(save_loc + "/{}.pth".format(name2), map_location=device))
    fc.load_state_dict(torch.load(save_loc + "/fc.pth", map_location=device))
