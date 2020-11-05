import pandas as pd
import torch.optim as optim
from util.util import *
import random
from data.data_preprocessing import make_data_set,make_dataset_embedding
from scipy.optimize import linear_sum_assignment

class FCClass:
    def __init__(self, num, y_col_name, info, max_y, embedding_set, concat, validation_ratio, drop, lower_sim, rank_sim_train, BATCH_SIZE, device):
        self.embedding_set = embedding_set
        self.device = device
        self.concat = concat
        self.rank_sim_train = rank_sim_train
        train_data, test_data, name_code, sim_matrix, final_json, train_price, self.test_price, train_revenue, \
        train_name, self.test_name, self.real_train_index, self.valid_index, self.y_max, self.y_min = \
        make_data_set(info, y_col_name, max_y, device, num, lower_sim, validation_ratio)
        self.sim_good = prepare_sim_matrix(sim_matrix, rank_sim_train, final_json, name_code, device)
        self.train_price, self.valid_price = train_price[self.real_train_index], train_price[self.valid_index]
        self.train_revenue, self.valid_revenue = train_revenue[self.real_train_index], train_revenue[self.valid_index]
        self.train_name, self.valid_name = train_name[self.real_train_index], train_name[self.valid_index]
        self.all_column_info, self.total_input_size, self.train_x_embedding, self.train_x_val, self.train_y_val = \
            make_dataset_embedding(train_data.loc[self.real_train_index], final_json, info, embedding_set, drop, device, concat, data_type= "train")
        _, _, self.valid_x_embedding, self.valid_x_val, self.valid_y_val = \
            make_dataset_embedding(train_data.loc[self.valid_index], final_json, info, embedding_set, drop, device, concat, data_type= "valid", before_x_embedding= self.train_x_embedding)
        _, _, self.test_x_embedding, self.test_x_val = \
            make_dataset_embedding(test_data, final_json, info, embedding_set, drop, device, concat, data_type= "test", before_x_embedding= self.train_x_embedding)

        self.BATCH_SIZE = BATCH_SIZE

    def fit(self, Model_Class, EPOCH, E_lr, FC_lr, E_beta, FC_beta, E_weight_decay, FC_weight_decay, reg_coef, coef, random_state, writer, save_file, save_loc):
        ########################
        # Model and Optimizer
        ########################
        self.net = Model_Class(self.total_input_size).to(self.device)
        E_param = []
        for val in self.train_x_embedding.values():
            E_param += list(val["model_basic"].parameters())
        optimizerE = optim.Adam(E_param, lr=E_lr, betas=E_beta, weight_decay=E_weight_decay)
        optimizerFC = optim.Adam(self.net.parameters(), lr=FC_lr, betas=FC_beta, weight_decay= FC_weight_decay)

        total_iter = 0
        train_MAPE_lst = []
        total_loss_r_lst = []
        test_MAPE_lst = []
        test_hat_y_real_lst = []
        test_y_real_lst = []
        train_index = list(range(len(self.real_train_index)))
        for i in range(EPOCH):
            random.shuffle(train_index)
            for j in range(len(train_index) // self.BATCH_SIZE):
                ###################
                # Train
                ###################
                self.net.train();
                total_iter += 1
                index = train_index[j * self.BATCH_SIZE: (j + 1) * self.BATCH_SIZE]
                for sim_num in range(-1, self.rank_sim_train):
                    embedded_value, regularizer = make_embedding(self.embedding_set, self.train_x_embedding, index, self.device,
                                                                 reg_coef, coef, self.concat, self.train_name, self.sim_good, sim_num, train= True)
                    if len(self.train_x_val) == 0:
                        x, y = embedded_value.to(self.device), self.train_y_val[index].to(self.device)
                    else:
                        x, y = torch.cat([self.train_x_val[index].to(self.device), embedded_value], axis=1).to(self.device), self.train_y_val[index].to(self.device)
                    hat_y = self.net(x).to(self.device)

                    _, _, FC_loss = cal_MAPE(hat_y, self.train_revenue[index], self.train_price[index], self.y_max, self.y_min, revenue_bool=False)
                    write_loss1(FC_loss, i, train_MAPE_lst, writer, "loss/train_MAPE")
                    write_loss1(regularizer, i, total_loss_r_lst, writer, "loss/regularizer")
                    total_loss = FC_loss + regularizer
                    optimizerE.zero_grad()
                    optimizerFC.zero_grad()
                    total_loss.backward()
                    optimizerE.step()
                    optimizerFC.step()
                    ################
                    # TEST
                    ################
                    for type_num in [-1,0]:
                        self.net.eval()
                        valid_embedded_value, _ = make_embedding(self.embedding_set, self.valid_x_embedding, list(range(len(self.valid_index))),
                                                                self.device, reg_coef, coef, self.concat, self.valid_name, self.sim_good, type_num, train= False)
                        if len(self.valid_x_val) == 0:
                            valid_x, valid_y = valid_embedded_value.to(self.device), self.valid_y_val.to(self.device)
                        else:
                            valid_x, valid_y = torch.cat([self.valid_x_val.to(self.device), valid_embedded_value],axis=1).to(self.device), \
                                             self.valid_y_val.to(self.device)
                        valid_hat_y = self.net(valid_x).to(self.device)

                        valid_y_real, valid_hat_y_real, MAPE = cal_MAPE(valid_hat_y, self.valid_revenue, self.valid_price, self.y_max, self.y_min,
                                                                        revenue_bool = False)
                        write_loss1(MAPE, i, test_MAPE_lst, writer, "loss/valid_MAPE_{}".format(type_num))
                        write_loss2(valid_hat_y_real.mean(), valid_y_real.mean(), "valid_hat_y_real_{}".format(type_num), "valid_y_real_{}".format(type_num), i,
                                    test_hat_y_real_lst, test_y_real_lst, writer, "loss/real_price_type_{}".format(type_num))

            if i % 50 == 49:
                save_file.write("{} epoch: {}, {}".format(i, round(min(test_MAPE_lst), 5), round(min(train_MAPE_lst), 5))  + "\n")
                print("{} epoch:".format(i), round(min(test_MAPE_lst), 5), round(min(train_MAPE_lst), 5))
        if EPOCH != 1:
            print("train")
            FC_embedding_model_save(self.net, self.train_x_embedding, save_loc)
        else:
            print("test")


    def load_bestmodel(self,save_loc):
        FC_embedding_model_load(self.net, self.test_x_embedding, save_loc, self.device)

    def predict(self, coef, save_loc):
        test_embedded_value = make_embedding_for_test(self.embedding_set, self.test_x_embedding, self.device, coef, self.concat, self.test_name, self.sim_good)
        if len(self.test_x_val) == 0:
            test_x = test_embedded_value.to(self.device)
        else:
            test_x = torch.cat([self.test_x_val.to(self.device), test_embedded_value], axis=1).to(self.device)
        hat_y = self.net(test_x).to(self.device)
        hat_y_real = (((hat_y + 1) / 2) * (self.y_max - self.y_min) + self.y_min) * self.test_price
        pd.DataFrame(hat_y_real.cpu().detach().numpy()).to_csv(save_loc + "/test_result.csv")
        return hat_y_real

    def optimize(self, coef, save_loc):
        self.net = self.net.to("cpu")
        good_ids = pd.read_csv("/home/bigdyl/jayoung/tmp/data/extra_data/test_상품id.csv",index_col=0)
        result = pd.read_csv("/home/bigdyl/jayoung/tmp/data/extra_data/test_날짜id.csv",index_col=0)
        result["상품"] = np.nan ; result["seq"] = 0 ; result["하루"] = 0
        opt_x_embedding = prepare_x_embedding_for_opt(self.test_x_embedding,self.device, self.test_name, self.sim_good, good_ids)
        total_sale = 0
        for rep_num in range(1, good_ids["반복"].max() + 1):
            print(good_ids["반복"].max(), ":" , rep_num)
            tmp = result[result["상품"].astype(str) == "nan"]
            if len(tmp) == 0:
                break
            tmp_result = []
            print(len(tmp.index),end=":")
            i_tmp = 0
            for day in list(tmp.index):
                i_tmp += 1 ; print(i_tmp, end=" ") if i_tmp % 20 == 0 else print("", end ="")
                y, good_num = cal_y_for_opt(opt_x_embedding, self.device, coef, rep_num, day, good_ids, result, self.net,self.y_max, self.y_min)
                tmp_result.append(y)
            cost = np.column_stack(tmp_result)
            row_ind, col_ind = linear_sum_assignment(cost,maximize=True)
            assign_result(good_num, col_ind, result)
            total_sale += cost[row_ind, col_ind].sum()
            result = arrange_result_matrix(result)
        result = result[["방송일시","노출(분)_all","상품"]]
        result.to_csv(save_loc + "/opt_plan.csv")
        return result, total_sale







