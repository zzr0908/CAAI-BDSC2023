import pandas as pd
import warnings
from data_loader import *
from trainer import *
from models import *
from samplers import *
from utils import *
import torch
torch.cuda.manual_seed_all(1998)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


if __name__ == '__main__':
    target_event = pd.read_json("data/target_event_preliminary_train_info.json")    # target域三元组
    source_event = pd.read_json("data/source_event_preliminary_train_info.json")    # source域三元组
    sub_graphs = pd.read_csv("data/subgroup.csv")   # 子群体
    user = pd.read_json("data/user_info.json")      # 用户信息
    root_event = pd.read_csv("data/root_event.csv")

    user["gender_id"] = user["gender_id"] + 1     # 处理特征，特征中存在-1(未知，转换为0以上得整数类别）
    user["age_level"] = user["age_level"] + 1
    user["user_level"] = user["user_level"] + 1

    events, roots = root_event.event_id.tolist(), root_event.root_event.tolist()
    event_root_dic = {events[i]: roots[i] for i in range(len(events))}

    demo_users = target_event["inviter_id"].unique().tolist() + target_event["voter_id"].unique().tolist()
    demo_users = demo_users + source_event["inviter_id"].unique().tolist() + source_event["voter_id"].unique().tolist()
    demo_users = list(set(demo_users))
    demo_target = target_event
    demo_source = source_event

    # demo_users: list = sub_graphs[sub_graphs["root"] == "a4d6f3b44edea6d489d72821d2ca3474"].user.tolist()
    # demo_target: pd.DataFrame = target_event[target_event["inviter_id"].isin(demo_users)].reset_index(drop=True)
    # demo_source: pd.DataFrame = source_event[source_event["inviter_id"].isin(demo_users)].reset_index(drop=True)

    demo_user_info: pd.DataFrame = user[user["user_id"].isin(demo_users)].reset_index(drop=True)

    demo_source["event_id"] = demo_source["event_id"].apply(lambda x: event_root_dic[x])
    demo_source = demo_source.drop_duplicates()

    user2id = {demo_users[i]: i for i in range(len(demo_users))}
    demo_target["inviter_id"] = demo_target["inviter_id"].apply(lambda x: user2id[x])
    demo_target["voter_id"] = demo_target["voter_id"].apply(lambda x: user2id[x])
    demo_source["inviter_id"] = demo_source["inviter_id"].apply(lambda x: user2id[x])
    demo_source["voter_id"] = demo_source["voter_id"].apply(lambda x: user2id[x])
    demo_user_info["user_id"] = demo_user_info["user_id"].apply(lambda x: user2id[x])
    demo_user_info = demo_user_info.sort_values(by="user_id").reset_index(drop=True)

    source2id = demo_source["event_id"].unique().tolist()
    source2id = {source2id[i]: i for i in range(len(source2id))}
    target2id = demo_target["event_id"].unique().tolist()
    target2id = {target2id[i]: i for i in range(len(target2id))}
    demo_target["event_id"] = demo_target["event_id"].apply(lambda x: target2id[x])
    demo_source["event_id"] = demo_source["event_id"].apply(lambda x: source2id[x])

    trainer = Trainer("Sage", "Sage", device='cuda:0')
    trainer.data_prepare(demo_source, demo_target, demo_user_info, {"source_val_frac": 0.3, "rs": 2023})

    model_config = {"embedding": 128, "hidden_feats": [128, 128]}
    pretrain_config = {"model_config": model_config,
                       "n_class": len(source2id)*2, "batch_size": 2048, "epoch": 0,
                       "loss": multi_label_loss, "sample_neighbor": [-1, -1]}
    trainer.pretrain(pretrain_config)
    #
    finetune_config = {"epoch": 0, "loss": multi_label_loss, "batch_size": 1024, "recall_neighbour_level": 2,
                       "min_recall": 10}
    val_df = trainer.finetune(finetune_config)

    def apply_cal_mrr(line):
        return cal_mrr(line.voter_id_list, line.prediction)


    val_df["mrr"] = val_df.apply(lambda x: apply_cal_mrr(x), axis=1)

    val_df.to_csv("result/main_subgroup_infer.csv", index=False)  # target域三元组

    print(val_df["mrr"].mean())
    #
    # evaluation_prediction = trainer.infer(trainer.evaluation_data)
    # mrr = mean_reciprocal_rank(evaluation_prediction)
    # print("MRR:", mrr)



