import pandas as pd
import warnings
from data_loader import *
from trainer import *
from models import *
from samplers import *
from utils import *
import torch
torch.cuda.manual_seed_all(1998)
import argparse

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


def run(model_config, data_config, pretrain_config, finetune_config):
    target_event = pd.read_json("data/target_event_preliminary_train_info.json")  # target域三元组
    source_event = pd.read_json("data/source_event_preliminary_train_info.json")  # source域三元组
    sub_graphs = pd.read_csv("data/subgroup.csv")  # 子群体
    user = pd.read_json("data/user_info.json")  # 用户信息
    root_event = pd.read_csv("data/root_event.csv")

    user["gender_id"] = user["gender_id"] + 1  # 处理特征，特征中存在-1(未知，转换为0以上得整数类别）
    user["age_level"] = user["age_level"] + 1
    user["user_level"] = user["user_level"] + 1

    events, roots = root_event.event_id.tolist(), root_event.root_event.tolist()
    event_root_dic = {events[i]: roots[i] for i in range(len(events))}

    # demo_users = target_event["inviter_id"].unique().tolist() + target_event["voter_id"].unique().tolist()
    # demo_users = demo_users + source_event["inviter_id"].unique().tolist() + source_event["voter_id"].unique().tolist()
    # demo_users = list(set(demo_users))
    # demo_target = target_event
    # demo_source = source_event

    demo_users: list = sub_graphs[sub_graphs["root"] != "a4d6f3b44edea6d489d72821d2ca3474"].user.tolist()
    demo_target: pd.DataFrame = target_event[target_event["inviter_id"].isin(demo_users)].reset_index(drop=True)
    demo_source: pd.DataFrame = source_event[source_event["inviter_id"].isin(demo_users)].reset_index(drop=True)

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

    # train
    trainer = Trainer(model_config['pretrain_model'], model_config['finetune_model'], device=model_config['device'])
    trainer.data_prepare(demo_source, demo_target, demo_user_info, data_config)
    pretrain_config['n_class'] = len(source2id) * 2
    trainer.pretrain(pretrain_config)
    trainer.finetune(finetune_config)
    val_df = trainer.infer()

    # evaluation
    val_df["mrr"] = val_df.apply(lambda x: apply_cal_mrr(x), axis=1)
    # val_df.to_csv("result/main_subgroup_infer.csv", index=False)  # target域三元组
    print(val_df.head())
    print(val_df["mrr"].mean())


def get_param():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pretrain_model', default='Sage')
    parser.add_argument('--finetune_model', default='Sage')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--source_val_frac', default=0.1)
    parser.add_argument('--rs', default=2023)

    # pretrain_config
    parser.add_argument('--pretrain_embedding', default=64)
    parser.add_argument('--pretrain_hidden_feats',  default=[64, 64])
    parser.add_argument('--pretrain_batch_size', default=2048)
    parser.add_argument('--pretrain_epoch', default=10)
    parser.add_argument('--pretrain_loss', default=multi_label_loss)
    parser.add_argument('--pretrain_sample_neighbor', default=[-1, -1])

    # finetune_config
    parser.add_argument('--finetune_epoch', default=15)
    parser.add_argument('--finetune_loss', default=multi_label_loss)
    parser.add_argument('--finetune_batch_size', default=1024)

    args = parser.parse_args()
    pretrain_model = args.pretrain_model
    finetune_model = args.finetune_model
    device = args.device
    source_val_frac = args.source_val_frac
    rs = args.rs
    pretrain_embedding = args.pretrain_embedding
    pretrain_hidden_feats = args.pretrain_hidden_feats
    pretrain_batch_size = args.pretrain_batch_size
    pretrain_epoch = args.pretrain_epoch
    pretrain_loss = args.pretrain_loss
    pretrain_sample_neighbor = args.pretrain_sample_neighbor
    finetune_epoch = args.finetune_epoch
    finetune_loss = args.finetune_loss
    finetune_batch_size = args.finetune_batch_size

    model_param = {
        'pretrain_model': pretrain_model,
        'finetune_model': finetune_model,
        'device': device
    }
    data_param = {
        'source_val_frac': source_val_frac,
        'rs': rs
    }

    pretrain_model_param = {
        'embedding': pretrain_embedding,
        'hidden_feats': pretrain_hidden_feats,
    }
    pretrain_param = {
        "model_config": pretrain_model_param,
        'batch_size': pretrain_batch_size,
        'epoch': pretrain_epoch,
        'loss': pretrain_loss,
        'sample_neighbor': pretrain_sample_neighbor
    }

    finetune_param = {
        'epoch': finetune_epoch,
        'loss': finetune_loss,
        'batch_size': finetune_batch_size
    }
    return model_param, data_param, pretrain_param, finetune_param


def apply_cal_mrr(line):
    return cal_mrr(line.voter_id_list, line.prediction)


def main():
    model_config, data_config, pretrain_config, finetune_config = get_param()
    run(model_config, data_config, pretrain_config, finetune_config)


if __name__ == '__main__':
    main()



