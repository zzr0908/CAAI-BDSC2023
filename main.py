import pandas as pd
import warnings
from data_loader import *
from trainer import *
from models import *
from samplers import *
from utils import *

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',100)


if __name__ == '__main__':
    target_event = pd.read_json("data/target_event_preliminary_train_info.json")    # target域三元组
    source_event = pd.read_json("data/source_event_preliminary_train_info.json")    # source域三元组
    sub_graphs = pd.read_csv("data/subgroup.csv")   # 子群体
    user = pd.read_json("data/user_info.json")      # 用户信息
    #
    demo_users: list = sub_graphs[sub_graphs["root"] == "d09ad25df105efc54ea571eaf498a521"].user.tolist()
    demo_target: pd.DataFrame = target_event[target_event["inviter_id"].isin(demo_users)].reset_index(drop=True)
    demo_source: pd.DataFrame = source_event[source_event["inviter_id"].isin(demo_users)].reset_index(drop=True)
    demo_user_info: pd.DataFrame = user[user["user_id"].isin(demo_users)].reset_index(drop=True)

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

    graph_dataset = GraphDataset(demo_source, demo_target, demo_user_info)
    graph_dataset.process()
    print(graph_dataset[0])
    print(demo_source.shape, demo_target.shape)







    # pos_sampler = dgl.dataloading.NeighborSampler([4, 4])
    # negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    #
    # trainer = Trainer(model_name="Sage", pos_sampler=pos_sampler, neg_sampler=negative_sampler,
    #                   loss=entropy_loss, device='cuda:0')
    # trainer.data_prepare(demo_event, demo_user_info)
    #
    # model_config = {"input": 3, "embedding": 64, "output": 32}
    # trainer.model_init(model_config)
    #
    # train_config = {"batch_size": 16, "opt": "Adam", "iter": 50}
    # trainer.train(train_config)
