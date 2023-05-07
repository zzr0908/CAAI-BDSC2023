import pandas as pd
from data_loader import *
from trainer import *
from models import *
from samplers import *
from utils import *


if __name__ == '__main__':
    target_event = pd.read_json("data/target_event_preliminary_train_info.json")
    source_event = pd.read_json("data/source_event_preliminary_train_info.json")
    sub_graphs = pd.read_csv("data/subgroup.csv")
    user = pd.read_json("data/user_info.json")
    #
    demo_users: list = sub_graphs[sub_graphs["root"] == "d09ad25df105efc54ea571eaf498a521"].user.tolist()
    demo_target: pd.DataFrame = target_event[target_event["inviter_id"].isin(demo_users)]
    demo_source: pd.DataFrame = source_event[source_event["inviter_id"].isin(demo_users)]
    demo_event: pd.DataFrame = pd.concat([demo_source, demo_target], axis=0)
    demo_user_info: pd.DataFrame = user[user["user_id"].isin(demo_users)]

    pos_sampler = dgl.dataloading.NeighborSampler([4, 4])
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)

    trainer = Trainer(model_name="Sage", pos_sampler=pos_sampler, neg_sampler=negative_sampler,
                      loss=entropy_loss, device='cuda:0')
    trainer.data_prepare(demo_event, demo_user_info)

    model_config = {"input": 3, "embedding": 64, "output": 32}
    trainer.model_init(model_config)

    train_config = {"batch_size": 16, "opt": "Adam", "iter": 50}
    trainer.train(train_config)
