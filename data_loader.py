import dgl
dgl.load_backend('pytorch')
from dgl.data import DGLDataset
import random
import numpy as np
from loss_function import *


class GraphDataset(DGLDataset):
    user2id: dict
    event2id: dict
    graph: dgl.DGLGraph

    def __init__(self,  event, user_info):
        self.event = event[["inviter_id", "event_id", "voter_id"]]
        self.user_info = user_info

    def process(self):
        # 将数据转换为图
        # 对user_id 、event_id编号
        users = self.user_info["user_id"].tolist()
        self.user2id: dict = {users[i]: i for i in range(len(users))}
        event_ids: list = self.event["event_id"].unique().tolist()
        self.event2id: dict = {event_ids[i]: i for i in range(len(event_ids))}

        # 将编码替换为序号
        self.event["inviter_id"] = self.event["inviter_id"].apply(lambda x: self.user2id[x])
        self.event["voter_id"] = self.event["voter_id"].apply(lambda x: self.user2id[x])
        self.event["event_id"] = self.event["event_id"].apply(lambda x: self.event2id[x])
        self.user_info["user_id"] = self.user_info["user_id"].apply(lambda x: self.user2id[x])
        self.user_info = self.user_info.sort_values(by="user_id")

        #构造图
        src_, dst_ = self.event["inviter_id"].tolist(), self.event["voter_id"].tolist()
        self.graph = dgl.graph((src_, dst_))
        self.graph.ndata['user_info'] = torch.tensor(np.array(self.user_info[["gender_id", "age_level", "user_level"]]),
                                            dtype=torch.float32)
        self.graph.edata['edge_type'] = torch.tensor(np.array(self.event["event_id"]), dtype=torch.int32)

    def __getitem__(self, idx):
        assert idx == 0
        return self.graph

    def __len__(self):
        return 1


# class GraphDataLoader(DataLoader):


