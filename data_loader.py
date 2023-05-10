from dgl.data import DGLDataset
import dgl
import random
import numpy as np
import pandas as pd
from loss_function import *
from utils import *


class GraphDataset(DGLDataset):
    graph: dgl.DGLGraph

    """
    init
    传入已经转化为id的source和target、user
    event中source, target分别标号
    user需要按id标号
    
    process构图：
        节点包括所有节点，节点特征为user的初始特征
        边包括source边与target边，target和source边中关系会被分别汇总为onehot向量
        边特征包括：
            source_mask: 来自source, size=1
            source_events: source event的one hot向量，size=(source event + target event)*2
            target_train_mask: 来自target训练集
            target_val_mask: 来自target训练集
            target_event 目标事件的id
    """
    def __init__(self,  source_event, target_event, user_info, val_frac=0.2, rs=2023):
        self.source_event = source_event[["inviter_id", "event_id", "voter_id"]]
        self.target_event = target_event[["inviter_id", "event_id", "voter_id"]]
        self.user_info = user_info
        self.val_frac = val_frac
        self.rs = 2023

    def process(self):
        # 将数据转换为图
        source_event_cnt: int = self.source_event.event_id.max()+1    # source event 的数量
        target_event_cnt: int = self.target_event.event_id.max() + 1  # source event 的数量
        reverse_source: pd.DataFrame = self.source_event.copy()

        # 生成逆向关系并合并
        reverse_source.columns = ["voter_id", "event_id", "inviter_id"]
        reverse_source["event_id"] += source_event_cnt
        all_source_event: pd.DataFrame = pd.concat([self.source_event, reverse_source])
        all_source_event = all_source_event.groupby(["inviter_id", "voter_id"]).event_id.apply(lambda x: x.tolist()).reset_index()
        all_source_event["source_events"] = all_source_event.event_id.apply(lambda x: to_one_hot(x, 2*source_event_cnt-1))
        all_source_event["source_mask"] = 1
        all_source_event["target_event"] = [[0]*target_event_cnt for i in range(all_source_event.shape[0])]    # 即这些边不存在target关系
        all_source_event["target_train_mask"] = 0
        all_source_event["target_val_mask"] = 0
        all_source_event = all_source_event[["voter_id", "inviter_id",
                                             "source_mask", "source_events",
                                             "target_train_mask", "target_val_mask", "target_event"]]

        # 对target做拆分
        self.target_event = self.target_event.groupby(["inviter_id", "voter_id"]).event_id.apply(lambda x: x.tolist()).reset_index()
        self.target_event["target_event"] = self.target_event.event_id.apply(lambda x: to_one_hot(x, target_event_cnt-1))
        self.target_event["source_mask"] = 0
        self.target_event["source_events"] = [[0]*(2*source_event_cnt) for i in range(self.target_event.shape[0])]
        target_train, target_val = train_val_split(self.target_event, frac=self.val_frac, seed=self.rs)

        target_train["target_train_mask"] = 1
        target_train["target_val_mask"] = 0
        target_val["target_train_mask"] = 0
        target_val["target_val_mask"] = 1

        all_target_event = pd.concat([target_train, target_val], axis=0)
        all_target_event = all_target_event[["voter_id", "inviter_id",
                                             "source_mask", "source_events",
                                             "target_train_mask", "target_val_mask", "target_event"]]

        events = pd.concat([all_source_event, all_target_event], axis=0)

        # 构图
        src_, dst_ = events["inviter_id"].tolist(), events["voter_id"].tolist()
        self.graph = dgl.graph((src_, dst_))
        self.graph.ndata['user_info'] = torch.tensor(np.array(self.user_info[["gender_id", "age_level", "user_level"]]),
                                            dtype=torch.float32)

        self.graph.edata['source_mask'] = torch.tensor(np.array(events["source_mask"]), dtype=torch.bool)
        self.graph.edata['source_events'] = torch.tensor(np.array(events["source_events"].tolist()), dtype=torch.float32)
        self.graph.edata['target_train_mask'] = torch.tensor(np.array(events["target_train_mask"]), dtype=torch.bool)
        self.graph.edata['target_val_mask'] = torch.tensor(np.array(events["target_val_mask"]), dtype=torch.bool)
        self.graph.edata['target_event'] = torch.tensor(np.array(events["target_event"].tolist()), dtype=torch.float32)

    def __getitem__(self, idx):
        assert idx == 0
        return self.graph

    def __len__(self):
        return 1


# class GraphDataLoader(DataLoader):


