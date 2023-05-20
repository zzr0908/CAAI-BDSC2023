from dgl.data import DGLDataset
import dgl
import random
import numpy as np
import pandas as pd
from loss_function import *
from utils import *


class GraphDataset(DGLDataset):
    graph: dgl.DGLGraph
    _target_data: pd.DataFrame  # 储存验证集用户
    """
    init
    传入已经转化为id的source和target、user
    event中source, target分别标号
    user需要按id标号
    
    process构图：
        节点包括所有节点，节点特征为user的初始特征
        边包括source边与target边，target和source边中关系会被分别汇总为onehot向量
        边特征包括：
            source_train_mask: 预训练任务的训练集
            source_val_mask: 预训练任务的验证集
            source_events: source event的one hot向量，size=(source event + target event)*2
            target_train_mask: 来自target训练集
            target_val_mask: 来自target训练集
            target_event 目标事件的id
    """
    def __init__(self,  source_event, target_event, user_info, source_val_frac=0.05, target_val_frac=0.2, rs=2023):
        self.source_event = source_event[["inviter_id", "event_id", "voter_id"]]
        self.target_event = target_event[["inviter_id", "event_id", "voter_id"]]
        self.user_info = user_info
        self.source_val_frac = source_val_frac
        self.target_val_frac = target_val_frac
        self.rs = rs
        self._meta_data = {}     # 储存各类数据

    def process(self):
        # 将数据转换为图
        source_event_cnt: int = self.source_event.event_id.max() + 1    # source event 的数量
        target_event_cnt: int = self.target_event.event_id.max() + 1  # target event 的数量

        # 生成逆向关系
        reverse_source: pd.DataFrame = self.source_event.copy()
        reverse_source.columns = ["voter_id", "event_id", "inviter_id"]
        reverse_source["event_id"] += source_event_cnt

        # 合并source并拆分为train和val
        all_source_event: pd.DataFrame = pd.concat([self.source_event, reverse_source], axis=0)
        all_source_event = all_source_event.groupby(["inviter_id", "voter_id"]).event_id.\
            apply(lambda x: x.tolist()).reset_index()
        all_source_event["source_events"] = all_source_event.event_id.apply(lambda x: to_one_hot(x, 2*source_event_cnt))
        np.random.seed(self.rs)
        all_source_event["source_train_mask"] = np.random.binomial(1, 1-self.source_val_frac, all_source_event.shape[0])
        all_source_event["source_val_mask"] = all_source_event["source_train_mask"].apply(lambda x: np.abs(1-x))
        all_source_event["target_event"] = [[0]*target_event_cnt for _ in range(all_source_event.shape[0])]
        all_source_event["target_train_mask"] = 0
        all_source_event["target_val_mask"] = 0
        all_source_event = all_source_event[["voter_id", "inviter_id", "source_train_mask", "source_val_mask",
                                             "source_events", "target_train_mask", "target_val_mask", "target_event"]]
        source_train_cnt = all_source_event["source_train_mask"].sum()
        source_val_cnt = all_source_event["source_train_mask"].count() - source_train_cnt

        # 对target做拆分
        np.random.seed(self.rs)
        self.target_event["target_train_mask"] = np.random.binomial(1, 1-self.target_val_frac, self.target_event.shape[0])
        self.target_event["target_val_mask"] = self.target_event["target_train_mask"].apply(lambda x: np.abs(1-x))

        # 保存target数据
        self._target_data = self.target_event.groupby(["inviter_id", "event_id", "target_val_mask"]).\
            voter_id.apply(lambda x: x.tolist()).reset_index()
        self._target_data.columns = ["inviter_id", "event_id", "is_val", "voter_id_list"]

        self.target_event = self.target_event.groupby(["inviter_id", "voter_id", "target_train_mask", "target_val_mask"]).\
            event_id.apply(lambda x: x.tolist()).reset_index()
        self.target_event["target_event"] = self.target_event.event_id.apply(
            lambda x: to_one_hot(x, target_event_cnt))

        self.target_event["source_train_mask"] = 0
        self.target_event["source_val_mask"] = 0
        self.target_event["source_events"] = [[0] * (2 * source_event_cnt) for _ in range(self.target_event.shape[0])]

        all_target_event = self.target_event[["voter_id", "inviter_id", "source_train_mask", "source_val_mask",
                                             "source_events", "target_train_mask", "target_val_mask", "target_event"]]

        target_train_event = all_target_event["target_train_mask"].sum()
        target_val_event = all_target_event["target_train_mask"].count() - target_train_event
        val_triple = self.target_data.is_val.sum()

        # 保存信息
        events = pd.concat([all_source_event, all_target_event], axis=0)

        # 构图
        src_, dst_ = events["inviter_id"].tolist(), events["voter_id"].tolist()
        self.graph = dgl.graph((src_, dst_))
        self.graph.ndata['user_info'] = torch.tensor(np.array(self.user_info[["gender_id", "age_level", "user_level"]]),
                                                     dtype=torch.float32)
        self.graph.edata['source_train_mask'] = torch.tensor(np.array(events["source_train_mask"]), dtype=torch.bool)
        self.graph.edata['source_val_mask'] = torch.tensor(np.array(events["source_val_mask"]), dtype=torch.bool)
        self.graph.edata['source_events'] = torch.tensor(np.array(events["source_events"].tolist()), dtype=torch.float32)
        self.graph.edata['target_train_mask'] = torch.tensor(np.array(events["target_train_mask"]), dtype=torch.bool)
        self.graph.edata['target_val_mask'] = torch.tensor(np.array(events["target_val_mask"]), dtype=torch.bool)
        self.graph.edata['target_event'] = torch.tensor(np.array(events["target_event"].tolist()), dtype=torch.float32)

        # 保存信息
        self._meta_data["source_event_cnt"] = source_event_cnt
        self._meta_data["target_event_cnt"] = target_event_cnt
        self._meta_data["source_train_cnt"] = source_train_cnt
        self._meta_data["source_val_cnt"] = source_val_cnt
        self._meta_data["target_train_event"] = target_train_event
        self._meta_data["target_val_event"] = target_val_event
        self._meta_data["val_triple"] = val_triple
        self._meta_data["num_nodes"] = self.graph.num_nodes()
        self._meta_data["num_edges"] = self.graph.num_edges()

    def __getitem__(self, idx):
        assert idx == 0
        return self.graph

    @property
    def target_data(self):
        return self._target_data

    @property
    def meta_data(self):
        return self._meta_data

    def __len__(self):
        return 1



