from data_loader import *
from models.ensemble import *
from loss_function import *
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import dgl
from models.ensemble import *
import random
from tqdm import tqdm
from evaluation import *


class Trainer:
    model: torch.nn.Module
    graph: dgl.DGLGraph
    _target_data: pd.DataFrame
    """
    训练阶段分为data prepare, pretrain, finetune三步
    data prepare阶段将数据完成数据预处理与构图工作
    pretrain阶段使用source阶段的数据进行预训练，获得节点embedding (暂定使用带负样本的边分类任务作为预训练任务）
    finetune阶段利用预训练的节点信息，完成target域的连接预测任务 (模型待定）
    
    func:
    __init__
        指定pretrain和finetune模型作为参数以确定训练策略，同时指定device
    data_prepare
        输入event和user的数据，完成数据预处理和构图
    pretrain
        根据指定的pretrain模型进行训练
        必要参数包括 loss, sampler, epoch, batch_size, opt
    """

    def __init__(self, pretrain_model, finetune_model, device):
        self.pretrain_model = pretrain_model    # 模型名称，用于指定模型
        self.finetune_model = finetune_model
        self.device = device
        self._meta_data = {}

    def data_prepare(self, source, target, users, data_config={}):
        print("======Graph data preparing=======")
        source_val_frac = data_config.get("source_val_frac", 0.05)
        target_val_frac = data_config.get("target_val_frac", 0.05)
        rs = data_config.get("rs", 1234)

        data = GraphDataset(source, target, users, source_val_frac, target_val_frac, rs)
        data.process()
        self.graph = data[0].to(self.device)    # 图先放在cpu, 训练阶段将子图放gpu
        self._target_data = data.target_data
        self._meta_data.update(data.meta_data)
        print("======prepare finished=======")

    def pretrain(self, pretrain_config):
        # params
        # loss_func = pretrain_config.get("loss", multi_label_loss)
        loss_func = pretrain_config.get("loss", auc_surrogate_loss)
        epoch = pretrain_config.get("epoch", 10)
        batch_size = pretrain_config.get("batch_size", 8)
        sample_neighbor = pretrain_config.get("sample_neighbor", [4, 4])
        model_config = pretrain_config.get("model_config", {})

        # model init
        node_cnt = self.meta_data.get("num_nodes")
        embedding = model_config.get("embedding", 64)
        hidden_feats = model_config.get("hidden_feats", [64, 64])
        out = self.meta_data.get("source_event_cnt") * 2
        pretrain_model = GraphSageModel(node_cnt, embedding, hidden_feats, out)
        pretrain_model.to(self.device)

        print("\n init graph sage model with: \n",
              f"number of nodes: {node_cnt} \n",
              f"node embedding size: {embedding} \n",
              f"size of hidden layers: {hidden_feats} \n",
              f"classify number: {out} \n",
              f"model device: {self.device} \n")

        # 将预训练抽样器用neighbor sampler, 参数传入
        sampler = dgl.dataloading.NeighborSampler(sample_neighbor)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude="self")

        # 获取用于pretrain训练的边与val的边
        eid = torch.arange(self.graph.num_edges()).to(self.device)
        train_eid = eid[self.graph.edata["source_train_mask"]]  # 训练集
        val_eid = eid[self.graph.edata["source_val_mask"]]  # 验证集

        # 定义训练的采样器
        train_loader = dgl.dataloading.DataLoader(
                    self.graph,
                    train_eid,
                    sampler,
                    device=self.device,
                    batch_size=batch_size,  # Batch size
                    shuffle=True,  # Whether to shuffle the nodes for every epoch.to(self.device)
                    drop_last=False,
                )
        val_loader = dgl.dataloading.DataLoader(
            self.graph,
            val_eid,
            sampler,
            device=self.device,
            batch_size=batch_size,  # Batch size
            shuffle=False,  # Whether to shuffle the nodes for every epoch.to(self.device)
            drop_last=False,
        )

        # training loop
        print("\n start pretraining \n")
        # writer = SummaryWriter("D:/zhan/CAAI-BDSC2023/logs")

        val_step = len(val_eid) / batch_size
        train_step = len(train_eid) / batch_size

        opt = torch.optim.Adam(pretrain_model.parameters(), weight_decay=0.05, lr=0.005)
        for i in range(epoch):

            val_loss = 0
            for step, (input_nodes, subgraph, mfgs) in enumerate(val_loader):
                with torch.no_grad():
                    nodes = mfgs[0].srcdata["_ID"]
                    predictions = pretrain_model(mfgs, nodes, subgraph)
                    edge_label = subgraph.edata["source_events"]
                    val_loss += loss_func(predictions, edge_label).item()

            train_loss = 0
            for step, (input_nodes, subgraph, mfgs) in enumerate(train_loader):
                nodes = mfgs[0].srcdata["_ID"]
                predictions = pretrain_model(mfgs, nodes, subgraph)
                edge_label = subgraph.edata["source_events"]
                loss = loss_func(predictions, edge_label)
                with torch.no_grad():
                    train_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()

            print(f"pretrain batch{i+1}/{epoch}  ------train loss:{round(train_loss/train_step, 4)}, val loss:{round(val_loss/val_step, 4)}")

        print("\n pretrain classifier evaluation\n")

        val_prediction = np.zeros((len(val_eid), out))
        val_target = np.zeros((len(val_eid), out))

        for step, (input_nodes, subgraph, mfgs) in enumerate(val_loader):
            with torch.no_grad():
                nodes = mfgs[0].srcdata["_ID"]
                predictions = pretrain_model(mfgs, nodes, subgraph).to("cpu").numpy()
                target = subgraph.edata["source_events"].to("cpu").numpy()
                val_prediction[step*batch_size: step * batch_size + predictions.shape[0], :] = predictions
                val_target[step * batch_size: step * batch_size + predictions.shape[0], :] = target

        evaluation = edge_classification_evaluation(val_prediction, val_target)
        evaluation = {key: round(np.mean(val), 4) for key, val in evaluation.items()}
        print(evaluation)

        self.model = pretrain_model     # 将预训练模型保存

    def finetune(self, finetune_config):

        neg_cnt = finetune_config.get("neg_cnt", 1)
        target_event_cnt = self.meta_data.get('target_event_cnt', 8)
        batch_size = finetune_config.get("batch_size", 64)
        epoch = finetune_config.get("epoch", 20)
        loss_func = finetune_config.get("loss", multi_label_loss)
        recall_neighbour_level = finetune_config.get("recall_neighbour_level", 2)
        min_recall = finetune_config.get("min_recall", 10)

        print("\n finetuning with: \n",
              f"negative samples ratio: {neg_cnt} \n",
              f"batch_size: {batch_size} \n",
              f"epoch: {epoch} \n",
              f"recall neighbour level: {recall_neighbour_level} \n",
              f"minial recall voter if exists: {min_recall} \n")

        eid = torch.arange(self.graph.num_edges()).to(self.device)
        train_eid = eid[self.graph.edata["target_train_mask"]]  # 训练集
        val_eid = eid[self.graph.edata["target_val_mask"]]      # 验证集

        # init model
        finetune_model = self.model
        finetune_model.reset_out_layer(target_event_cnt)
        frozen_layer(finetune_model, "embedding_layer")

        # dataloader with negative sampler
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(neg_cnt)
        sampler = dgl.dataloading.NeighborSampler([-1, -1])
        sampler = dgl.dataloading.as_edge_prediction_sampler(
            sampler, negative_sampler=neg_sampler
        )
        finetune_loader = dgl.dataloading.DataLoader(
            self.graph,
            train_eid,
            sampler,
            device=self.device,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = dgl.dataloading.DataLoader(
            self.graph,
            val_eid,
            sampler,
            device=self.device,
            batch_size=val_eid.shape[0],
            shuffle=True,
            drop_last=False,
        )

        # training loop
        print("\n start finetune training \n")
        opt = torch.optim.Adam(finetune_model.parameters(), weight_decay=0.05, lr=0.005)
        val_step = len(val_eid) / batch_size
        train_step = len(train_eid) / batch_size

        print(val_step, train_step)
        # writer = SummaryWriter("D:/zhan/CAAI-BDSC2023/logs")
        for i in range(epoch):
            val_loss = 0
            train_loss = 0

            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(val_loader):
                with torch.no_grad():
                    nodes = mfgs[0].srcdata["_ID"]

                    # 正样本
                    pos_predictions = finetune_model(mfgs, nodes, pos_graph)
                    pos_edge_label = pos_graph.edata["target_event"]
                    pos_loss = loss_func(pos_predictions, pos_edge_label)

                    # 负样本
                    neg_predictions = finetune_model(mfgs, nodes, neg_graph)
                    neg_edge_label = torch.zeros(neg_predictions.shape).to(self.device)
                    neg_loss = loss_func(neg_predictions, neg_edge_label)

                    val_loss += pos_loss.item()
                    val_loss += neg_loss.item()

            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(finetune_loader):
                nodes = mfgs[0].srcdata["_ID"]

                # 正样本
                pos_predictions = finetune_model(mfgs, nodes, pos_graph)
                pos_edge_label = pos_graph.edata["target_event"]
                pos_loss = loss_func(pos_predictions, pos_edge_label)

                # 负样本
                neg_predictions = finetune_model(mfgs, nodes, neg_graph)
                neg_edge_label = torch.zeros(neg_predictions.shape).to(self.device)
                neg_loss = loss_func(neg_predictions, neg_edge_label)

                loss = pos_loss + neg_loss
                with torch.no_grad():
                    train_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()

            print(f"finetune batch{i+1}/{epoch} ------- train loss:{round(train_loss/train_step, 4)}, val loss:{round(val_loss/val_step, 4)} \n")
            # writer.add_scalar('finetune train_loss/batch', train_loss, i)
            # writer.add_scalar('finetune val_loss/batch', val_loss, i)

        print("\n finetune training finished \n")

        target_df = self.target_data
        val_df = target_df[target_df["is_val"] == 1]
        train_df = target_df[target_df["is_val"] == 0].set_index(["inviter_id", "event_id"])

        inviter_ids = val_df.inviter_id.tolist()
        event_ids = val_df.event_id.tolist()
        predictions = []
        candidates = []
        val_graph = self.graph.to("cpu")
        val_graph.remove_edges(val_eid.to("cpu"))
        for i in range(len(inviter_ids)):
            white = train_df.voter_id_list.get((inviter_ids[i], event_ids[i]), [])
            p, c = self.infer(inviter_ids[i], event_ids[i], val_graph, recall_level=recall_neighbour_level
                              , min_recall=min_recall,white_list=white)
            predictions.append(p)
            candidates.append(c)

        val_df["prediction"] = predictions
        val_df["candidate_cnt"] = candidates
        return val_df

    def infer(self, inviter_id, event_id, g, recall_level=3, min_recall=10, max_recall=100, k=5, white_list=[]):
        """
        对某个点a，召回其k层的点a_recall共n个，并移除白名单中的点
        最内层的图 a 与 A_recall的边
        以{a, A_recall 为起点，向外扩两层}
        采样N个点的2(暂定)层邻居生成子图，输入模型中，计算目标点与找回点的分数并计分
        """
        candidates = recall(g, inviter_id, recall_level, min_recall, white_list)

        if len(candidates) == 0:
            return [], []
        # todo: 1.白名单处理  2.改为批量
        u, v = [0] * len(candidates), [i for i in range(1, len(candidates)+1)]
        sub_graph = dgl.graph((u, v)).to(self.device)

        seed_nodes = [inviter_id] + candidates
        sampler = dgl.dataloading.NeighborSampler([-1, -1])
        input_nodes, output_nodes, mfgs = sampler.sample(g, seed_nodes)

        input_nodes = input_nodes.to(self.device)
        mfgs = [sg.to(self.device) for sg in mfgs]

        with torch.no_grad():
            scores = self.model(mfgs, input_nodes, sub_graph)
            scores = scores[:, event_id].to("cpu").numpy()
        pred_idx = np.argsort(scores)[::-1]
        return np.array(candidates)[pred_idx].tolist()[: k], candidates

    @property
    def meta_data(self):
        return self._meta_data

    @property
    def target_data(self):
        return self._target_data


def get_subgraph_vec(subgraph, mfgs, feat):
    u, v = subgraph.edges()
    feat = mfgs[0].srcdata[feat]
    u, v = feat[u], feat[v]
    return u, v


def recall(graph: dgl.DGLGraph, inviter: int, neighbour_level=3, min_recall=10, white_list=[]):
    """
    从子图中召回
    注意graph默认在cpu上
    """
    candidates_list = dgl.bfs_nodes_generator(graph, inviter)   # [np.array()]的逐级邻居
    candidates = []
    for i in range(1, neighbour_level+1):
        if i >= len(candidates_list):
            break
        temp = candidates_list[i].numpy().tolist()
        temp = [item for item in temp if item not in white_list]    # reject white list
        candidates += temp
        if len(candidates) > min_recall:
            break
    return candidates


def frozen_layer(model, layers):
    """
    根据层名字冻结模型某些层
    """
    for name, child in model.named_children():
        if name not in layers:
            continue
        for param in child.parameters():
            param.requires_grad = False








