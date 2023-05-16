from data_loader import *
from models.models import *
from samplers import *
from utils import *
from loss_function import *
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import dgl
import random
from tqdm import tqdm
from evaluation import *


class Trainer:
    model: torch.nn.Module
    graph: dgl.DGLGraph
    _val_data: pd.DataFrame
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
        rs = data_config.get("rs", 2023)

        data = GraphDataset(source, target, users, source_val_frac, target_val_frac, rs)
        data.process()
        self.graph = data[0].to(self.device)    # 图先放在cpu, 训练阶段将子图放gpu
        self._val_data = data.val_data
        self._meta_data.update(data.meta_data)
        print("======prepare finished=======")

    def pretrain(self, pretrain_config):
        # params
        # loss_func = pretrain_config.get("loss", multi_label_loss)
        loss_func = pretrain_config.get("loss", auc_surrogate_loss)
        epoch = pretrain_config.get("epoch", 10)
        batch_size = pretrain_config.get("batch_size", 8)
        sample_neighbor = pretrain_config.get("sample_neighbor", [4, 4])

        # model init
        if self.pretrain_model == "Sage":
            inp, emb, out = pretrain_config["input"], pretrain_config["embedding"], pretrain_config["output"]
            pretrain_model = SageModel(inp, emb, out).to(self.device)
            n_class = pretrain_config["n_class"]
            head = EdgeClassifyHead(out, n_class).to(self.device)

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
            batch_size=val_eid.shape[0],  # Batch size
            shuffle=False,  # Whether to shuffle the nodes for every epoch.to(self.device)
            drop_last=False,
        )

        # training loop
        print("\n start pretraining \n")
        # writer = SummaryWriter("D:/zhan/CAAI-BDSC2023/logs")

        opt = torch.optim.Adam(list(pretrain_model.parameters()) + list(head.parameters()), weight_decay=0.05)
        for i in range(epoch):
            epoch_loss = 0
            for step, (input_nodes, subgraph, mfgs) in enumerate(train_loader):
                inputs = mfgs[0].srcdata["user_info"]
                predictions = pretrain_model(mfgs, inputs)
                score = head(subgraph, predictions)
                edge_label = subgraph.edata["source_events"]
                loss = loss_func(score, edge_label)
                with torch.no_grad():
                    epoch_loss += loss
                loss.backward()
            train_step = step
            # 每次对全图训练完进行梯度下降
            opt.step()
            opt.zero_grad()

            for step, (input_nodes, subgraph, mfgs) in enumerate(val_loader):
                with torch.no_grad():
                    inputs = mfgs[0].srcdata["user_info"]
                    predictions = pretrain_model(mfgs, inputs)
                    score = head(subgraph, predictions)
                    edge_label = subgraph.edata["source_events"]
                    evaluation = edge_classification_evaluation(score.to("cpu").numpy(), edge_label.to("cpu").numpy())
                    val_loss = loss_func(score, edge_label)

            print(f"batch{i}/{epoch-1}  ------train loss:{epoch_loss/(train_step+1)}, val loss:{val_loss}")
            evaluation = {key: np.mean(val) for key, val in evaluation.items()}
            print(evaluation)
            # writer.add_scalar('pretraining loss/batch', loss, i)

        print("\n pretrain classifier evaluation\n")

        for step, (input_nodes, subgraph, mfgs) in enumerate(val_loader):
            with torch.no_grad():
                inputs = mfgs[0].srcdata["user_info"]
                predictions = pretrain_model(mfgs, inputs)
                score = head(subgraph, predictions).to("cpu").numpy()
                target = subgraph.edata["source_events"].to("cpu").numpy()
                evaluation = edge_classification_evaluation(score, target)
        evaluation = {key: np.mean(val) for key, val in evaluation.items()}
        print(evaluation)

        # 获取节点的embedding
        node_sampler = dgl.dataloading.NeighborSampler(sample_neighbor)
        node_loader = dgl.dataloading.DataLoader(
                        self.graph,
                        torch.arange(self.graph.num_nodes()).to(self.device),
                        node_sampler,
                        device=self.device,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False
                    )

        print("\n calculating node embedding \n")
        node_embeddings = torch.zeros((self.graph.num_nodes(), out)).to(self.device)
        with tqdm(node_loader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                with torch.no_grad():
                    inputs = mfgs[0].srcdata["user_info"]
                    outputs = pretrain_model(mfgs, inputs)
                    node_embeddings[output_nodes] = outputs
        self.graph.ndata["embedding"] = node_embeddings

    def finetune(self, finetune_config):

        neg_cnt = finetune_config.get("neg_cnt", 1)
        hidden = finetune_config.get("hidden", 64)
        target_event_cnt = finetune_config.get("target_event_cnt", 8)   # 得弄成内部传递
        node_embedding = finetune_config.get("node_feat", 128)
        batch_size = finetune_config.get("batch_size", 64)
        epoch = finetune_config.get("epoch", 20)
        opt = finetune_config.get("opt", "Adam")
        # loss_func = finetune_config.get("loss", multi_label_loss)
        loss_func = finetune_config.get("loss", auc_surrogate_loss)

        eid = torch.arange(self.graph.num_edges()).to(self.device)
        train_eid = eid[self.graph.edata["target_train_mask"]]  # 训练集
        val_eid = eid[self.graph.edata["target_val_mask"]]      # 验证集

        # init model
        finetune_model = FcLinkPredictor(node_embedding, hidden, target_event_cnt).to(self.device)

        # dataloader with negative sampler
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(neg_cnt)
        sampler = dgl.dataloading.NeighborSampler([0])
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
        print("\n start finetune training")
        # writer = SummaryWriter("D:/zhan/CAAI-BDSC2023/logs")
        for i in range(epoch):
            epoch_loss = 0
            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(finetune_loader):
                pos_u, pos_v = self.get_subgraph_vec(pos_graph, mfgs, "embedding")
                neg_u, neg_v = self.get_subgraph_vec(neg_graph, mfgs, "embedding")
                pos_pred = finetune_model(pos_u, pos_v)
                neg_pred = finetune_model(neg_u, neg_v)
                pred = torch.cat((pos_pred, neg_pred), 0)  # (2*batch, 1)
                pos_label = pos_graph.edata['target_event']
                neg_label = torch.zeros((neg_graph.num_edges(), pos_label.shape[1])).to(self.device)
                label = torch.cat((pos_label, neg_label), 0)
                loss = loss_func(pred, label)
                with torch.no_grad():
                    epoch_loss += loss
                if opt == "Adam":
                    opt = torch.optim.Adam(finetune_model.parameters(), weight_decay=0.05)
                else:
                    pass
                opt.zero_grad()
                loss.backward()
                opt.step()
            train_loss = epoch_loss / (step + 1)
            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(val_loader):
                with torch.no_grad():
                    pos_u, pos_v = self.get_subgraph_vec(pos_graph, mfgs, "embedding")
                    neg_u, neg_v = self.get_subgraph_vec(neg_graph, mfgs, "embedding")
                    pos_pred = finetune_model(pos_u, pos_v)
                    neg_pred = finetune_model(neg_u, neg_v)
                    pred = torch.cat((pos_pred, neg_pred), 0)  # (2*batch, 1)
                    pos_label = pos_graph.edata['target_event']
                    neg_label = torch.zeros((neg_graph.num_edges(), pos_label.shape[1])).to(self.device)
                    label = torch.cat((pos_label, neg_label), 0)
                    val_loss = loss_func(pred, label)
            print(f"batch{i}/{epoch}  --------------train loss:{train_loss}, val loss:{val_loss}")
            # writer.add_scalar('finetune train_loss/batch', train_loss, i)
            # writer.add_scalar('finetune val_loss/batch', val_loss, i)

        self.model = finetune_model

    def infer(self, test_event, batch_size=200, topK=5):
        # 对每个user_idx，计算与所有user的score, 输出最大的
        user_num = self.graph.num_nodes()   # 图节点个数
        candidate_voter_list = []
        inviter_id_list = test_event['inviter_id'].tolist()
        event_id_list = test_event['event_id'].tolist()
        for i in range(test_event.shape[0]):
            user_idx = inviter_id_list[i]
            event_id = event_id_list[i]
            scores = torch.zeros((user_num, )).to(self.device)
            for j in range(self.graph.num_nodes()//batch_size + 1):
                with torch.no_grad():
                    batch_idx = torch.arange(j*batch_size, min((j+1)*batch_size, user_num)).to(self.device)
                    v = self.graph.ndata["embedding"][batch_idx]
                    u = self.graph.ndata["embedding"][user_idx].repeat((v.shape[0], 1))
                    output = self.model(u, v)
                    scores[batch_idx] = output[:, event_id]
            scores = scores.to("cpu").numpy()
            white_list = [user_idx]
            predictions = []
            pred_idx = np.argsort(scores)[::-1]

            k = 0
            while len(predictions) < topK:
                if pred_idx[k] not in white_list:
                    predictions.append(pred_idx[k])
                k += 1
            candidate_voter_list.append(predictions)
        test_event['candidate_voter_list'] = candidate_voter_list

        return test_event

    def get_subgraph_vec(self, subgraph, mfgs, feat):
        u, v = subgraph.edges()
        feat = mfgs[0].srcdata[feat]
        u, v = feat[u], feat[v]
        return u, v

    @property
    def meta_data(self):
        return self._meta_data

    @property
    def val_data(self):
        return self._val_data








