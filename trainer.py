from data_loader import *
from models import *
from samplers import *
from utils import *
from loss_function import *
import torch
import torch.nn as nn
import dgl
import random
from  tqdm import tqdm


class Trainer:
    model: torch.nn.Module
    graph: dgl.DGLGraph
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

    def data_prepare(self, source, target, users):
        print("======Graph data preparing=======")
        data = GraphDataset(source, target, users)
        data.process()
        self.graph = data[0].to(self.device) # 图先放在cpu, 训练阶段将子图放gpu
        print("======prepare finished=======")
        print(f"当前显存占用{torch.cuda.memory_allocated()}")
        show_graph_info(self.graph)

    def pretrain(self, pretrain_config):
        # params
        loss_func = pretrain_config.get("loss", multi_label_loss)
        epoch = pretrain_config.get("epoch", 10)
        batch_size = pretrain_config.get("batch_size", 8)
        opt = pretrain_config.get("opt", "Adam")
        sample_neighbor = pretrain_config.get("opt", [6, 10])

        # model init
        if self.pretrain_model == "Sage":
            inp, emb, out = pretrain_config["input"], pretrain_config["embedding"], pretrain_config["output"]
            pretrain_model = SageModel(inp, emb, out).to(self.device)
            n_class = pretrain_config["n_class"]
            head = EdgeClassifyHead(out, n_class).to(self.device)

        # 将预训练抽样器用neighbor sampler, 参数传入
        sampler = dgl.dataloading.NeighborSampler(sample_neighbor)

        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude="self")
        train_loader = dgl.dataloading.DataLoader(
                    self.graph,
                    torch.arange(self.graph.num_edges()).to(self.device),  # The edges to iterate over
                    sampler,
                    device=self.device,
                    batch_size=batch_size,  # Batch size
                    shuffle=True,  # Whether to shuffle the nodes for every epoch.to(self.device)
                    drop_last=False,
                )

        # training loop
        print("\n\n start pretraining")
        for i in range(epoch):
            epoch_loss = 0
            for step, (input_nodes, subgraph, mfgs) in enumerate(train_loader):
                inputs = mfgs[0].srcdata["user_info"]
                predictions = pretrain_model(mfgs, inputs)
                score = head(subgraph, predictions)
                source_mask, edge_label = subgraph.edata["source_mask"], subgraph.edata["source_events"]
                loss = loss_func(score[source_mask], edge_label[source_mask])
                with torch.no_grad():
                    epoch_loss += loss

                if opt == "Adam":
                    opt = torch.optim.Adam(list(pretrain_model.parameters()) + list(head.parameters()))
                else:
                    pass
                opt.zero_grad()
                loss.backward()
                opt.step()
            print(f"batch{i}/{epoch}  --------------mean loss:{epoch_loss/(step+1)}")

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

        print("calculating node embedding")
        node_embeddings = torch.zeros((self.graph.num_nodes(), out)).to(self.device)
        with tqdm(node_loader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                with torch.no_grad():
                    inputs = mfgs[0].srcdata["user_info"]
                    outputs = pretrain_model(mfgs, inputs)
                    node_embeddings[input_nodes] = outputs
        self.graph.ndata["embedding"] = node_embeddings



    def finetune(self, finetune_config):
        pass


