from data_loader import *
from models import *
from samplers import *
from utils import *
from loss_function import *
import torch
import torch.nn as nn
import dgl
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    data: GraphDataset
    model: torch.nn.Module

    def __init__(self, model_name, pos_sampler, neg_sampler, loss, device):
        self.model_name = model_name
        self.sampler = dgl.dataloading.as_edge_prediction_sampler(pos_sampler, negative_sampler=neg_sampler)
        self.loss = loss
        self.device = device

    def data_prepare(self, events, users):
        self.data = GraphDataset(events, users)
        self.data.process()

    def model_init(self, model_config):
        if self.model_name == "Sage":
            inp, emb, out = model_config["input"], model_config["embedding"], model_config["output"]
            self.model = SageModel(inp, emb, out).to(self.device)

    def train(self, train_config):
        graph = self.data[0].to(self.device)

        bs = train_config["batch_size"]
        opt = train_config["opt"]
        iter = train_config["iter"]


        train_loader = dgl.dataloading.DataLoader(
            graph,
            torch.arange(graph.num_nodes()).to(self.device),  # The edges to iterate over
            self.sampler,
            device=self.device,
            batch_size=bs,  # Batch size
            shuffle=True,  # Whether to shuffle the nodes for every epoch
            drop_last=False,
        )

        writer = SummaryWriter("D:/zhan/CAAI-BDSC2023/logs")
        for i in range(iter):
            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(train_loader):
                inputs = mfgs[0].srcdata["user_info"]
                predictions = self.model(mfgs, inputs)  # 得到节点embedding

                predictor = DotPredictor().to(self.device)
                pos_score = predictor(pos_graph, predictions)   # 正向图边的预测得分
                neg_score = predictor(neg_graph, predictions)
                #
                loss = self.loss(pos_score, neg_score)  # 损失值
                if opt == "Adam":
                    opt = torch.optim.Adam(list(self.model.parameters()) + list(predictor.parameters()), weight_decay=0.05)
                else:
                    pass
                print(f"batch{i}/{iter} step{step} --------------loss:{loss}")
                opt.zero_grad()
                loss.backward()
                opt.step()
            writer.add_scalar('loss/batch', loss, i)

    def infer(self, test):
        pass
