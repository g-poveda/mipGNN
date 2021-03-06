import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import os.path as osp
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleBipartiteLayer(MessagePassing):
    def __init__(self, edge_dim, dim):
        super(SimpleBipartiteLayer, self).__init__(aggr="add", flow="source_to_target")

        self.edge_encoder = Sequential(Linear(edge_dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                                       BN(dim))

        self.mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, source, target, edge_index, edge_attr, size):
        # Map edge features to embeddings with the same number of components as node embeddings.
        edge_embedding = self.edge_encoder(edge_attr)

        tmp = self.propagate(edge_index, x=source, edge_attr=edge_embedding, size=size)

        out = self.mlp((1 + self.eps) * target + tmp)

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        reset(self.node_encoder)
        reset(self.edge_encoder)
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden):
        super(SimpleNet, self).__init__()

        # Embed initial node features.
        self.var_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))
        self.con_node_encoder = Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden))

        # Bipartite GNN architecture.
        self.var_con_1 = SimpleBipartiteLayer(1, hidden)
        self.con_var_1 = SimpleBipartiteLayer(1, hidden)
        self.var_con_2 = SimpleBipartiteLayer(1, hidden)
        self.con_var_2 = SimpleBipartiteLayer(1, hidden)
        self.var_con_3 = SimpleBipartiteLayer(1, hidden)
        self.con_var_3 = SimpleBipartiteLayer(1, hidden)
        self.var_con_4 = SimpleBipartiteLayer(1, hidden)
        self.con_var_4 = SimpleBipartiteLayer(1, hidden)

        # MLP used for classification.
        self.lin1 = Linear(5*hidden, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.lin3 = Linear(hidden, hidden)
        self.lin4 = Linear(hidden, 2)

    def reset_parameters(self):
        self.var_con_1.reset_parameters()
        self.con_var_1.reset_parameters()
        self.var_con_2.reset_parameters()
        self.con_var_2.reset_parameters()
        self.var_con_3.reset_parameters()
        self.con_var_3.reset_parameters()
        self.var_con_4.reset_parameters()
        self.con_var_4.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, data):

        # Get data of batch.
        var_node_features = data.var_node_features
        con_node_features = data.con_node_features
        edge_index_var = data.edge_index_var
        edge_index_con = data.edge_index_con
        edge_features_var = data.edge_features_var
        edge_features_con = data.edge_features_con
        num_nodes_var = data.num_nodes_var
        num_nodes_con = data.num_nodes_con

        # Compute initial node embeddings.
        var_node_features_0 = self.var_node_encoder(var_node_features)
        con_node_features_0 = self.con_node_encoder(con_node_features)


        con_node_features_1 = F.relu(self.var_con_1(var_node_features_0, con_node_features_0, edge_index_var, edge_features_var,
                           (num_nodes_var.sum(), num_nodes_con.sum())))
        var_node_features_1 = F.relu(self.con_var_1(con_node_features_1, var_node_features_0, edge_index_con, edge_features_con,
                           (num_nodes_con.sum(), num_nodes_var.sum())))

        con_node_features_2 = F.relu(self.var_con_2(var_node_features_1, con_node_features_1, edge_index_var, edge_features_var,
                           (num_nodes_var.sum(), num_nodes_con.sum())))
        var_node_features_2 = F.relu(self.con_var_2(con_node_features_2, var_node_features_1, edge_index_con, edge_features_con,
                           (num_nodes_con.sum(), num_nodes_var.sum())))

        con_node_features_3 = F.relu(self.var_con_3(var_node_features_2, con_node_features_2, edge_index_var, edge_features_var,
                           (num_nodes_var.sum(), num_nodes_con.sum())))
        var_node_features_3 = F.relu(self.con_var_3(con_node_features_3, var_node_features_2, edge_index_con, edge_features_con,
                           (num_nodes_con.sum(), num_nodes_var.sum())))

        con_node_features_4 = F.relu(self.var_con_4(var_node_features_3, con_node_features_3, edge_index_var, edge_features_var,
                           (num_nodes_var.sum(), num_nodes_con.sum())))
        var_node_features_4 = F.relu(self.con_var_4(con_node_features_4, var_node_features_3, edge_index_con, edge_features_con,
                           (num_nodes_con.sum(), num_nodes_var.sum())))

        x = torch.cat([var_node_features_0,var_node_features_1,var_node_features_2,var_node_features_3,var_node_features_4], dim=-1)

        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin3(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

# Preprocessing to create Torch dataset.
class GraphDataset(InMemoryDataset):
    def __init__(self, root, data_path, bias_threshold, transform=None, pre_transform=None,
                 pre_filter=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_path = data_path
        self.bias_threshold = bias_threshold

    @property
    def raw_file_names(self):
        return "SET2_SPO"

    @property
    def processed_file_names(self):
        return "SET2_SPO"

    def download(self):
        pass

    def process(self):
        print("Preprocessing.")

        data_list = []
        num_graphs = len(os.listdir(data_path))

        # Iterate over instance files and create data objects.
        for num, filename in enumerate(os.listdir(data_path)):
            print(filename, num, num_graphs)

            # Get graph.
            graph = nx.read_gpickle(data_path + filename)

            # Make graph directed.
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.to_directed() if not nx.is_directed(graph) else graph
            data = Data()

            #  Maps networkx ids to new variable node ids.
            node_to_varnode = {}
            #  Maps networkx ids to new constraint node ids.
            node_to_connode = {}

            # Number of variables.
            num_nodes_var = 0
            # Number of constraints.
            num_nodes_con = 0
            # Targets (classes).
            y = []
            # Features for variable nodes.
            feat_var = []
            # Feature for constraints nodes.
            feat_con = []
            # Right-hand sides of equations.
            feat_rhs = []

            # Iterate over nodes, and collect features.
            for i, (node, node_data) in enumerate(graph.nodes(data=True)):
                # Node is a variable node.
                if node_data['bipartite'] == 0:
                    node_to_varnode[i] = num_nodes_var
                    num_nodes_var += 1

                    if (node_data['bias'] < bias_threshold):
                        y.append(0)
                    else:
                        y.append(1)

                    feat_var.append([node_data['objcoeff'], graph.degree[i]])

                # Node is constraint node.
                elif node_data['bipartite'] == 1:
                    node_to_connode[i] = num_nodes_con
                    num_nodes_con += 1

                    rhs = node_data['rhs']
                    feat_rhs.append(rhs)
                    feat_con.append([rhs, graph.degree[i]])
                else:
                    print("Error in graph format.")
                    exit(-1)

            # Edge list for var->con graphs.
            edge_list_var = []
            # Edge list for con->var graphs.
            edge_list_con = []

            # Create features matrices for variable nodes.
            edge_features_var = []
            # Create features matrices for constraint nodes.
            edge_features_con = []

            # Remark: graph is directed, i.e., each edge exists for each direction.
            # Flow of messages: source -> target.
            for i, (s, t, edge_data) in enumerate(graph.edges(data=True)):
                # Source node is con, target node is var.

                if graph.nodes[s]['bipartite'] == 1:
                    # Source node is constraint. C->V.
                    edge_list_con.append([node_to_connode[s], node_to_varnode[t]])
                    edge_features_con.append([edge_data['coeff']])
                else:
                    # Source node is variable. V->C.
                    edge_list_var.append([node_to_varnode[s], node_to_connode[t]])
                    edge_features_var.append([edge_data['coeff']])

            edge_index_var = torch.tensor(edge_list_var).t().contiguous()
            edge_index_con = torch.tensor(edge_list_con).t().contiguous()

            # Create data object.
            data.edge_index_var = edge_index_var
            data.edge_index_con = edge_index_con

            data.y = torch.from_numpy(np.array(y)).to(torch.long)
            data.var_node_features = torch.from_numpy(np.array(feat_var)).to(torch.float)
            data.con_node_features = torch.from_numpy(np.array(feat_con)).to(torch.float)
            data.rhs = torch.from_numpy(np.array(feat_rhs)).to(torch.float)
            data.edge_features_con = torch.from_numpy(np.array(edge_features_con)).to(torch.float)
            data.edge_features_var = torch.from_numpy(np.array(edge_features_var)).to(torch.float)
            data.num_nodes_var = num_nodes_var
            data.num_nodes_con = num_nodes_con

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# Preprocess indices of bipartite graphs to make batching work.
class MyData(Data):
    def __inc__(self, key, value):
        if key in ['edge_index_var']:
            return torch.tensor([self.num_nodes_var, self.num_nodes_con]).view(2, 1)
        elif key in ['edge_index_con']:
            return torch.tensor([self.num_nodes_con, self.num_nodes_var]).view(2, 1)
        else:
            return 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


# Prepare data.
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'DS')
# Path to raw graph data.
data_path = '../DATA1/er_SET2/300_300/alpha_0.25_setParam_100/train/'
# Threshold for computing class labels.
bias_threshold = 0.05
# Create dataset.
dataset = GraphDataset(path, data_path, bias_threshold, transform=MyTransform())#.shuffle()
len(dataset)


# TODO: Fixed split for testing purposes.
# Split data.
l = len(dataset)
train_index, rest = train_test_split(list(range(0, l)), test_size=0.2)
l = len(rest)
val_index = rest[0:int(l / 2)]
test_index = rest[int(l / 2):]


train_dataset = dataset[train_index].shuffle()
val_dataset = dataset[val_index].shuffle()
test_dataset = dataset[test_index].shuffle()

# TODO: Do not change this.
# np.savetxt("index_er_200_SET2_1k_20", test_index, delimiter=",", fmt="%d")

print(len(val_dataset))
print(len(test_dataset))
print(1 - test_dataset.data.y.sum().item() / test_dataset.data.y.size(-1))

# Prepare batch loaders.
batch_size = 15
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


print("### DATA LOADED.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet(hidden=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Play with this.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.8, patience=10,
                                                       min_lr=0.0000001)
print("### SETUP DONE.")

def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += batch_size * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    l = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).float().mean().item()
        l += 1

    return correct / l


best_val = 0.0
test_acc = 0.0
for epoch in range(1, 100):

    train_loss = train(epoch)
    train_acc = test(train_loader)

    val_acc = test(val_loader)
    scheduler.step(val_acc)
    lr = scheduler.optimizer.param_groups[0]['lr']

    if val_acc > best_val:
        best_val = val_acc
        test_acc = test(test_loader)

    # Break if learning rate is smaller 10**-6.
    if lr < 0.000001:
        break

    print('Epoch: {:03d}, LR: {:.7f}, Train Loss: {:.7f},  '
          'Train Acc: {:.7f}, Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, lr, train_loss,
                                                       train_acc, val_acc, test_acc))

#torch.save(model.state_dict(), "trained_model_er_200_SET2_1k")
