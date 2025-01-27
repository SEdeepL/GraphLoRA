import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import ipdb

class GCNWithSubgraphs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNWithSubgraphs, self).__init__()

        self.linear1 = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.subgraph_conv = GCNConv(in_channels, hidden_channels)

        self.global_conv = GCNConv(in_channels, hidden_channels)

        self.fc = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, subgraph_data_list, global_data):

        # ipdb.set_trace()
        subgraph_embeddings = []
        for subgraph_data in subgraph_data_list:
            subnode_feature = self.linear1(torch.cat(subgraph_data.x,subgraph_data.node_attri))
            x = F.relu(self.subgraph_conv(subnode_feature, subgraph_data.edge_index))
            pooled_x = global_mean_pool(x, subgraph_data.batch)
            global_data.x[subgraph_data.index-1] = global_data.x[subgraph_data.index-1] + pooled_x
            subgraph_embeddings.append(pooled_x)


        # combined_subgraph_embedding = torch.stack(subgraph_embeddings).mean(dim=0)


        node_feature = self.linear2(torch.cat(global_data.x,global_data.node_attri))
        global_x = F.relu(self.global_conv(node_feature, global_data.edge_index))
        global_embedding = global_mean_pool(global_x, global_data.batch)


        # combined_embedding = torch.cat([combined_subgraph_embedding, global_embedding], dim=-1)

        out = self.fc(global_embedding)
        return out


ipdb.set_trace()

global_node_features = torch.randn(8, 3)
global_node_attributions = torch.randn(8, 3)
global_edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                                  [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]], dtype=torch.long)
global_data = Data(x=global_node_features, edge_index=global_edge_index, node_attri = global_node_attributions, batch=torch.zeros(global_node_features.size(0), dtype=torch.long))

subgraph_index = [2,5,6]
subgraph_data_list = []
for i in range(3): 
    subgraph_node_features = torch.randn(4, 3) 
    subgraph_node_attributions = torch.randn(4, 3)
    subgraph_edge_index = torch.tensor([[0, 1, 2, 3],
                                        [1, 0, 3, 2]], dtype=torch.long) 
    subgraph_batch = torch.zeros(subgraph_node_features.size(0), dtype=torch.long)  # batch 索引
    subgraph_data = Data(x=subgraph_node_features, edge_index=subgraph_edge_index, node_attri = subgraph_node_attributions, batch=subgraph_batch, index = subgraph_index[i])

    subgraph_data_list.append(subgraph_data)


model = GCNWithSubgraphs(in_channels=3, hidden_channels=3, out_channels=2)


out = model(subgraph_data_list, global_data)
print(out)
