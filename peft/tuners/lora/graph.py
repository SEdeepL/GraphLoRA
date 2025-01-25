import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import ipdb
# 定义图神经网络
class GCNWithSubgraphs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNWithSubgraphs, self).__init__()
        # 用于处理子图的图卷积层
        self.subgraph_conv = GCNConv(in_channels, hidden_channels)
        # 用于处理全局图的图卷积层
        self.global_conv = GCNConv(in_channels, hidden_channels)
        # 用于整合子图和全局图特征的全连接层
        self.fc = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, subgraph_data_list, global_data):
        # 处理子图特征
        ipdb.set_trace()
        subgraph_embeddings = []
        for subgraph_data in subgraph_data_list:
            x = F.relu(self.subgraph_conv(subgraph_data.x, subgraph_data.edge_index))
            pooled_x = global_mean_pool(x, subgraph_data.batch)
            global_data.x[subgraph_data.index-1] = global_data.x[subgraph_data.index-1] + pooled_x
            subgraph_embeddings.append(pooled_x)

        # 将所有子图的嵌入合并
        # combined_subgraph_embedding = torch.stack(subgraph_embeddings).mean(dim=0)

        # 处理全局图特征
        global_x = F.relu(self.global_conv(global_data.x, global_data.edge_index))
        global_embedding = global_mean_pool(global_x, global_data.batch)

        # 合并子图和全局图的特征
        # combined_embedding = torch.cat([combined_subgraph_embedding, global_embedding], dim=-1)

        # 最终的输出（分类）
        out = self.fc(global_embedding)
        return out

# 示例数据创建
ipdb.set_trace()
# 创建一个总图的特征
global_node_features = torch.randn(8, 3)  # 8个节点，每个节点有3个特征
global_edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                                  [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]], dtype=torch.long)
global_data = Data(x=global_node_features, edge_index=global_edge_index, batch=torch.zeros(global_node_features.size(0), dtype=torch.long))

# 创建多个子图的特征
subgraph_index = [2,5,6]
subgraph_data_list = []
for i in range(3):  # 3个子图
    subgraph_node_features = torch.randn(4, 3)  # 每个子图有4个节点，每个节点有3个特征
    subgraph_edge_index = torch.tensor([[0, 1, 2, 3],
                                        [1, 0, 3, 2]], dtype=torch.long)  # 子图的边
    subgraph_batch = torch.zeros(subgraph_node_features.size(0), dtype=torch.long)  # batch 索引
    subgraph_data = Data(x=subgraph_node_features, edge_index=subgraph_edge_index, batch=subgraph_batch, index = subgraph_index[i])

    subgraph_data_list.append(subgraph_data)

# 模型初始化
model = GCNWithSubgraphs(in_channels=3, hidden_channels=3, out_channels=2)

# 前向传播
out = model(subgraph_data_list, global_data)
print(out)
