"""GNN architectures: GraphSAGE + GIN."""
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential


# Imported lazily inside model classes to allow CPU-only environments
# where torch_geometric isn't installed (for tests etc.)


class GraphSAGEClassifier(torch.nn.Module):
    """3-layer GraphSAGE + global mean pool."""

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        from torch_geometric.nn import SAGEConv  # lazy import

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout
        self.classifier = Linear(out_dim, num_classes)

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GINClassifier(torch.nn.Module):
    """3-layer GIN + global ADD pool. Кращий за SAGE на graph classification."""

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        from torch_geometric.nn import GINConv  # lazy import

        self.conv1 = GINConv(
            Sequential(
                Linear(in_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            ),
            train_eps=True,
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            ),
            train_eps=True,
        )
        self.conv3 = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, out_dim),
                ReLU(),
            ),
            train_eps=True,
        )
        self.dropout = dropout
        self.classifier = Linear(out_dim, num_classes)

    def forward(self, data):
        from torch_geometric.nn import global_add_pool

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        return self.classifier(x)


def build_gnn_model(
    architecture: str,
    in_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> torch.nn.Module:
    """Factory: 'sage' or 'gin'."""
    if architecture == "sage":
        return GraphSAGEClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim // 2,
            dropout=dropout,
        )
    elif architecture == "gin":
        return GINClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim // 2,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown GNN architecture: {architecture}")
