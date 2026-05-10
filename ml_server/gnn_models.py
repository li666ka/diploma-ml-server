"""GNN architectures: GraphSAGE + GIN (configurable layers and pooling)."""
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential, ModuleList


class GraphSAGEClassifier(torch.nn.Module):
    """N-layer GraphSAGE з configurable pooling."""

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        num_layers: int = 3,
        pooling: str = "mean",
        aggregator: str = "mean",
    ):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        self.num_layers = num_layers
        self.pooling = pooling

        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggregator))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        self.convs.append(SAGEConv(hidden_dim, out_dim, aggr=aggregator))

        self.dropout = dropout
        self.classifier = Linear(out_dim, num_classes)

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index).relu()
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.classifier(x)


class GINClassifier(torch.nn.Module):
    """N-layer GIN з configurable pooling."""

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        num_layers: int = 3,
        pooling: str = "sum",
    ):
        super().__init__()
        from torch_geometric.nn import GINConv

        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        self.num_layers = num_layers
        self.pooling = pooling

        self.convs = ModuleList()

        def make_mlp(in_d, out_d, with_bn=True):
            layers = [Linear(in_d, out_d)]
            if with_bn:
                layers.append(BatchNorm1d(out_d))
            layers.extend([ReLU(), Linear(out_d, out_d), ReLU()])
            return Sequential(*layers)

        self.convs.append(GINConv(make_mlp(in_dim, hidden_dim), train_eps=True))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(make_mlp(hidden_dim, hidden_dim), train_eps=True))
        self.convs.append(GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, out_dim),
                ReLU(),
            ),
            train_eps=True,
        ))

        self.dropout = dropout
        self.classifier = Linear(out_dim, num_classes)

    def forward(self, data):
        from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "sum":
            x = global_add_pool(x, batch)
        elif self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.classifier(x)


def build_gnn_model(
    architecture: str,
    in_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    num_layers: int = 3,
    pooling: str | None = None,
    aggregator: str = "mean",
) -> torch.nn.Module:
    """Factory: 'sage' or 'gin' з configurable params."""
    if architecture == "sage":
        return GraphSAGEClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim // 2,
            dropout=dropout,
            num_layers=num_layers,
            pooling=pooling or "mean",
            aggregator=aggregator,
        )
    elif architecture == "gin":
        return GINClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim // 2,
            dropout=dropout,
            num_layers=num_layers,
            pooling=pooling or "sum",
        )
    else:
        raise ValueError(f"Unknown GNN architecture: {architecture}")
