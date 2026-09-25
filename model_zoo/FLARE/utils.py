import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")
from dgllife.model import GCN, GAT
import dgl
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class MSEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        d_hidden: int,
        d_out: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
        pool: str | None = None,
        projection_dims: list[int] | None = None,
    ):
        super().__init__()
        projection_dims = list(projection_dims or [d_model])
        if not projection_dims or projection_dims[-1] != d_model:
            raise ValueError("The final formula-projection width must equal d_model.")
        projection_layers = []
        width = d_in
        for index, next_width in enumerate(projection_dims):
            projection_layers.append(nn.Linear(width, next_width))
            if index + 1 < len(projection_dims):
                projection_layers.append(nn.ReLU())
            width = next_width
        self.input_proj = nn.Sequential(*projection_layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_hidden,
            dropout=dropout,
            batch_first=True,  # expects (N, L, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, d_out)
        self.pool = pool


    def forward(self, ms: dict) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ms: dict with
                'tokens': (N, L, d_in)
                'mask':   (N, L) boolean, True = ignore (pytorch convention)
        Returns:
            pool=None:  (node_embeddings, node_masks) → (N, L, d_out), (N, L)
            pool='mean': (N, d_out) mean-pooled representation
            pool='max':  (N, d_out) max-pooled representation
        """
        x = self.input_proj(ms['tokens'])                         # (N, L, d_model)
        x = self.transformer(x, src_key_padding_mask=ms['mask'])  # (N, L, d_model)
        x = self.output_proj(x)                                   # (N, L, d_out)

        # Switch mask to True = keep convention
        peak_masks = ~ms['mask']                                   # (N, L)

        if self.pool == "mean":
            h = (x * peak_masks.unsqueeze(-1)).sum(dim=1) / peak_masks.sum(dim=1, keepdim=True).clamp(min=1)
            return h
        elif self.pool == "max":
            h = x.masked_fill(~peak_masks.unsqueeze(-1), float('-inf')).max(dim=1).values
            return h
        elif self.pool is None:
            out = x, peak_masks
            return out
        else:
            raise ValueError(f"Unsupported pooling method: {self.pool}")

class MolEncoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 gnn_type = "gcn",
                 gnn_dropout = 0.0,
                 mlp_dropout = 0.0,
                 gnn_channels = [64,128,256],
                 attn_heads = [12,12,12],
                 gnn_hidden_dim = 512,
                 pool: str | None = None,
                 ):
        super().__init__()

        dropout = [gnn_dropout for _ in range(len(gnn_channels))]
        batchnorm = [True for _ in range(len(gnn_channels))]
        gnn_map = {
            "gcn": GCN(in_dim, gnn_channels, batchnorm = batchnorm, dropout = dropout),
            "gat": GAT(in_dim, gnn_channels, attn_heads)
        }
        self.GNN = gnn_map[gnn_type]
        self.pool = pool
        
        if pool is not None:
            self.out_mlp = nn.Sequential(
                nn.Linear(gnn_channels[-1], gnn_hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(gnn_hidden_dim * 2, out_dim),
            )
        else:
            self.out_linear = nn.Linear(gnn_channels[-1], out_dim)
    
    def _dgl_to_node_embeddings(self, g) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract node embeddings and masks from a batched DGL graph.
        Returns:
            node_embeddings: (B, N_max, d)
            node_masks:      (B, N_max) — True = valid node
        """
        counts = g.batch_num_nodes().to(g.ndata['f'].device)
        embeddings = torch.split(g.ndata['f'], counts.tolist())
        node_embeddings = pad_sequence(embeddings, batch_first=True)
        node_masks = (
            torch.arange(node_embeddings.shape[1], device=node_embeddings.device)
            .unsqueeze(0)
            < counts.unsqueeze(1)
        )

        return node_embeddings, node_masks

    def forward(self, g) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # GNN forward pass
        f = self.GNN(g, g.ndata['h'])
        g.ndata['f'] = f

        # Convert DGL graph to node embeddings and masks
        node_embeddings, node_masks = self._dgl_to_node_embeddings(g)

        # Apply pooling using node_embeddings and node_masks
        if self.pool == "max":
            # Set masked-out nodes to -inf so they don't win the max
            masked = node_embeddings.masked_fill(~node_masks.unsqueeze(-1), float('-inf'))
            out = masked.max(dim=1).values
            out = self.out_mlp(out)
        elif self.pool == "mean":
            # Zero out masked nodes and average over valid ones
            masked = node_embeddings * node_masks.unsqueeze(-1)
            out = masked.sum(dim=1) / node_masks.sum(dim=1, keepdim=True).clamp(min=1)
            out = self.out_mlp(out)
        elif self.pool == "sum":
            masked = node_embeddings * node_masks.unsqueeze(-1)
            out = masked.sum(dim=1)
            out = self.out_mlp(out)
        elif self.pool is None:
            node_embeddings = self.out_linear(node_embeddings)
            out = node_embeddings, node_masks
            
        else:
            raise ValueError(f"Unsupported pooling method: {self.pool}")
        
        return out

def batch_infonce_from_logits(logits):
    labels = torch.arange(len(logits)).to(logits.device)  # (B,)
    loss_x_to_y = nn.CrossEntropyLoss()(logits, labels)
    loss_y_to_x = nn.CrossEntropyLoss()(logits.T, labels)
    loss = (loss_x_to_y + loss_y_to_x) / 2
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, acc


def bidirectional_batch_infonce(
    spectra_to_molecules: torch.Tensor,
    molecules_to_spectra: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Symmetric InfoNCE using the two genuinely directional FLARE scores.

    Both matrices use ``[spectrum, molecule]`` indexing. The second matrix is
    transposed only when molecules become the contrastive anchors.
    """
    if spectra_to_molecules.shape != molecules_to_spectra.shape:
        raise ValueError("The two directional score matrices must have the same shape.")
    labels = torch.arange(spectra_to_molecules.shape[0], device=spectra_to_molecules.device)
    loss_s_to_m = F.cross_entropy(spectra_to_molecules, labels)
    loss_m_to_s = F.cross_entropy(molecules_to_spectra.T, labels)
    loss = 0.5 * (loss_s_to_m + loss_m_to_s)
    combined = 0.5 * (spectra_to_molecules + molecules_to_spectra)
    accuracy = (combined.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy

def candidate_infonce_from_logits(logits):
    log_probs = F.log_softmax(logits, dim=-1)  # (B, K)
    loss = -log_probs[:, 0].mean()
    acc = log_probs.argmax(dim=-1).eq(0).float().mean().item()  # Assuming the first candidate is the positive one
    return loss, acc

def candidate_retrieval_accuracy_from_logits(logits, candidate_mask=None):
    """Compute conservative candidate-set retrieval metrics from logits.

    ``candidate_mask`` excludes ragged-batch padding.  The first candidate of
    every row is the target; tied valid negatives are counted ahead of it by
    the shared metric implementation.
    """
    from models.retrieval import retrieval_metrics_from_scores

    if candidate_mask is None:
        candidate_mask = torch.ones_like(logits, dtype=torch.bool)
    return {
        name: value.item()
        for name, value in retrieval_metrics_from_scores(logits, candidate_mask).items()
    }
