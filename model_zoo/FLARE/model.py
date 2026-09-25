from lightning.pytorch import LightningModule
import torch
import dgl
from .utils import MSEncoder, MolEncoder
from transforms.spectra_transforms import Subformula_Transform
from transforms.molecules_transforms import MoleculeToGraph
import torch.nn.functional as F
import torch.nn as nn
from model_zoo.common import optimizer_with_scheduler
from .utils import (
    batch_infonce_from_logits,
    bidirectional_batch_infonce,
    candidate_retrieval_accuracy_from_logits,
)

class FLARE(LightningModule):
    def __init__(self, config):
        super(FLARE, self).__init__()
        
        self.use_max_sim = config['use_max_sim']
        
        mol_transform = MoleculeToGraph()
        self.mol_encoder = MolEncoder(mol_transform.get_dim(),
                                      config['shared_dim'],
                                      gnn_type = config['gnn_type'],
                                      gnn_dropout = config['gnn_dropout'],
                                      mlp_dropout = config['mlp_dropout'],
                                      gnn_channels = config['gnn_channels'],
                                      attn_heads = config['attn_heads'],
                                      gnn_hidden_dim = config['gnn_hidden_dim'],
                                      pool = None if self.use_max_sim else config['pooling_mol'])
        
        ms_transform = Subformula_Transform()
        self.ms_encoder = MSEncoder(d_in = ms_transform.get_dim(),
                                    d_model=config['transformer_d_model'],
                                    d_hidden=config['transformer_d_hidden'],
                                    d_out=config['shared_dim'],
                                    n_layers=config['transformer_n_layers'],
                                    n_heads=config['transformer_n_heads'],
                                    dropout=config['transformer_dropout'],
                                    pool=None if self.use_max_sim else config['pooling_ms'],
                                    projection_dims=config.get('formula_projection'),
                                    )
        
        self.log_epsilon = nn.Parameter(
            torch.log(torch.tensor(float(config['temperature']))),
            requires_grad=bool(config['trainable_temperature']),
        )
        
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.n_warmup_steps = config['n_warmup_steps']
        
        self.save_hyperparameters(config)
        
    def forward(self, ms, mol):
        ms = self.ms_encoder(ms, pooling=None if self.use_max_sim else self.pooling)
        mol = self.mol_encoder(mol)
        return ms, mol
    
    def configure_optimizers(self):
        n_max_steps = int(self.trainer.estimated_stepping_batches)
        return optimizer_with_scheduler(
            parameters=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_warmup_steps=self.n_warmup_steps,
            n_max_steps=n_max_steps
        )

    @staticmethod
    def _packed_to_dgl(packed):
        """Reconstruct a batched DGL graph after ordinary tensors reach the GPU."""
        graph = dgl.graph(
            (packed['src'], packed['dst']),
            num_nodes=int(packed['node_counts'].sum().item()),
            device=packed['h'].device,
        )
        graph.ndata['h'] = packed['h']
        graph.set_batch_num_nodes(packed['node_counts'])
        graph.set_batch_num_edges(packed['edge_counts'])
        return graph

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move IPC-safe packed tensors first, then construct the DGL graph."""
        ms, packed = super().transfer_batch_to_device(batch, device, dataloader_idx)
        graph = self._packed_to_dgl(packed)
        if 'query_counts' in packed:
            return ms, (graph, packed['query_counts'])
        return ms, graph
        
    def robust_stack(self, tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stack a ragged list of tensors into a padded batch.

        Args:
            tensors: list of B tensors, each of shape (c_i, d)
        Returns:
            stacked:  (B, C_max, d) padded tensors
            pad_mask: (B, C_max) boolean, True = padded (invalid)
        """
        B, C_max, d = len(tensors), max(t.shape[0] for t in tensors), tensors[0].shape[1]
        device = tensors[0].device

        stacked  = torch.zeros(B, C_max, d, device=device)
        pad_mask = torch.ones(B, C_max, dtype=torch.bool, device=device)  # True = padded

        for i, t in enumerate(tensors):
            stacked[i, :t.shape[0]]  = t
            pad_mask[i, :t.shape[0]] = False

        return stacked, pad_mask
    
    def loss(self, ms, mol):
        
        if not self.use_max_sim:
            ms = F.normalize(self.ms_encoder(ms), p=2, dim=1)
            mol = F.normalize(self.mol_encoder(mol), p=2, dim=1)
            logits = torch.matmul(ms, mol.t()) / self.log_epsilon.exp()
        else:
            ms_tokens, peak_masks = self.ms_encoder(ms)
            mol_nodes, node_masks = self.mol_encoder(mol)
            ms_tokens  = F.normalize(ms_tokens,  p=2, dim=-1)
            mol_nodes  = F.normalize(mol_nodes,  p=2, dim=-1)

            # All pairwise peak-node cosine similarities.
            # ms_tokens = (spec, peak, dim)
            # mol_nodes = (mol, node, dim)
            similarities = torch.einsum('spd,mnd->smpn', ms_tokens, mol_nodes)

            # h(S,M): average over peaks of each peak's best matching atom.
            spectra_to_molecules = similarities.masked_fill(
                ~node_masks.unsqueeze(0).unsqueeze(2), float('-inf')
            ).max(dim=-1).values
            spectra_to_molecules = spectra_to_molecules.masked_fill(
                ~peak_masks.unsqueeze(1), 0.0
            )
            spectra_to_molecules = spectra_to_molecules.sum(dim=-1) / peak_masks.sum(
                dim=-1, keepdim=True
            ).clamp(min=1)

            # h(M,S): average over atoms of each atom's best matching peak.
            molecules_to_spectra = similarities.masked_fill(
                ~peak_masks.unsqueeze(1).unsqueeze(-1), float('-inf')
            ).max(dim=-2).values
            molecules_to_spectra = molecules_to_spectra.masked_fill(
                ~node_masks.unsqueeze(0), 0.0
            )
            molecules_to_spectra = molecules_to_spectra.sum(dim=-1) / node_masks.sum(
                dim=-1
            ).clamp(min=1).unsqueeze(0)

            temperature = self.log_epsilon.exp()
            return bidirectional_batch_infonce(
                spectra_to_molecules / temperature,
                molecules_to_spectra / temperature,
            )

        return batch_infonce_from_logits(logits)

    def retrieval_accuracy(self, ms, candidates):

        candidate_graph, query_counts = candidates
        batchsize = len(query_counts)
        n_max_candidates = int(query_counts.max().item())
        # Invalid padded candidate positions must never outrank real candidates,
        # whose cosine similarities may legitimately be negative.
        logits = torch.full(
            (batchsize, n_max_candidates),
            fill_value=float('-inf'),
            device=ms['tokens'].device,
        )
        candidate_mask = torch.zeros_like(logits, dtype=torch.bool)
        if not self.use_max_sim:
            ms = F.normalize(self.ms_encoder(ms), p=2, dim=1)
            all_candidates = F.normalize(self.mol_encoder(candidate_graph), p=2, dim=-1)
            for i, cands_i in enumerate(torch.split(all_candidates, query_counts.tolist())):
                logits_i = torch.einsum('d,cd->c', ms[i], cands_i) / self.log_epsilon.exp()
                logits[i, :cands_i.size(0)] = logits_i
                candidate_mask[i, :cands_i.size(0)] = True
        else:
            ms_tokens, peak_masks = self.ms_encoder(ms)
            ms_tokens = F.normalize(ms_tokens, p=2, dim=-1)
            all_nodes, all_masks = self.mol_encoder(candidate_graph)
            encoded = list(zip(
                torch.split(all_nodes, query_counts.tolist(), dim=0),
                torch.split(all_masks, query_counts.tolist(), dim=0),
            ))
            max_nodes = max(nodes.shape[1] for nodes, _ in encoded)
            d = encoded[0][0].shape[-1]
            mol_nodes = ms_tokens.new_zeros(batchsize, n_max_candidates, max_nodes, d)
            node_masks = torch.zeros(
                batchsize, n_max_candidates, max_nodes,
                dtype=torch.bool, device=ms_tokens.device,
            )
            for i, (nodes, masks) in enumerate(encoded):
                n_candidates, n_nodes = masks.shape
                mol_nodes[i, :n_candidates, :n_nodes] = nodes
                node_masks[i, :n_candidates, :n_nodes] = masks
                candidate_mask[i, :n_candidates] = True

            mol_nodes = F.normalize(mol_nodes, p=2, dim=-1)
            similarities = torch.einsum('bpd,bcnd->bcpn', ms_tokens, mol_nodes)
            spectra_to_molecules = similarities.masked_fill(
                ~node_masks.unsqueeze(2), float('-inf')
            ).max(dim=-1).values
            spectra_to_molecules = spectra_to_molecules.masked_fill(
                ~peak_masks.unsqueeze(1), 0.0
            ).sum(dim=-1) / peak_masks.sum(dim=-1, keepdim=True).clamp(min=1)

            molecules_to_spectra = similarities.masked_fill(
                ~peak_masks[:, None, :, None], float('-inf')
            ).max(dim=-2).values
            molecules_to_spectra = molecules_to_spectra.masked_fill(
                ~node_masks, 0.0
            ).sum(dim=-1) / node_masks.sum(dim=-1).clamp(min=1)
            logits = 0.5 * (spectra_to_molecules + molecules_to_spectra)
            logits = logits.masked_fill(~candidate_mask, float('-inf'))
        log = candidate_retrieval_accuracy_from_logits(logits, candidate_mask)
        return log

          
    
    def training_step(self, batch):
        batch_size = batch[0]['tokens'].size(0)
        ms, mol = batch  
        loss, acc = self.loss(ms, mol)
        self.log('loss (train)', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('R@1 - batch (train)', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch):
        batch_size = batch[0]['tokens'].size(0)
        ms, mol = batch  
        loss, acc = self.loss(ms, mol)
        self.log('loss (val)', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('R@1 - batch (val)', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    
    def test_step(self, batch):
        ms, candidates = batch
        batch_size = ms['tokens'].size(0)
        log = self.retrieval_accuracy(ms, candidates)
        log = {f'{k} (test)': v for k, v in log.items()}
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size
        )
        return log['R@1 (test)']  # Return hard candidate top1 accuracy for checkpointing
