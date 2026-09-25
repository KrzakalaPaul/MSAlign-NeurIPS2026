from lightning.pytorch import LightningModule
import torch
from .utils import MolEncoder, MSEncoder
from transforms.molecules_transforms import MoleculeToGraph
import torch.nn.functional as F
import torch.nn as nn
import dgl
from model_zoo.common import optimizer_with_scheduler, batch_infonce, candidate_infonce
from models.retrieval import candidate_retrieval_accuracy

class JESTR(LightningModule):
    def __init__(self, config, mode='pretrain'):
        super(JESTR, self).__init__()
        
        mol_transform = MoleculeToGraph()
        self.mol_encoder = MolEncoder(mol_transform.get_dim(),
                                      config['shared_dim'],
                                      gnn_type = config['gnn_type'],
                                      gnn_dropout = config['gnn_dropout'],
                                      mlp_dropout = config['mlp_dropout'],
                                      gnn_channels = config['gnn_channels'],
                                      attn_heads = config['attn_heads'],
                                      gnn_hidden_dim = config['gnn_hidden_dim'],
                                      pool = config['pooling'])
        
        self.ms_encoder = MSEncoder(max_mz=config['max_mz'],
                                    bin_width=config['bin_width'],
                                    mlp_dropout=config['mlp_dropout'],
                                    hidden_dim=config['ms_encoder_hidden_dim'],
                                    out_dim=config['shared_dim'])
        
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.07)), requires_grad=config["trainable_temperature"])
        
        self.mode = mode
        self.lr = config['lr']
        self.lr_finetune = config['lr_finetune']
        self.weight_decay = config['weight_decay']
        self.n_warmup_steps = config['n_warmup_steps']
        self.n_warmup_steps_finetune = config['n_warmup_steps_finetune']
        self.candidate_gnn_batch_size = int(config.get('candidate_gnn_batch_size', 1024))
        if self.candidate_gnn_batch_size <= 0:
            raise ValueError("candidate_gnn_batch_size must be positive.")
        
        assert self.mode in ['pretrain', 'finetune'], "Mode must be either 'pretrain' or 'finetune'"
        
        self.save_hyperparameters(config)
        
    def forward(self, ms, mol):
        ms = self.ms_encoder(ms)
        mol = self.mol_encoder(mol)
        return ms, mol
    
    def configure_optimizers(self):
        if self.mode == 'finetune':
            lr = self.lr_finetune
            n_warmup_steps = self.n_warmup_steps_finetune
        else:
            lr = self.lr
            n_warmup_steps = self.n_warmup_steps

        # Both JESTR stages are epoch-based. Lightning knows the corresponding
        # number of optimizer updates after attaching the stage's dataloader.
        n_max_steps = int(self.trainer.estimated_stepping_batches)

        return optimizer_with_scheduler(
            parameters=self.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
            n_warmup_steps=n_warmup_steps,
            n_max_steps=n_max_steps
        )
    
    def stack_with_padding_and_masking(self, tensors, pad_value=0):
        '''
        Stack a list of tensors with padding and create a mask for valid entries.
        tensors: list of (N_i, D) tensors
        Returns:
            padded_tensors: (B, max_N, D) tensor with padded values
            mask: (B, max_N) boolean tensor where True indicates valid entries
        '''
        max_size = max(t.size(0) for t in tensors)
        padded_tensors = []
        mask = torch.zeros((len(tensors), max_size), dtype=torch.bool, device=tensors[0].device)
        for i, t in enumerate(tensors):
            mask[i, :t.size(0)] = True  # Mark valid entries
            if t.size(0) < max_size:
                padding = torch.full((max_size - t.size(0), t.size(1)), pad_value, device=t.device)
                padded_tensors.append(torch.cat([t, padding], dim=0))
            else:
                padded_tensors.append(t)
        return torch.stack(padded_tensors, dim=0), mask

    def encode_candidates(self, candidates):
        """Encode candidates in large, bounded GNN invocations.

        Candidate sets stay intact inside each chunk, so their embeddings can
        be split back into query groups without unbatching individual graphs.
        """

        counts = [candidate_batch.batch_size for candidate_batch in candidates]
        chunk_embeddings = []
        chunk = []
        chunk_size = 0
        for candidate_batch, count in zip(candidates, counts):
            if chunk and chunk_size + count > self.candidate_gnn_batch_size:
                chunk_embeddings.append(self.mol_encoder(dgl.batch(chunk)))
                chunk = []
                chunk_size = 0
            chunk.append(candidate_batch)
            chunk_size += count
        if chunk:
            chunk_embeddings.append(self.mol_encoder(dgl.batch(chunk)))

        flat_embeddings = F.normalize(torch.cat(chunk_embeddings, dim=0), p=2, dim=1)
        per_query_embeddings = torch.split(flat_embeddings, counts)
        return self.stack_with_padding_and_masking(per_query_embeddings)
    
    def weak_loss(self, ms, mol):
        ms = self.ms_encoder(ms)
        mol = self.mol_encoder(mol)
        ms = F.normalize(ms, p=2, dim=1)
        mol = F.normalize(mol, p=2, dim=1)
        loss, acc = batch_infonce(ms, mol, temperature=self.log_epsilon.exp())
        return loss, acc
        
    def strong_loss(self, ms, candidates):
        ms = self.ms_encoder(ms)
        ms = F.normalize(ms, p=2, dim=1) 
        candidates_emb, candidates_mask = self.encode_candidates(candidates)
        
        # First recompute the weak loss
        mol = candidates_emb[:, 0, :]  # Assuming the first candidate is the positive one
        loss_weak, acc_weak = batch_infonce(ms, mol, temperature=self.log_epsilon.exp())
        
        # Then compute the strong loss with all candidates
        loss_strong, acc = candidate_infonce(ms, candidates_emb, candidates_mask, temperature=self.log_epsilon.exp())
        
        loss = 0.9*loss_weak + 0.1*loss_strong
        
        return loss, acc, acc_weak
    
    def training_step(self, batch):
        if self.mode == 'pretrain':
            ms, mol = batch  
            loss, acc_weak = self.weak_loss(ms, mol)
            self.log('loss (pretrain)', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('R@1 - batch (train)', acc_weak, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            return loss
        elif self.mode == 'finetune':
            ms, candidates = batch  
            loss, acc, acc_weak = self.strong_loss(ms, candidates)
            self.log('loss (finetune)', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('R@1 (train)', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('R@1 - batch (train)', acc_weak, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            return loss
    
    def validation_step(self, batch):
        if self.mode == 'pretrain':
            ms, mol = batch  
            loss, acc_weak = self.weak_loss(ms, mol)
            self.log('R@1 - batch (val)', acc_weak, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            return loss
        elif self.mode == 'finetune':
            ms, candidates = batch  
            loss, acc, acc_weak = self.strong_loss(ms, candidates)
            self.log('R@1 (val)', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            self.log('R@1 - batch (val)', acc_weak, on_step=False, on_epoch=True, prog_bar=True, batch_size=ms.size(0))
            return loss

    def test_step(self, batch):
        ms, candidates = batch 
        ms = F.normalize(self.ms_encoder(ms), p=2, dim=1) 
        candidates_emb, candidates_mask = self.encode_candidates(candidates)

        log = candidate_retrieval_accuracy(ms, candidates_emb, candidates_mask)
        log = {f'{k} (test)': v for k, v in log.items()}
        
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=ms.size(0)
        )

        return log['R@1 (test)']  # Return hard candidate top1 accuracy for checkpointing
