from model_zoo.common import optimizer_with_scheduler, AlignmentMLP, candidate_infonce
from models.retrieval import candidate_retrieval_accuracy
import torch.nn.functional as F
import torch.nn as nn
from transforms.molecules_transforms import MorganFingerprintTransform
from transforms.spectra_transforms import BIN_Transform
import torch
from lightning.pytorch import LightningModule

class EmbCos(LightningModule):
    def __init__(self, config):
        
        LightningModule.__init__(self)

        mol_transform = MorganFingerprintTransform(fp_size=config['fingerprint_size'])
        d_mol = mol_transform.get_dim()
        spectra_transform = BIN_Transform(max_mz=config['max_mz'], bin_width=config['bin_width'])
        d_ms = spectra_transform.get_dim()
        
        self.fc_ms = AlignmentMLP(d_in=d_ms,
                                  d_hidden=config['d_hidden'],
                                  d_shared=config['d_shared'],
                                  n_hidden_layers=config['n_hidden_layers_ms'],
                                  dropout=config['dropout'],
                                  layernorm=config['layernorm'],
                                  residual=False,
                                  orthogonal_init=False) 
        
        self.fc_mol = AlignmentMLP(d_in=d_mol,
                                   d_hidden=config['d_hidden'],
                                   d_shared=config['d_shared'],
                                   n_hidden_layers=config['n_hidden_layers_mol'],
                                   dropout=config['dropout'],
                                   layernorm=config['layernorm'],
                                   residual=False,
                                   orthogonal_init=False)
        
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.n_warmup_steps = config['n_warmup_steps']
        self.n_max_steps = config['n_max_steps']

        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.07)), requires_grad=True)
        
        self.save_hyperparameters(config)
        
    def encode_ms(self, ms):
        ms = F.normalize(self.fc_ms(ms), p=2, dim=-1)               # (B, D')
        return ms
    
    def encode_mol(self, mol):
        mol = F.normalize(self.fc_mol(mol), p=2, dim=-1)
        return mol
        
    def forward(self, ms, mol):
        ms = self.encode_ms(ms)
        mol = self.encode_mol(mol)
        return ms, mol
    
    def configure_optimizers(self):
        return optimizer_with_scheduler(
            parameters=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_warmup_steps=self.n_warmup_steps,
            n_max_steps=self.n_max_steps
        )

        
    def training_step(self, batch):
        ms, candidates, candidates_mask = batch  # ms: (B, D), candidates: (B, K, D), candidates_mask: (B, K)
        
        # Encode all
        ms = self.encode_ms(ms)
        candidates = self.encode_mol(candidates)
        temperature = self.log_epsilon.exp()
        loss, acc = candidate_infonce(ms, candidates, candidates_mask, temperature=temperature)
        
        log = {
            'loss (train)': loss.detach().item(),
            'R@1 (train)': acc
        }
        
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=ms.size(0)
        )

        return loss
    
    def validation_step(self, batch):
        ms, candidates, candidates_mask = batch  # ms: (B, D), candidates: (B, K, D), candidates_mask: (B, K)
        
        # Project to shared space
        ms = self.encode_ms(ms)
        candidates = self.encode_mol(candidates)
        
        # Do it once with the correct "hard" candidates
        log = candidate_retrieval_accuracy(ms, candidates, candidates_mask)
        log = {f'{k} (val)': v for k, v in log.items()}
        
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=ms.size(0)
        )
        
        return log['R@1 (val)']  # Return hard candidate top1 accuracy for checkpointing
    
    def test_step(self, batch):
        ms, candidates, candidates_mask = batch  # ms: (B, D), candidates: (B, K, D), candidates_mask: (B, K)

        # Project to shared space
        ms = self.encode_ms(ms)
        candidates = self.encode_mol(candidates)
        
        # Do it once with the correct "hard" candidates
        log = candidate_retrieval_accuracy(ms, candidates, candidates_mask)
        log = {f'{k} (test)': v for k, v in log.items()}
        
        self.log_dict(
            log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=ms.size(0)
        )
        
        return log['R@1 (test)']  # Return hard candidate top1 accuracy for checkpointing
