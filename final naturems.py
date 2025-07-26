
#apply xg boost. parquet file system.more effective



#smile output corrected.
#vae based encoder

import torch
from torch.cuda import amp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from datasets import load_dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdFMCS, EnumerateStereoisomers
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from tqdm import tqdm
import math
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.cuda import amp
from torch.cuda.amp import GradScaler, autocast
import optuna
from nltk.translate.bleu_score import sentence_bleu
from Levenshtein import distance
import matplotlib inline

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define token variables early
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
MASK_TOKEN = "[MASK]"

# Load and preprocess dataset
dataset = load_dataset('roman-bushuiev/MassSpecGym', split='train')
df = pd.DataFrame(dataset)

# Inspect dataset
print("Dataset Columns:", df.columns.tolist())
print("\nFirst few rows of the dataset:")
print(df[['identifier', 'mzs', 'intensities', 'smiles', 'adduct', 'precursor_mz']].head())
print("\nUnique adduct values:", df['adduct'].unique())

# Data augmentation: SMILES enumeration and spectral noise
def augment_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            stereoisomers = EnumerateStereoisomers.EnumerateStereoisomers(mol)
            return [Chem.MolToSmiles(m, canonical=True) for m in stereoisomers]
        return [smiles]
    except:
        return [smiles]

def bin_spectrum_to_graph(mzs, intensities, ion_mode, precursor_mz, adduct, n_bins=1000, max_mz=1000, noise_level=0.05):
    spectrum = np.zeros(n_bins)
    for mz, intensity in zip(mzs, intensities):
        try:
            mz = float(mz)
            intensity = float(intensity)
            if mz < max_mz:
                bin_idx = int((mz / max_mz) * n_bins)
                spectrum[bin_idx] += intensity
        except (ValueError, TypeError):
            continue
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()
    # Add synthetic noise
    spectrum += np.random.normal(0, noise_level, spectrum.shape).clip(0, 1)
    # Graph data for GNN
    x = torch.tensor(spectrum, dtype=torch.float).unsqueeze(-1)
    edge_index = []
    for i in range(n_bins-1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    ion_mode = torch.tensor([ion_mode], dtype=torch.float)
    precursor_mz = torch.tensor([precursor_mz], dtype=torch.float)
    adduct_idx = adduct_to_idx.get(adduct, 0)
    return spectrum, Data(x=x, edge_index=edge_index, ion_mode=ion_mode, precursor_mz=precursor_mz, adduct_idx=adduct_idx)

# Canonicalize SMILES and augment
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None

df['smiles'] = df['smiles'].apply(canonicalize_smiles)
df = df.dropna(subset=['smiles'])
df['smiles_list'] = df['smiles'].apply(augment_smiles)
df = df.explode('smiles_list').dropna(subset=['smiles_list']).rename(columns={'smiles_list': 'smiles'})

# Preprocess ion mode, precursor m/z, and adducts
df['ion_mode'] = df['adduct'].apply(lambda x: 0 if '+' in str(x) else 1 if '-' in str(x) else 0).fillna(0)
df['precursor_bin'] = pd.qcut(df['precursor_mz'], q=100, labels=False, duplicates='drop')
adduct_types = df['adduct'].unique()
adduct_to_idx = {adduct: i for i, adduct in enumerate(adduct_types)}
df['adduct_idx'] = df['adduct'].map(adduct_to_idx)

df[['binned', 'graph_data']] = df.apply(
    lambda row: pd.Series(bin_spectrum_to_graph(row['mzs'], row['intensities'], row['ion_mode'], row['precursor_mz'], row['adduct'])),
    axis=1
)

# SMILES Tokenization
all_smiles = df['smiles'].tolist()
unique_chars = set(''.join(all_smiles)) | {MASK_TOKEN}
valid_atoms = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'}
tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MASK_TOKEN] + sorted(unique_chars - {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MASK_TOKEN})
token_to_idx = {tok: i for i, tok in enumerate(tokens) if tok in valid_atoms or tok in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MASK_TOKEN, '(', ')', '=', '#', '@', '[', ']', '/', '\\', '.', ':'}}
idx_to_token = {i: tok for tok, i in token_to_idx.items()}
vocab_size = len(token_to_idx)
PRETRAIN_MAX_LEN = 100
SUPERVISED_MAX_LEN = max(len(s) + 2 for s in all_smiles)
print(f"Vocabulary size: {vocab_size}, Supervised MAX_LEN: {SUPERVISED_MAX_LEN}, Pretrain MAX_LEN: {PRETRAIN_MAX_LEN}")

def encode_smiles(smiles, max_len=PRETRAIN_MAX_LEN):
    tokens = [SOS_TOKEN] + [c for c in smiles[:max_len-2] if c in token_to_idx] + [EOS_TOKEN]
    token_ids = [token_to_idx.get(tok, token_to_idx[PAD_TOKEN]) for tok in tokens]
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids += [token_to_idx[PAD_TOKEN]] * (max_len - len(token_ids))
    return token_ids

# Precompute Morgan fingerprints for SMILES
all_smiles = df['smiles'].unique().tolist()
all_fingerprints = {}
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
for smiles in all_smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        all_fingerprints[smiles] = morgan_gen.GetFingerprint(mol)

# Dataset class
class MSMSDataset(Dataset):
    def __init__(self, dataframe, max_len=PRETRAIN_MAX_LEN, is_ssl=False):
        self.spectra = np.stack(dataframe['binned'].values)
        self.graph_data = dataframe['graph_data'].values
        self.ion_modes = dataframe['ion_mode'].values
        self.precursor_bins = dataframe['precursor_bin'].values
        self.adduct_indices = dataframe['adduct_idx'].values
        self.raw_smiles = dataframe['smiles'].values
        self.is_ssl = is_ssl
        if is_ssl:
            self.smiles = []
            self.masked_smiles = []
            for s in self.raw_smiles:
                masked_s, orig_s = self.mask_smiles(s)
                self.smiles.append(encode_smiles(orig_s, max_len))
                self.masked_smiles.append(encode_smiles(masked_s, max_len))
        else:
            self.smiles = [encode_smiles(s, max_len=SUPERVISED_MAX_LEN) for s in self.raw_smiles]

    def mask_smiles(self, smiles, mask_ratio=0.10):
        chars = list(smiles)[:PRETRAIN_MAX_LEN-2]
        masked_chars = chars.copy()
        n_mask = int(mask_ratio * len(chars))
        mask_indices = np.random.choice(len(chars), n_mask, replace=False)
        for idx in mask_indices:
            masked_chars[idx] = MASK_TOKEN
        return ''.join(masked_chars), ''.join(chars)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        if self.is_ssl:
            return (
                torch.tensor(self.spectra[idx], dtype=torch.float),
                self.graph_data[idx],
                torch.tensor(self.smiles[idx], dtype=torch.long),
                torch.tensor(self.masked_smiles[idx], dtype=torch.long),
                torch.tensor(self.ion_modes[idx], dtype=torch.long),
                torch.tensor(self.precursor_bins[idx], dtype=torch.long),
                torch.tensor(self.adduct_indices[idx], dtype=torch.long),
                self.raw_smiles[idx]
            )
        return (
            torch.tensor(self.spectra[idx], dtype=torch.float),
            self.graph_data[idx],
            torch.tensor(self.smiles[idx], dtype=torch.long),
            torch.tensor(self.ion_modes[idx], dtype=torch.long),
            torch.tensor(self.precursor_bins[idx], dtype=torch.long),
            torch.tensor(self.adduct_indices[idx], dtype=torch.long),
            self.raw_smiles[idx]
        )

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Transformer Encoder
class SpectrumTransformerEncoder(nn.Module):
    def __init__(self, input_dim=1000, d_model=768, nhead=12, num_layers=8, dim_feedforward=2048, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.metadata_emb = nn.Linear(2 + 32, 64)  # Include adduct embedding
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model + 64, d_model // 2)
        self.adduct_emb = nn.Embedding(len(adduct_types), 32)

    def forward(self, src, ion_mode_idx, precursor_idx, adduct_idx):
        src = self.input_proj(src).unsqueeze(1)
        adduct_embed = self.adduct_emb(adduct_idx)
        metadata = self.metadata_emb(torch.cat([ion_mode_idx.unsqueeze(-1).float(), precursor_idx.unsqueeze(-1).float(), adduct_embed], dim=-1))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src).squeeze(1)
        output = self.norm(output)
        output = torch.cat([output, metadata], dim=-1)
        output = self.fc(output)
        return output, self.transformer_encoder.layers[-1].self_attn(src, src, src)[1]

# GNN Encoder
class SpectrumGNNEncoder(MessagePassing):
    def __init__(self, d_model=768, hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__(aggr='mean')
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_proj = nn.Linear(1, hidden_dim)
        self.message_nets = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.update_nets = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.metadata_emb = nn.Linear(2 + 32, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.substructure_head = nn.Linear(hidden_dim, 10)  # Predict 10 common substructures
        self.adduct_emb = nn.Embedding(len(adduct_types), 32)

    def forward(self, graph_data, ion_mode_idx, precursor_idx, adduct_idx):
        batch = Batch.from_data_list(graph_data).to(device)
        x, edge_index = batch.x, batch.edge_index
        ion_mode = batch.ion_mode
        precursor_mz = batch.precursor_mz
        adduct_embed = self.adduct_emb(adduct_idx)

        x = self.input_proj(x)
        metadata = self.metadata_emb(torch.cat([ion_mode.unsqueeze(-1), precursor_mz.unsqueeze(-1), adduct_embed], dim=-1))

        edge_weights = []
        for i in range(self.num_layers):
            self._propagate_layer = i
            x_before = x.clone()
            x = self.propagate(edge_index, x=x)
            x = self.update_nets[i](torch.cat([x, metadata], dim=-1))
            x = self.norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            edge_weights.append((x - x_before).norm(dim=-1))

        x = global_mean_pool(x, batch.batch)
        substructure_pred = self.substructure_head(x)
        x = self.output_layer(x)
        return x, substructure_pred, edge_weights

    def message(self, x_i, x_j):
        return F.relu(self.message_nets[self._propagate_layer](torch.cat([x_i, x_j], dim=-1)))

# Novel Decoder with Chemical Constraints and Substructure Guidance
class SmilesTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=8, dim_feedforward=2048, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.valence_rules = {
            'C': 4, 'N': 3, 'O': 2, 'S': 2, 'P': 3, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'H': 1
        }
        self.substructure_condition = nn.Linear(10, d_model)  # Condition on GNN substructure predictions

    def compute_valence(self, smiles_tokens, batch_size):
        valence_counts = torch.zeros(batch_size, len(self.valence_rules)).to(smiles_tokens.device)
        atom_indices = {tok: i for i, tok in enumerate(self.valence_rules.keys())}
        for t in range(smiles_tokens.size(1)):
            for tok, idx in atom_indices.items():
                mask = smiles_tokens[:, t] == token_to_idx[tok]
                valence_counts[mask, idx] += self.valence_rules[tok]
        return valence_counts

    def forward(self, tgt, memory, substructure_pred, tgt_mask=None, memory_key_padding_mask=None):
        embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        substructure_emb = self.substructure_condition(substructure_pred).unsqueeze(1)
        embedded = embedded + substructure_emb
        output = self.transformer_decoder(embedded, memory, tgt_mask, memory_key_padding_mask)
        output = self.norm(output)
        logits = self.output_layer(output)
        valence_counts = self.compute_valence(tgt, tgt.size(0))
        valence_penalty = torch.relu(valence_counts - torch.tensor(list(self.valence_rules.values()), device=tgt.device)).sum(dim=-1)
        return logits, valence_penalty

# Full Model with Combined Encoders and Fingerprint Prediction
class MSMS2SmilesHybrid(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=8, dim_feedforward=2048, dropout=0.2, fp_size=2048):
        super().__init__()
        self.transformer_encoder = SpectrumTransformerEncoder(input_dim=1000, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.gnn_encoder = SpectrumGNNEncoder(d_model=d_model, hidden_dim=256, num_layers=3, dropout=dropout)
        self.decoder = SmilesTransformerDecoder(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.combine_layer = nn.Linear(d_model, d_model)
        self.fp_head = nn.Linear(d_model, fp_size)
        self.fp_size = fp_size
        self.log_sigma_smiles = nn.Parameter(torch.zeros(1))
        self.log_sigma_fp = nn.Parameter(torch.zeros(1))

    def generate_square_subsequent_mask(self, tgt_len):
        mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask

    def forward(self, spectrum, graph_data, tgt, ion_mode_idx, precursor_idx, adduct_idx, tgt_mask=None, memory_key_padding_mask=None):
        trans_output, attn_weights = self.transformer_encoder(spectrum, ion_mode_idx, precursor_idx, adduct_idx)
        gnn_output, substructure_pred, edge_weights = self.gnn_encoder(graph_data, ion_mode_idx, precursor_idx, adduct_idx)
        memory = self.combine_layer(torch.cat([trans_output, gnn_output], dim=-1)).unsqueeze(1)
        smiles_output, valence_penalty = self.decoder(tgt, memory, substructure_pred, tgt_mask, memory_key_padding_mask)
        fp_output = self.fp_head(memory.squeeze(1))
        return smiles_output, fp_output, valence_penalty, attn_weights, edge_weights, substructure_pred

# SSL Pretraining with Mixed Precision
def ssl_pretrain(model, dataloader, epochs=3, lr=1e-4):
    model.train()
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_idx[PAD_TOKEN])
    for epoch in range(epochs):
        total_loss = 0
        for spectra, graph_data, smiles_tokens, masked_tokens, ion_modes, precursor_bins, adduct_indices, _ in tqdm(dataloader, desc=f"SSL Epoch {epoch+1}/{epochs}"):
            spectra = spectra.to(device)
            ion_modes = ion_modes.to(device)
            precursor_bins = precursor_bins.to(device)
            adduct_indices = adduct_indices.to(device)
            smiles_tokens = smiles_tokens.to(device)
            masked_tokens = masked_tokens.to(device)
            tgt_input = masked_tokens[:, :-1]
            tgt_output = smiles_tokens[:, 1:]
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            optimizer.zero_grad()
            with autocast():
                smiles_output, _, valence_penalty, _, _, _ = model(spectra, graph_data, tgt_input, ion_modes, precursor_bins, adduct_indices, tgt_mask)
                loss = criterion(smiles_output.reshape(-1, vocab_size), tgt_output.reshape(-1)) + 0.1 * valence_penalty.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"SSL Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, f'ssl_checkpoint_epoch_{epoch+1}.pt')
        print(f"Saved SSL checkpoint: ssl_checkpoint_epoch_{epoch+1}.pt")

# Supervised Training with Dynamic Loss Weighting
def supervised_train(model, train_loader, val_loader, epochs=30, lr=1e-4, patience=5):
    model.train()
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    smiles_criterion = nn.CrossEntropyLoss(ignore_index=token_to_idx[PAD_TOKEN])
    fp_criterion = nn.BCEWithLogitsLoss()
    mw_criterion = nn.MSELoss()
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for spectra, graph_data, smiles_tokens, ion_modes, precursor_bins, adduct_indices, raw_smiles in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            spectra = spectra.to(device)
            ion_modes = ion_modes.to(device)
            precursor_bins = precursor_bins.to(device)
            adduct_indices = adduct_indices.to(device)
            smiles_tokens = smiles_tokens.to(device)
            tgt_input = smiles_tokens[:, :-1]
            tgt_output = smiles_tokens[:, 1:]
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            optimizer.zero_grad()
            with autocast():
                smiles_output, fp_output, valence_penalty, _, _, _ = model(spectra, graph_data, tgt_input, ion_modes, precursor_bins, adduct_indices, tgt_mask)
                smiles_loss = smiles_criterion(smiles_output.reshape(-1, vocab_size), tgt_output.reshape(-1))
                fp_loss = 0
                mw_loss = 0
                valid_count = 0
                for smiles, fp in zip(raw_smiles, fp_output):
                    mol = Chem.MolFromSmiles(smiles, sanitize=True)
                    if mol:
                        true_fp = morgan_gen.GetFingerprint(mol)
                        fp_loss += fp_criterion(fp, torch.tensor([int(b) for b in true_fp.ToBitString()], dtype=torch.float, device=device))
                        mw_loss += mw_criterion(torch.tensor(Descriptors.MolWt(mol), dtype=torch.float, device=device), torch.tensor(500.0, dtype=torch.float, device=device))
                        valid_count += 1
                fp_loss = fp_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=device)
                mw_loss = mw_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=device)
                sigma_smiles = torch.exp(model.log_sigma_smiles)
                sigma_fp = torch.exp(model.log_sigma_fp)
                loss = (smiles_loss / (2 * sigma_smiles**2) + model.log_sigma_smiles) + (fp_loss / (2 * sigma_fp**2) + model.log_sigma_fp) + 0.1 * valence_penalty.mean() + 0.1 * mw_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for spectra, graph_data, smiles_tokens, ion_modes, precursor_bins, adduct_indices, raw_smiles in val_loader:
                spectra = spectra.to(device)
                ion_modes = ion_modes.to(device)
                precursor_bins = precursor_bins.to(device)
                adduct_indices = adduct_indices.to(device)
                smiles_tokens = smiles_tokens.to(device)
                tgt_input = smiles_tokens[:, :-1]
                tgt_output = smiles_tokens[:, 1:]
                tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                with autocast():
                    smiles_output, fp_output, valence_penalty, _, _, _ = model(spectra, graph_data, tgt_input, ion_modes, precursor_bins, adduct_indices, tgt_mask)
                    smiles_loss = smiles_criterion(smiles_output.reshape(-1, vocab_size), tgt_output.reshape(-1))
                    fp_loss = 0
                    mw_loss = 0
                    valid_count = 0
                    for smiles, fp in zip(raw_smiles, fp_output):
                        mol = Chem.MolFromSmiles(smiles, sanitize=True)
                        if mol:
                            true_fp = morgan_gen.GetFingerprint(mol)
                            fp_loss += fp_criterion(fp, torch.tensor([int(b) for b in true_fp.ToBitString()], dtype=torch.float, device=device))
                            mw_loss += mw_criterion(torch.tensor(Descriptors.MolWt(mol), dtype=torch.float, device=device), torch.tensor(500.0, dtype=torch.float, device=device))
                            valid_count += 1
                    fp_loss = fp_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=device)
                    mw_loss = mw_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=device)
                    sigma_smiles = torch.exp(model.log_sigma_smiles)
                    sigma_fp = torch.exp(model.log_sigma_fp)
                    loss = (smiles_loss / (2 * sigma_smiles**2) + model.log_sigma_smiles) + (fp_loss / (2 * sigma_fp**2) + model.log_sigma_fp) + 0.1 * valence_penalty.mean() + 0.1 * mw_loss
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'token_to_idx': token_to_idx,
                'idx_to_token': idx_to_token
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'token_to_idx': token_to_idx,
                'idx_to_token': idx_to_token
            }, 'best_msms_hybrid.pt')
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_val_loss

# SMILES Syntax Validator
def is_valid_smiles_syntax(smiles):
    stack = []
    for c in smiles:
        if c in '([':
            stack.append(c)
        elif c == ')':
            if not stack or stack[-1] != '(':
                return False
            stack.pop()
        elif c == ']':
            if not stack or stack[-1] != '[':
                return False
            stack.pop()
    if stack:
        return False
    i = 0
    while i < len(smiles):
        if smiles[i] == '[':
            j = smiles.find(']', i)
            if j == -1:
                return False
            atom = smiles[i+1:j]
            if not any(a in atom for a in valid_atoms):
                return False
            i = j + 1
        else:
            if smiles[i] in valid_atoms or smiles[i] in '()=#/\\.:':
                i += 1
            else:
                return False
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return mol is not None
    except:
        return False

# RDKit-based Molecular Property Filter
def is_plausible_molecule(smiles, true_mol, max_mw=1500, min_logp=-7, max_logp=7):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if not mol or not is_valid_smiles_syntax(smiles):
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    true_mw = Descriptors.MolWt(true_mol) if true_mol else 500
    return mw <= max_mw and min_logp <= logp <= max_logp and abs(mw - true_mw) < 300

# Alternative Evaluation Metrics
def dice_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 and mol2:
        fp1 = morgan_gen.GetFingerprint(mol1)
        fp2 = morgan_gen.GetFingerprint(mol2)
        return DataStructs.DiceSimilarity(fp1, fp2)
    return 0.0

def mcs_similarity(true_smiles, pred_smiles):
    mol1 = Chem.MolFromSmiles(true_smiles)
    mol2 = Chem.MolFromSmiles(pred_smiles)
    if mol1 and mol2:
        mcs = rdFMCS.FindMCS([mol1, mol2], timeout=30)
        return mcs.numAtoms / max(mol1.GetNumAtoms(), mol2.GetNumAtoms())
    return 0.0

def mw_difference(true_smiles, pred_smiles):
    mol1 = Chem.MolFromSmiles(true_smiles)
    mol2 = Chem.MolFromSmiles(pred_smiles)
    if mol1 and mol2:
        return abs(Descriptors.MolWt(mol1) - Descriptors.MolWt(mol2))
    return float('inf')

def logp_difference(true_smiles, pred_smiles):
    mol1 = Chem.MolFromSmiles(true_smiles)
    mol2 = Chem.MolFromSmiles(pred_smiles)
    if mol1 and mol2:
        return abs(Descriptors.MolLogP(mol1) - Descriptors.MolLogP(mol2))
    return float('inf')

def substructure_match(true_smiles, pred_smiles, substructures=['C=O', 'C=C', 'c1ccccc1']):
    mol1 = Chem.MolFromSmiles(true_smiles)
    mol2 = Chem.MolFromSmiles(pred_smiles)
    if not mol1 or not mol2:
        return 0
    matches = 0
    for smarts in substructures:
        pattern = Chem.MolFromSmarts(smarts)
        if mol1.HasSubstructMatch(pattern) and mol2.HasSubstructMatch(pattern):
            matches += 1
    return matches / len(substructures)

def validity_rate(pred_smiles_list):
    valid = sum(1 for smiles in pred_smiles_list if Chem.MolFromSmiles(smiles, sanitize=True) is not None)
    return valid / len(pred_smiles_list) * 100

def tanimoto_similarity(smiles1, smiles2, precomputed_fps=None):
    mol1 = Chem.MolFromSmiles(smiles1, sanitize=True)
    if not mol1:
        return 0.0
    fp1 = morgan_gen.GetFingerprint(mol1)
    if precomputed_fps and smiles2 in precomputed_fps:
        fp2 = precomputed_fps[smiles2]
    else:
        mol2 = Chem.MolFromSmiles(smiles2, sanitize=True)
        if not mol2:
            return 0.0
        fp2 = morgan_gen.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def prediction_diversity(pred_smiles_list):
    if len(pred_smiles_list) < 2:
        return 0.0
    total_tanimoto = 0
    count = 0
    for i in range(len(pred_smiles_list)):
        for j in range(i+1, len(pred_smiles_list)):
            total_tanimoto += tanimoto_similarity(pred_smiles_list[i], pred_smiles_list[j])
            count += 1
    return 1 - (total_tanimoto / count) if count > 0 else 0.0

# Hybrid VAE Encoder (keeps the working hybrid system)
class HybridVAEEncoder(nn.Module):
    def __init__(self, d_model=768, latent_dim=256):
        super().__init__()
        self.transformer_encoder = SpectrumTransformerEncoder(input_dim=1000, d_model=d_model, nhead=12, num_layers=8, dim_feedforward=2048, dropout=0.2)
        self.gnn_encoder = SpectrumGNNEncoder(d_model=d_model, hidden_dim=256, num_layers=3, dropout=0.2)
        self.combine_layer = nn.Linear(d_model, d_model)
        self.mu_layer = nn.Linear(d_model, latent_dim)
        self.logvar_layer = nn.Linear(d_model, latent_dim)
        
    def forward(self, spectrum, graph_data, ion_mode_idx, precursor_idx, adduct_idx):
        trans_output, attn_weights = self.transformer_encoder(spectrum, ion_mode_idx, precursor_idx, adduct_idx)
        gnn_output, substructure_pred, edge_weights = self.gnn_encoder(graph_data, ion_mode_idx, precursor_idx, adduct_idx)
        combined = self.combine_layer(torch.cat([trans_output, gnn_output], dim=-1))
        mu = self.mu_layer(combined)
        logvar = self.logvar_layer(combined)
        return mu, logvar, substructure_pred, attn_weights, edge_weights

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class VAEDecoder(nn.Module):
    def __init__(self, vocab_size, latent_dim=256, d_model=768, nhead=12, num_layers=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, latent_z, tgt_mask=None):
        # Project latent to memory
        memory = self.latent_proj(latent_z).unsqueeze(1)  # [batch, 1, d_model]
        
        # Embed target tokens
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Transpose for transformer
        tgt_emb = tgt_emb.transpose(0, 1)  # [seq_len, batch, d_model]
        memory = memory.transpose(0, 1)    # [1, batch, d_model]
        
        # Decode
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)  # [batch, seq_len, d_model]
        
        return self.output_proj(output)

class MSMS2SmilesVAE(nn.Module):
    def __init__(self, vocab_size, d_model=768, latent_dim=256, nhead=12, num_layers=6):
        super().__init__()
        self.encoder = HybridVAEEncoder(d_model, latent_dim)
        self.decoder = VAEDecoder(vocab_size, latent_dim, d_model, nhead, num_layers)
        self.vocab_size = vocab_size
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, spectrum, graph_data, tgt, ion_mode_idx, precursor_idx, adduct_idx, tgt_mask=None):
        # Encode with hybrid system to latent space
        mu, logvar, substructure_pred, attn_weights, edge_weights = self.encoder(spectrum, graph_data, ion_mode_idx, precursor_idx, adduct_idx)
        z = self.reparameterize(mu, logvar)
        
        # Decode to SMILES
        output = self.decoder(tgt, z, tgt_mask)
        
        return output, mu, logvar, substructure_pred, attn_weights, edge_weights
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    # Reconstruction loss (cross-entropy)
    CE = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), ignore_index=token_to_idx[PAD_TOKEN])
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= mu.size(0)  # Average over batch
    
    return CE + beta * KLD, CE, KLD

def supervised_train(model, train_loader, val_loader, epochs=30, lr=1e-4, patience=5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            spectrum, graph_data, smiles, ion_mode, precursor_bin, adduct_idx, _ = batch
            spectrum = spectrum.to(device)
            smiles = smiles.to(device)
            ion_mode = ion_mode.to(device)
            precursor_bin = precursor_bin.to(device)
            adduct_idx = adduct_idx.to(device)
            
            # Prepare target (input and output for teacher forcing)
            tgt_input = smiles[:, :-1]
            tgt_output = smiles[:, 1:]
            
            # Create mask
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            optimizer.zero_grad()
            output, mu, logvar, _, _, _ = model(spectrum, graph_data, tgt_input, ion_mode, precursor_bin, adduct_idx, tgt_mask)
            
            loss, ce_loss, kl_loss = vae_loss_function(output, tgt_output, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                spectrum, graph_data, smiles, ion_mode, precursor_bin, adduct_idx, _ = batch
                spectrum = spectrum.to(device)
                smiles = smiles.to(device)
                ion_mode = ion_mode.to(device)
                precursor_bin = precursor_bin.to(device)
                adduct_idx = adduct_idx.to(device)
                
                tgt_input = smiles[:, :-1]
                tgt_output = smiles[:, 1:]
                tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                
                output, mu, logvar, _, _, _ = model(spectrum, graph_data, tgt_input, ion_mode, precursor_bin, adduct_idx, tgt_mask)
                loss, _, _ = vae_loss_function(output, tgt_output, mu, logvar)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return best_val_loss

# Beam Search with SMILES Constraints and Canonicalization
def beam_search(model, spectrum, graph_data, ion_mode_idx, precursor_idx, adduct_idx, true_smiles, beam_width=10, max_len=150, nucleus_p=0.9, device='cpu'):
    model.eval()
    true_mol = Chem.MolFromSmiles(true_smiles) if true_smiles else None
    with torch.no_grad():
        spectrum = spectrum.unsqueeze(0).to(device)
        graph_data = Batch.from_data_list([graph_data]).to(device)
        ion_mode_idx = torch.tensor([ion_mode_idx], dtype=torch.long).to(device)
        precursor_idx = torch.tensor([precursor_idx], dtype=torch.long).to(device)
        adduct_idx = torch.tensor([adduct_idx], dtype=torch.long).to(device)
        memory = model.transformer_encoder(spectrum, ion_mode_idx, precursor_idx, adduct_idx)[0]
        gnn_output, substructure_pred, _ = model.gnn_encoder(graph_data, ion_mode_idx, precursor_idx, adduct_idx)
        memory = model.combine_layer(torch.cat([memory, gnn_output], dim=-1)).unsqueeze(1)
        sequences = [([token_to_idx[SOS_TOKEN]], 0.0)]

        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == token_to_idx[EOS_TOKEN]:
                    all_candidates.append((seq, score))
                    continue
                partial_smiles = ''.join([idx_to_token.get(idx, '') for idx in seq[1:]])
                if not is_valid_smiles_syntax(partial_smiles):
                    continue
                tgt_input = torch.tensor([seq], dtype=torch.long).to(device)
                tgt_mask = model.generate_square_subsequent_mask(len(seq)).to(device)
                outputs, valence_penalty = model.decoder(tgt_input, memory, substructure_pred, tgt_mask)
                log_probs = F.log_softmax(outputs[0, -1], dim=-1).cpu().numpy() - 0.1 * valence_penalty.cpu().numpy()
                sorted_probs = np.sort(np.exp(log_probs))[::-1]
                cumulative_probs = np.cumsum(sorted_probs)
                cutoff_idx = np.searchsorted(cumulative_probs, nucleus_p)
                top_tokens = np.argsort(log_probs)[-cutoff_idx:] if cutoff_idx > 0 else np.argsort(log_probs)[-1:]
                top_probs = np.exp(log_probs[top_tokens]) / np.sum(np.exp(log_probs[top_tokens]))
                for tok in np.random.choice(top_tokens, size=min(beam_width, len(top_tokens)), p=top_probs):
                    new_smiles = partial_smiles + idx_to_token.get(int(tok), '')
                    if is_valid_smiles_syntax(new_smiles):
                        diversity_penalty = 0.2 * sum(1 for s, _ in sequences if tok in s[1:-1])
                        all_candidates.append((seq + [int(tok)], score + log_probs[tok] - diversity_penalty))
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq[-1] == token_to_idx[EOS_TOKEN] for seq, _ in sequences):
                break

        results = []
        for seq, score in sequences:
            smiles = ''.join([idx_to_token.get(idx, '') for idx in seq[1:-1]])
            try:
                mol = Chem.MolFromSmiles(smiles, sanitize=True)
                if mol and is_plausible_molecule(smiles, true_mol):
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    confidence = np.exp(score / len(seq))
                    results.append((smiles, confidence))
            except:
                continue
        return results if results else [("Invalid SMILES", 0.0)]

# Visualization Functions
def plot_attention_weights(attn_weights, title="Transformer Attention Weights"):
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_weights.squeeze().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.show()

def plot_gnn_edge_weights(edge_weights, edge_index, title="GNN Edge Importance"):
    edge_scores = edge_weights[-1].cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.hist(edge_scores, bins=50)
    plt.title(title)
    plt.xlabel("Edge Weight Magnitude")
    plt.ylabel("Frequency")
    plt.show()

# Error Analysis
def error_analysis(pred_smiles_list, true_smiles_list, adducts, precomputed_fps):
    errors = {'small': 0, 'large': 0, 'aromatic': 0, 'aliphatic': 0}
    adduct_errors = {adduct: [] for adduct in adduct_types}
    for pred_smiles, true_smiles, adduct in zip(pred_smiles_list, true_smiles_list, adducts):
        tanimoto = tanimoto_similarity(pred_smiles, true_smiles, precomputed_fps)
        if tanimoto < 0.3:
            mol = Chem.MolFromSmiles(true_smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
                errors['small' if mw < 300 else 'large'] += 1
                errors['aromatic' if is_aromatic else 'aliphatic'] += 1
                adduct_errors[adduct].append(tanimoto)
    print("Error Analysis:")
    print(f"Small molecules (<300 Da) errors: {errors['small']}")
    print(f"Large molecules (≥300 Da) errors: {errors['large']}")
    print(f"Aromatic molecule errors: {errors['aromatic']}")
    print(f"Aliphatic molecule errors: {errors['aliphatic']}")
    for adduct, scores in adduct_errors.items():
        if scores:
            print(f"Adduct {adduct} - Avg Tanimoto: {np.mean(scores):.4f}, Count: {len(scores)}")

# Hyperparameter Tuning with Optuna
def objective(trial, train_data, val_data):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    train_dataset = MSMSDataset(train_data, max_len=SUPERVISED_MAX_LEN, is_ssl=False)
    val_dataset = MSMSDataset(val_data, max_len=SUPERVISED_MAX_LEN, is_ssl=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)
    model = MSMS2SmilesVAE(vocab_size=vocab_size, input_dim=1000, d_model=768, latent_dim=256, nhead=12, num_layers=6).to(device)
    return supervised_train(model, train_loader, val_loader, epochs=10, lr=lr)

# Cross-Validation and Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\nFold {fold+1}/5")
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    ssl_data = train_data.sample(frac=0.3, random_state=42)

    train_dataset = MSMSDataset(train_data, max_len=SUPERVISED_MAX_LEN, is_ssl=False)
    val_dataset = MSMSDataset(val_data, max_len=SUPERVISED_MAX_LEN, is_ssl=False)
    ssl_dataset = MSMSDataset(ssl_data, max_len=PRETRAIN_MAX_LEN, is_ssl=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)
    ssl_loader = DataLoader(ssl_dataset, batch_size=128, shuffle=True, num_workers=2)

    # Hyperparameter tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_data, val_data), n_trials=10)
    best_lr = study.best_params['lr']
    print(f"Best learning rate for fold {fold+1}: {best_lr:.6f}")

    # Initialize and train model
    model = MSMS2SmilesVAE(vocab_size=vocab_size, input_dim=1000, d_model=768, latent_dim=256, nhead=12, num_layers=6).to(device)
    # Skip SSL pretraining for VAE (can be added later if needed)
    print(f"Starting supervised training for fold {fold+1}...")
    best_val_loss = supervised_train(model, train_loader, val_loader, epochs=30, lr=best_lr, patience=5)
    fold_results.append(best_val_loss)
    torch.save({
        'model_state_dict': model.state_dict(),
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token
    }, f'best_msms_hybrid_fold_{fold+1}.pt')

print(f"Cross-validation results: {fold_results}")
print(f"Average validation loss: {np.mean(fold_results):.4f}")

# Inference and Evaluation
num_samples = 5
pred_smiles_list = []
true_smiles_list = []
adducts_list = []
for sample_idx in range(num_samples):
    sample_spectrum = val_dataset[sample_idx][0]
    sample_graph = val_dataset[sample_idx][1]
    sample_ion_mode = val_dataset[sample_idx][3]
    sample_precursor_bin = val_dataset[sample_idx][4]
    sample_adduct_idx = val_dataset[sample_idx][5]
    true_smiles = val_dataset[sample_idx][6]

    predicted_results = beam_search(model, sample_spectrum, sample_graph, sample_ion_mode, sample_precursor_bin, sample_adduct_idx, true_smiles, beam_width=10, max_len=SUPERVISED_MAX_LEN, device=device)
    pred_smiles_list.extend([smiles for smiles, _ in predicted_results])
    true_smiles_list.extend([true_smiles] * len(predicted_results))
    adducts_list.extend([df.iloc[val_idx[sample_idx]]['adduct']] * len(predicted_results))

    print(f"\nSample {sample_idx} - True SMILES: {true_smiles}")
    print("Top Predicted SMILES:")
    metrics = {'tanimoto': [], 'dice': [], 'mcs': [], 'mw_diff': [], 'logp_diff': [], 'substructure': []}
    for smiles, confidence in predicted_results[:3]:
        metrics['tanimoto'].append(tanimoto_similarity(smiles, true_smiles, all_fingerprints))
        metrics['dice'].append(dice_similarity(smiles, true_smiles))
        metrics['mcs'].append(mcs_similarity(smiles, true_smiles))
        metrics['mw_diff'].append(mw_difference(smiles, true_smiles))
        metrics['logp_diff'].append(logp_difference(smiles, true_smiles))
        metrics['substructure'].append(substructure_match(smiles, true_smiles))
        print(f"SMILES: {smiles}, Confidence: {confidence:.4f}, Tanimoto: {metrics['tanimoto'][-1]:.4f}, Dice: {metrics['dice'][-1]:.4f}, MCS: {metrics['mcs'][-1]:.4f}")
        if len(smiles) > 100 and smiles.count('C') > len(smiles) * 0.8:
            print("Warning: Predicted SMILES is a long carbon chain, indicating potential model underfitting.")
        if smiles != "Invalid SMILES":
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol:
                print(f"Molecular Weight: {Descriptors.MolWt(mol):.2f}, LogP: {Descriptors.MolLogP(mol):.2f}")

    # Visualize molecules
    if predicted_results[0][0] != "Invalid SMILES":
        pred_mol = Chem.MolFromSmiles(predicted_results[0][0], sanitize=True)
        true_mol = Chem.MolFromSmiles(true_smiles, sanitize=True)
        if pred_mol and true_mol:
            img = Draw.MolsToGridImage([true_mol, pred_mol], molsPerRow=2, subImgSize=(300, 300), legends=['True', 'Predicted'])
            img_array = np.array(img.convert('RGB'))
            plt.figure(figsize=(10, 5))
            plt.imshow(img_array)
            plt.axis('off')
            plt.title(f"Sample {sample_idx} - Tanimoto: {metrics['tanimoto'][0]:.4f}")
            plt.show()

    # Visualize attention and GNN weights for first sample
    if sample_idx == 0:
        with torch.no_grad():
            spectrum = sample_spectrum.unsqueeze(0).to(device)
            graph_data = Batch.from_data_list([sample_graph]).to(device)
            ion_mode_idx = torch.tensor([sample_ion_mode], dtype=torch.long).to(device)
            precursor_idx = torch.tensor([sample_precursor_bin], dtype=torch.long).to(device)
            adduct_idx = torch.tensor([sample_adduct_idx], dtype=torch.long).to(device)
            _, attn_weights = model.transformer_encoder(spectrum, ion_mode_idx, precursor_idx, adduct_idx)
            _, _, edge_weights = model.gnn_encoder(graph_data, ion_mode_idx, precursor_idx, adduct_idx)
            plot_attention_weights(attn_weights, title=f"Fold {fold+1} Transformer Attention Weights")
            plot_gnn_edge_weights(edge_weights, sample_graph.edge_index, title=f"Fold {fold+1} GNN Edge Importance")

# Final Evaluation
print(f"Validity Rate: {validity_rate(pred_smiles_list):.2f}%")
print(f"Prediction Diversity: {prediction_diversity(pred_smiles_list):.4f}")
error_analysis(pred_smiles_list, true_smiles_list, adducts_list, all_fingerprints)
