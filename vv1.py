

# Import packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

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

# Binning spectra
def bin_spectrum(mzs, intensities, n_bins=1000, max_mz=1000):
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
    return spectrum

df['binned'] = df.apply(lambda row: bin_spectrum(row['mzs'], row['intensities']), axis=1)

# Canonicalize SMILES
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None

df['smiles'] = df['smiles'].apply(canonicalize_smiles)
df = df.dropna(subset=['smiles'])

# Preprocess ion mode and precursor m/z
df['ion_mode'] = df['adduct'].apply(lambda x: 0 if '+' in str(x) else 1 if '-' in str(x) else 0).fillna(0)
df['precursor_bin'] = pd.qcut(df['precursor_mz'], q=100, labels=False, duplicates='drop')

# Subsample for SSL pretraining (50% of training data)
df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
df_ssl = df_train.sample(frac=0.5, random_state=42)

# SMILES Tokenization
all_smiles = df_train['smiles'].tolist()
unique_chars = set(''.join(all_smiles)) | {MASK_TOKEN}
tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MASK_TOKEN] + sorted(unique_chars - {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MASK_TOKEN})
token_to_idx = {tok: i for i, tok in enumerate(tokens)}
idx_to_token = {i: tok for tok, i in token_to_idx.items()}
vocab_size = len(tokens)
PRETRAIN_MAX_LEN = 100  # Reduced for SSL pretraining
SUPERVISED_MAX_LEN = max(len(s) + 2 for s in all_smiles)  # +2 for SOS and EOS
print(f"Vocabulary size: {vocab_size}, Supervised MAX_LEN: {SUPERVISED_MAX_LEN}, Pretrain MAX_LEN: {PRETRAIN_MAX_LEN}")

def encode_smiles(smiles, max_len=PRETRAIN_MAX_LEN):
    tokens = [SOS_TOKEN] + list(smiles)[:max_len-2] + [EOS_TOKEN]
    token_ids = [token_to_idx.get(tok, token_to_idx[PAD_TOKEN]) for tok in tokens]
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids += [token_to_idx[PAD_TOKEN]] * (max_len - len(token_ids))
    return token_ids

# Dataset class
class MSMSDataset(Dataset):
    def __init__(self, dataframe, max_len=PRETRAIN_MAX_LEN, is_ssl=False):
        self.spectra = np.stack(dataframe['binned'].values)
        self.ion_modes = dataframe['ion_mode'].values
        self.precursor_bins = dataframe['precursor_bin'].values
        self.raw_smiles = dataframe['smiles'].values
        self.is_ssl = is_ssl
        if is_ssl:
            # Precompute masked SMILES for SSL
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
                torch.tensor(self.smiles[idx], dtype=torch.long),
                torch.tensor(self.masked_smiles[idx], dtype=torch.long),
                torch.tensor(self.ion_modes[idx], dtype=torch.long),
                torch.tensor(self.precursor_bins[idx], dtype=torch.long),
                self.raw_smiles[idx]
            )
        return (
            torch.tensor(self.spectra[idx], dtype=torch.float),
            torch.tensor(self.smiles[idx], dtype=torch.long),
            torch.tensor(self.ion_modes[idx], dtype=torch.long),
            torch.tensor(self.precursor_bins[idx], dtype=torch.long),
            self.raw_smiles[idx]
        )

train_dataset = MSMSDataset(df_train, max_len=SUPERVISED_MAX_LEN, is_ssl=False)
val_dataset = MSMSDataset(df_val, max_len=SUPERVISED_MAX_LEN, is_ssl=False)
ssl_dataset = MSMSDataset(df_ssl, max_len=PRETRAIN_MAX_LEN, is_ssl=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)
ssl_loader = DataLoader(ssl_dataset, batch_size=64, shuffle=True, num_workers=2)

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

# Metadata Embedding
class SpectrumMetadataEmbedding(nn.Module):
    def __init__(self, emb_dim=64, ion_mode_dim=2, precursor_bins=100):
        super().__init__()
        self.ion_emb = nn.Embedding(ion_mode_dim, emb_dim)
        self.prec_emb = nn.Embedding(precursor_bins, emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, ion_mode_idx, precursor_idx):
        ion_vec = self.ion_emb(ion_mode_idx)
        prec_vec = self.prec_emb(precursor_idx)
        combined = torch.cat([ion_vec, prec_vec], dim=-1)
        return self.linear(combined)

# Transformer Encoder
class SpectrumTransformerEncoder(nn.Module):
    def __init__(self, input_dim=1000, d_model=768, nhead=12, num_layers=8, dim_feedforward=2048, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.metadata_emb = SpectrumMetadataEmbedding(emb_dim=64)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.BatchNorm1d(d_model)
        self.fc = nn.Linear(d_model + 64, d_model)

    def forward(self, src, ion_mode_idx, precursor_idx):
        src = self.input_proj(src).unsqueeze(1)
        metadata = self.metadata_emb(ion_mode_idx, precursor_idx)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src).squeeze(1)
        output = self.norm(output)
        output = torch.cat([output, metadata], dim=-1)
        output = self.fc(output)
        return output

# Transformer Decoder
class SmilesTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=8, dim_feedforward=2048, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.norm = nn.BatchNorm1d(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.norm(output.transpose(1, 2)).transpose(1, 2)
        return self.output_layer(output)

# Full Transformer Model
class MSMS2SmilesTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=8, dim_feedforward=2048, dropout=0.2):
        super().__init__()
        self.encoder = SpectrumTransformerEncoder(input_dim=1000, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = SmilesTransformerDecoder(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout)

    def generate_square_subsequent_mask(self, tgt_len):
        mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask

    def forward(self, src, tgt, ion_mode_idx, precursor_idx, tgt_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, ion_mode_idx, precursor_idx).unsqueeze(1)
        output = self.decoder(tgt, memory, tgt_mask, memory_key_padding_mask)
        return output

# SSL Pretraining
def ssl_pretrain(model, dataloader, epochs=3, lr=1e-4):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_idx[PAD_TOKEN])
    for epoch in range(epochs):
        total_loss = 0
        for spectra, smiles_tokens, masked_tokens, ion_modes, precursor_bins, _ in tqdm(dataloader, desc=f"SSL Epoch {epoch+1}/{epochs}"):
            spectra = spectra.to(device)
            ion_modes = ion_modes.to(device)
            precursor_bins = precursor_bins.to(device)
            smiles_tokens = smiles_tokens.to(device)
            masked_tokens = masked_tokens.to(device)
            tgt_input = masked_tokens[:, :-1]
            tgt_output = smiles_tokens[:, 1:]
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            optimizer.zero_grad()
            output = model(spectra, tgt_input, ion_modes, precursor_bins, tgt_mask)
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"SSL Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

# Supervised Training with Checkpoints
def supervised_train(model, train_loader, val_loader, epochs=30, lr=1e-4, patience=5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_idx[PAD_TOKEN])
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for spectra, smiles_tokens, ion_modes, precursor_bins, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            spectra, smiles_tokens = spectra.to(device), smiles_tokens.to(device)
            ion_modes, precursor_bins = ion_modes.to(device), precursor_bins.to(device)
            tgt_input = smiles_tokens[:, :-1]
            tgt_output = smiles_tokens[:, 1:]
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            memory_key_padding_mask = None
            optimizer.zero_grad()
            output = model(spectra, tgt_input, ion_modes, precursor_bins, tgt_mask, memory_key_padding_mask)
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for spectra, smiles_tokens, ion_modes, precursor_bins, _ in val_loader:
                spectra, smiles_tokens = spectra.to(device), smiles_tokens.to(device)
                ion_modes, precursor_bins = ion_modes.to(device), precursor_bins.to(device)
                tgt_input = smiles_tokens[:, :-1]
                tgt_output = smiles_tokens[:, 1:]
                tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                memory_key_padding_mask = None
                output = model(spectra, tgt_input, ion_modes, precursor_bins, tgt_mask, memory_key_padding_mask)
                loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(avg_val_loss)

        # Save checkpoint every 10 epochs
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
            }, 'best_msms_transformer.pt')
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_val_loss

# RDKit-based Molecular Property Filter
def is_plausible_molecule(smiles, true_mol, max_mw=1500, min_logp=-7, max_logp=7):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    true_mw = Descriptors.MolWt(true_mol) if true_mol else 500
    return mw <= max_mw and min_logp <= logp <= max_logp and abs(mw - true_mw) < 300

# Beam Search with Nucleus Sampling
def beam_search(model, spectrum, ion_mode_idx, precursor_idx, true_smiles, beam_width=20, max_len=150, nucleus_p=0.9, device='cpu'):
    model.eval()
    true_mol = Chem.MolFromSmiles(true_smiles) if true_smiles else None
    with torch.no_grad():
        spectrum = spectrum.unsqueeze(0).to(device)
        ion_mode_idx = torch.tensor([ion_mode_idx], dtype=torch.long).to(device)
        precursor_idx = torch.tensor([precursor_idx], dtype=torch.long).to(device)
        memory = model.encoder(spectrum, ion_mode_idx, precursor_idx).unsqueeze(1)
        sequences = [([token_to_idx[SOS_TOKEN]], 0.0)]

        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == token_to_idx[EOS_TOKEN]:
                    all_candidates.append((seq, score))
                    continue
                tgt_input = torch.tensor([seq], dtype=torch.long).to(device)
                tgt_mask = model.generate_square_subsequent_mask(len(seq)).to(device)
                outputs = model.decoder(tgt_input, memory, tgt_mask)
                log_probs = F.log_softmax(outputs[0, -1], dim=-1).cpu().numpy()
                sorted_probs = np.sort(np.exp(log_probs))[::-1]
                cumulative_probs = np.cumsum(sorted_probs)
                cutoff_idx = np.searchsorted(cumulative_probs, nucleus_p)
                top_tokens = np.argsort(log_probs)[-cutoff_idx:] if cutoff_idx > 0 else np.argsort(log_probs)[-1:]
                top_probs = np.exp(log_probs[top_tokens]) / np.sum(np.exp(log_probs[top_tokens]))
                for tok in np.random.choice(top_tokens, size=min(beam_width, len(top_tokens)), p=top_probs):
                    diversity_penalty = 0.2 * sum(1 for s, _ in sequences if tok in s[1:-1])
                    candidate = (seq + [int(tok)], score + log_probs[tok] - diversity_penalty)
                    all_candidates.append(candidate)
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq[-1] == token_to_idx[EOS_TOKEN] for seq, _ in sequences):
                break

        results = []
        for seq, score in sequences:
            smiles = ''.join([idx_to_token.get(idx, '') for idx in seq[1:-1]])
            try:
                if Chem.MolFromSmiles(smiles) and is_plausible_molecule(smiles, true_mol):
                    confidence = np.exp(score / len(seq))
                    results.append((smiles, confidence))
            except:
                continue
        return results if results else [("Invalid SMILES", 0.0)]

# Tanimoto Similarity with MorganGenerator
def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 and mol2:
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = morgan_gen.GetFingerprint(mol1)
        fp2 = morgan_gen.GetFingerprint(mol2)
        return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
    return 0.0

# Initialize and Train Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MSMS2SmilesTransformer(vocab_size=vocab_size, d_model=768, nhead=12, num_layers=8, dim_feedforward=2048, dropout=0.2).to(device)

# SSL Pretraining
print("Starting SSL pretraining...")
ssl_pretrain(model, ssl_loader, epochs=3)

# Supervised Training
print("Starting supervised training...")
best_val_loss = supervised_train(model, train_loader, val_loader, epochs=30, patience=5)

print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
print("Model saved as 'best_msms_transformer.pt'")

# Inference and Visualization on Multiple Samples
num_samples = 5
for sample_idx in range(num_samples):
    sample_spectrum = torch.tensor(df_val['binned'].iloc[sample_idx], dtype=torch.float)
    sample_ion_mode = df_val['ion_mode'].iloc[sample_idx]
    sample_precursor_bin = df_val['precursor_bin'].iloc[sample_idx]
    true_smiles = df_val['smiles'].iloc[sample_idx]

    predicted_results = beam_search(model, sample_spectrum, sample_ion_mode, sample_precursor_bin, true_smiles, beam_width=20, max_len=SUPERVISED_MAX_LEN, device=device)
    print(f"\nSample {sample_idx} - True SMILES: {true_smiles}")
    print("Top Predicted SMILES:")
    for smiles, confidence in predicted_results[:3]:
        similarity = tanimoto_similarity(true_smiles, smiles)
        print(f"SMILES: {smiles}, Confidence: {confidence:.4f}, Tanimoto: {similarity:.4f}")
        if len(smiles) > 100 and smiles.count('C') > len(smiles) * 0.8:
            print("Warning: Predicted SMILES is a long carbon chain, indicating potential model underfitting.")
        if smiles != "Invalid SMILES":
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                print(f"Molecular Weight: {Descriptors.MolWt(mol):.2f}, LogP: {Descriptors.MolLogP(mol):.2f}")

    # Visualize molecules
    if predicted_results[0][0] != "Invalid SMILES":
        pred_mol = Chem.MolFromSmiles(predicted_results[0][0])
        true_mol = Chem.MolFromSmiles(true_smiles)
        if pred_mol and true_mol:
            img = Draw.MolsToGridImage([true_mol, pred_mol], molsPerRow=2, subImgSize=(300, 300), legends=['True', 'Predicted'])
            img_array = np.array(img.convert('RGB'))
            plt.figure(figsize=(10, 5))
            plt.imshow(img_array)
            plt.axis('off')
            plt.title(f"Sample {sample_idx} - Tanimoto: {tanimoto_similarity(true_smiles, predicted_results[0][0]):.4f}")
            plt.show()
