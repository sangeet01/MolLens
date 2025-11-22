"""
ChEMBL Pretraining for SimpleMass Model
Download ChEMBL data, generate synthetic spectra, pretrain model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import Dict, List, Optional
import logging

# ChEMBL API
from chembl_webresource_client.new_client import new_client

# RDKit for molecular processing
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors

# Import from main model
from simplemass_production import HybridMassConfig, HybridSimpleMass, MassSpecGymMetadataEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChEMBLDataLoader:
    """Download and process ChEMBL data"""
    
    def __init__(self, max_molecules: int = 100000):
        self.max_molecules = max_molecules
        self.molecule_client = new_client.molecule
        
    def download_chembl_molecules(self) -> pd.DataFrame:
        """Download molecules from ChEMBL API"""
        logger.info(f"Downloading {self.max_molecules} molecules from ChEMBL...")
        
        molecules = self.molecule_client.filter(
            molecule_properties__mw_freebase__lte=1000,  # MW <= 1000 Da
            molecule_properties__mw_freebase__gte=100,   # MW >= 100 Da
            molecule_structures__isnull=False
        ).only(['molecule_chembl_id', 'molecule_structures', 'molecule_properties'])
        
        chembl_data = []
        count = 0
        
        for mol in tqdm(molecules, desc="Processing molecules"):
            if count >= self.max_molecules:
                break
                
            if (mol['molecule_structures'] and 
                mol['molecule_structures']['canonical_smiles'] and
                mol['molecule_properties']):
                
                smiles = mol['molecule_structures']['canonical_smiles']
                
                # Validate SMILES
                if self._is_valid_smiles(smiles):
                    chembl_data.append({
                        'chembl_id': mol['molecule_chembl_id'],
                        'smiles': smiles,
                        'mw': mol['molecule_properties']['mw_freebase'],
                        'logp': mol['molecule_properties'].get('alogp', None),
                        'psa': mol['molecule_properties'].get('psa', None)
                    })
                    count += 1
        
        df = pd.DataFrame(chembl_data)
        logger.info(f"Downloaded {len(df)} valid molecules")
        return df
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Validate SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None and len(smiles) <= 200
        except:
            return False

class SyntheticSpectrumGenerator:
    """Generate synthetic MS/MS spectra from SMILES"""
    
    def __init__(self):
        self.collision_energies = [10, 20, 30, 40, 50]
        self.adducts = {
            '[M+H]+': 1.007276,
            '[M-H]-': -1.007276,
            '[M+Na]+': 22.989218,
            '[M+K]+': 38.963158
        }
        
    def generate_spectrum(self, smiles: str, collision_energy: float = 30.0, 
                         adduct: str = '[M+H]+') -> Optional[Dict]:
        """Generate synthetic spectrum from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            
            # Calculate precursor m/z
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            precursor_mz = mw + self.adducts[adduct]
            
            # Generate fragments
            fragments = self._generate_fragments(mol, collision_energy, precursor_mz)
            
            if len(fragments) < 5:  # Need minimum peaks
                return None
            
            return {
                'smiles': smiles,
                'precursor_mz': precursor_mz,
                'collision_energy': collision_energy,
                'adduct': adduct,
                'peaks': fragments,
                'parent_mass': mw
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate spectrum for {smiles}: {e}")
            return None
    
    def _generate_fragments(self, mol, collision_energy: float, precursor_mz: float) -> List[List[float]]:
        """Generate synthetic fragment peaks"""
        # Simple fragmentation model based on molecular properties
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        # Number of fragments based on collision energy and molecular size
        base_fragments = min(50, max(5, int(num_bonds * collision_energy / 100)))
        
        fragments = []
        
        # Add precursor peak (low intensity at high CE)
        precursor_intensity = max(10, 1000 * (50 - collision_energy) / 50)
        fragments.append([precursor_mz, precursor_intensity])
        
        # Generate fragment peaks
        for i in range(base_fragments):
            # Fragment m/z (random but realistic)
            if i < base_fragments // 3:  # Low m/z fragments
                frag_mz = np.random.uniform(50, mw * 0.3)
            elif i < 2 * base_fragments // 3:  # Mid m/z fragments
                frag_mz = np.random.uniform(mw * 0.3, mw * 0.7)
            else:  # High m/z fragments
                frag_mz = np.random.uniform(mw * 0.7, mw * 0.95)
            
            # Intensity based on collision energy and fragment type
            base_intensity = np.random.exponential(500)
            intensity_factor = collision_energy / 30.0  # Normalize to CE=30
            intensity = base_intensity * intensity_factor
            
            fragments.append([frag_mz, intensity])
        
        # Sort by m/z and normalize intensities
        fragments = sorted(fragments, key=lambda x: x[0])
        max_intensity = max(f[1] for f in fragments)
        fragments = [[f[0], (f[1] / max_intensity) * 999] for f in fragments]
        
        return fragments[:50]  # Top 50 peaks

class ChEMBLPretrainDataset(Dataset):
    """ChEMBL pretraining dataset"""
    
    def __init__(self, synthetic_spectra: List[Dict], vocab: Dict[str, int], config: HybridMassConfig):
        self.data = synthetic_spectra
        self.vocab = vocab
        self.config = config
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab.keys())}
        
        # Adduct mappings
        self.adduct_to_idx = {
            '[M+H]+': 1, '[M-H]-': 2, '[M+Na]+': 3, '[M+K]+': 4
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize SMILES
        smiles_tokens = self._tokenize_smiles(item['smiles'])
        
        # Process spectrum peaks
        peaks = torch.tensor(item['peaks'], dtype=torch.float32)
        
        # Pad/truncate peaks
        if len(peaks) > self.config.max_peaks:
            peaks = peaks[:self.config.max_peaks]
        else:
            padding = torch.zeros(self.config.max_peaks - len(peaks), 2)
            peaks = torch.cat([peaks, padding], dim=0)
        
        # Metadata
        metadata = {
            'precursor_mz': torch.tensor(item['precursor_mz'], dtype=torch.float32),
            'collision_energy': torch.tensor(item['collision_energy'], dtype=torch.float32),
            'adduct': torch.tensor(self.adduct_to_idx.get(item['adduct'], 0), dtype=torch.long),
            'instrument': torch.tensor(0, dtype=torch.long),  # Unknown for synthetic
            'parent_mass': torch.tensor(item['parent_mass'], dtype=torch.float32)
        }
        
        return {
            'peaks': peaks,
            'smiles_tokens': smiles_tokens,
            'metadata': metadata,
            'smiles_str': item['smiles']
        }
    
    def _tokenize_smiles(self, smiles: str) -> torch.Tensor:
        """Convert SMILES to token indices"""
        tokens = [self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 0)) for char in smiles]
        
        # Pad/truncate
        if len(tokens) > self.config.max_smiles_len:
            tokens = tokens[:self.config.max_smiles_len]
        else:
            tokens.extend([self.char_to_idx.get('<PAD>', 0)] * (self.config.max_smiles_len - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long)

def create_vocab_from_chembl(chembl_df: pd.DataFrame) -> Dict[str, int]:
    """Create vocabulary from ChEMBL SMILES"""
    all_chars = set()
    for smiles in chembl_df['smiles']:
        all_chars.update(smiles)
    
    # Add special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
    for i, char in enumerate(sorted(all_chars), 4):
        vocab[char] = i
    
    logger.info(f"Created vocabulary with {len(vocab)} tokens")
    return vocab

def generate_synthetic_dataset(chembl_df: pd.DataFrame, output_path: str = "chembl_synthetic.pkl"):
    """Generate synthetic spectra dataset"""
    generator = SyntheticSpectrumGenerator()
    synthetic_data = []
    
    logger.info("Generating synthetic spectra...")
    
    for _, row in tqdm(chembl_df.iterrows(), total=len(chembl_df), desc="Generating spectra"):
        smiles = row['smiles']
        
        # Generate multiple spectra per molecule (different conditions)
        for ce in [20, 30, 40]:  # Different collision energies
            for adduct in ['[M+H]+', '[M-H]-']:  # Different adducts
                spectrum = generator.generate_spectrum(smiles, ce, adduct)
                if spectrum:
                    synthetic_data.append(spectrum)
    
    logger.info(f"Generated {len(synthetic_data)} synthetic spectra")
    
    # Save dataset
    with open(output_path, 'wb') as f:
        pickle.dump(synthetic_data, f)
    
    return synthetic_data

def pretrain_model(synthetic_data: List[Dict], vocab: Dict[str, int], 
                  model_save_path: str = "chembl_pretrained.pt"):
    """Pretrain model on ChEMBL synthetic data"""
    
    config = HybridMassConfig()
    config.vocab_size = len(vocab)
    config.max_epochs = 50  # Pretraining epochs
    config.batch_size = 32
    config.learning_rate = 1e-4
    
    # Create dataset and dataloader
    dataset = ChEMBLPretrainDataset(synthetic_data, vocab, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = HybridSimpleMass(config, vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    # Training loop
    model.train()
    logger.info("Starting pretraining...")
    
    for epoch in range(config.max_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.max_epochs}"):
            # Move to device
            peaks = batch['peaks'].to(device)
            smiles_tokens = batch['smiles_tokens'].to(device)
            metadata = {k: v.to(device) for k, v in batch['metadata'].items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = model(
                    spectrum=peaks,
                    metadata=metadata,
                    target_smiles=smiles_tokens
                )
                
                loss = outputs.get('total_loss', outputs.get('flow_loss', 0))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'vocab': vocab,
                'loss': avg_loss
            }, f"chembl_pretrain_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab': vocab,
        'final_loss': avg_loss
    }, model_save_path)
    
    logger.info(f"Pretraining completed. Model saved to {model_save_path}")
    return model

def main():
    """Main pretraining pipeline"""
    
    # 1. Download ChEMBL data
    chembl_loader = ChEMBLDataLoader(max_molecules=50000)  # Start with 50K
    chembl_df = chembl_loader.download_chembl_molecules()
    chembl_df.to_csv("chembl_molecules.csv", index=False)
    
    # 2. Create vocabulary
    vocab = create_vocab_from_chembl(chembl_df)
    with open("chembl_vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
    
    # 3. Generate synthetic spectra
    synthetic_data = generate_synthetic_dataset(chembl_df, "chembl_synthetic.pkl")
    
    # 4. Pretrain model
    model = pretrain_model(synthetic_data, vocab, "chembl_pretrained.pt")
    
    logger.info("ChEMBL pretraining pipeline completed!")
    logger.info(f"- Downloaded {len(chembl_df)} molecules")
    logger.info(f"- Generated {len(synthetic_data)} synthetic spectra")
    logger.info(f"- Vocabulary size: {len(vocab)}")
    logger.info("- Model saved to chembl_pretrained.pt")

if __name__ == "__main__":
    main()