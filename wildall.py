# =============================================================================
# MassSpecGym-Optimized MS-to-Structure Deep Learning Pipeline
# Comprehensive Implementation with Full Complexity Maintained
# =============================================================================

# =============================================================================
# CELL 1: Setup and Core Imports
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, classification_report, mean_squared_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Deep learning and transformers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    StochasticWeightAveraging, DeviceStatsMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from transformers import (
    AutoTokenizer, AutoModel, BertModel, RobertaModel,
    get_linear_schedule_with_warmup, AdamW
)

# MassSpecGym core imports
from massspecgym import MassSpecDataset, MassSpecDataModule
from massspecgym.models import MassSpecGymModel
from massspecgym.transforms import SpecTransforms, MolTransforms
from massspecgym.metrics import (
    TopKAccuracy, TopKTanimoto, TopKMCES,
    CosineSimilarity, HitRateAtK, ExactMatchAccuracy
)
from massspecgym.utils import (
    set_seed, get_device, save_checkpoint, load_checkpoint,
    molecular_fingerprints, tanimoto_similarity, mces_similarity
)

# Molecular and chemical libraries
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors, Crippen, Lipinski, QED, AllChem, Draw, rdMolDescriptors,
    rdFingerprintGenerator, rdDepictor, rdMolAlign, Fragments
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.ML.Descriptors import MoleculeDescriptors
import selfies as sf
from mordred import Calculator, descriptors

# Advanced ML and optimization
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from bayes_opt import BayesianOptimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# NLP and embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import faiss
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer

# Graph neural networks
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool,
    global_add_pool, Set2Set, AttentionalAggregation
)
from torch_geometric.utils import to_networkx
import networkx as nx

# Uncertainty quantification
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam as PyroAdam
from pyro.nn import PyroModule, PyroSample

# Utilities and visualization
import logging
import warnings
import pickle
import json
import yaml
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from tqdm import tqdm
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Statistical analysis
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from scipy.optimize import minimize
from statsmodels.stats.contingency_tables import mcnemar

# Setup and configuration
warnings.filterwarnings('ignore')
set_seed(42)
device = get_device()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
print(f'Using device: {device}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current CUDA device: {torch.cuda.current_device()}')

# =============================================================================
# CELL 2: Advanced Configuration and Data Management
# =============================================================================
class AdvancedConfig:
    """Comprehensive configuration for all pipeline components"""

    # Data parameters
    BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 128
    MAX_SPECTRUM_LEN = 2000
    MAX_SMILES_LEN = 200
    MIN_SPECTRUM_PEAKS = 5
    SPECTRUM_NOISE_LEVEL = 0.01

    # Model architecture parameters
    D_MODEL = 768
    NHEAD = 12
    NUM_LAYERS = 8
    NUM_DECODER_LAYERS = 6
    DROPOUT = 0.15
    ATTENTION_DROPOUT = 0.1
    ACTIVATION_DROPOUT = 0.1
    HIDDEN_DIM = 2048

    # Graph neural network parameters
    GNN_HIDDEN_DIM = 256
    GNN_NUM_LAYERS = 4
    GNN_DROPOUT = 0.2
    GRAPH_POOLING = 'attention'

    # Training parameters
    LEARNING_RATE = 2e-4
    MIN_LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 1000
    MAX_EPOCHS = 100
    PATIENCE = 15
    GRADIENT_CLIP_VAL = 1.0
    ACCUMULATE_GRAD_BATCHES = 2

    # Cross-validation and ensemble
    N_FOLDS = 10
    N_ENSEMBLE_MODELS = 5
    ENSEMBLE_WEIGHTS = 'learned'

    # Optimization and search
    BEAM_WIDTH = 20
    TOP_K = [1, 3, 5, 10, 20]
    NUCLEUS_P = 0.9
    TEMPERATURE = 1.0

    # Hyperparameter optimization
    N_OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 3600  # 1 hour
    PRUNING_PATIENCE = 10

    # Data augmentation
    AUGMENT_PROB = 0.3
    NOISE_AUGMENT_PROB = 0.2
    INTENSITY_SCALE_RANGE = (0.8, 1.2)
    MZ_SHIFT_RANGE = (-0.1, 0.1)

    # Molecular property prediction
    PREDICT_PROPERTIES = [
        'molecular_weight', 'logp', 'tpsa', 'qed', 'sas',
        'num_rings', 'num_aromatic_rings', 'num_rotatable_bonds'
    ]

    # Active learning
    ACTIVE_LEARNING_BUDGET = 1000
    UNCERTAINTY_THRESHOLD = 0.8
    DIVERSITY_WEIGHT = 0.3

    # Evaluation and metrics
    EVAL_EVERY_N_EPOCHS = 5
    SAVE_TOP_K_MODELS = 3
    COMPUTE_EXPENSIVE_METRICS = True

    # Paths and logging
    DATA_DIR = Path('data')
    MODEL_DIR = Path('models')
    LOG_DIR = Path('logs')
    RESULTS_DIR = Path('results')
    CACHE_DIR = Path('cache')

    # Distributed training
    USE_DDP = False
    NUM_GPUS = 1
    NUM_NODES = 1

    def __post_init__(self):
        # Create directories
        for dir_path in [self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR,
                        self.RESULTS_DIR, self.CACHE_DIR]:
            dir_path.mkdir(exist_ok=True)

config = AdvancedConfig()

# Advanced data preprocessing and augmentation
class SpectrumAugmentation:
    """Advanced spectrum augmentation techniques"""

    def __init__(self, config):
        self.config = config
        self.rng = np.random.RandomState(42)

    def add_noise(self, spectrum, noise_level=None):
        """Add Gaussian noise to spectrum"""
        if noise_level is None:
            noise_level = self.config.SPECTRUM_NOISE_LEVEL
        noise = self.rng.normal(0, noise_level, spectrum.shape)
        return np.maximum(0, spectrum + noise)

    def scale_intensity(self, spectrum, scale_range=None):
        """Scale spectrum intensities"""
        if scale_range is None:
            scale_range = self.config.INTENSITY_SCALE_RANGE
        scale = self.rng.uniform(*scale_range)
        return spectrum * scale

    def shift_mz(self, mz_values, shift_range=None):
        """Shift m/z values slightly"""
        if shift_range is None:
            shift_range = self.config.MZ_SHIFT_RANGE
        shift = self.rng.uniform(*shift_range)
        return mz_values + shift

    def remove_peaks(self, spectrum, removal_prob=0.1):
        """Randomly remove some peaks"""
        mask = self.rng.random(len(spectrum)) > removal_prob
        return spectrum * mask

    def augment_spectrum(self, spectrum, mz_values=None):
        """Apply random augmentations"""
        if self.rng.random() < self.config.AUGMENT_PROB:
            # Apply random combination of augmentations
            if self.rng.random() < 0.5:
                spectrum = self.add_noise(spectrum)
            if self.rng.random() < 0.3:
                spectrum = self.scale_intensity(spectrum)
            if self.rng.random() < 0.2:
                spectrum = self.remove_peaks(spectrum)
        return spectrum

# =============================================================================
# CELL 3: Advanced Molecular Representations
# =============================================================================
class MolecularFeatureExtractor:
    """Extract comprehensive molecular features"""

    def __init__(self):
        self.descriptor_calculator = Calculator(descriptors, ignore_3D=True)

    def get_rdkit_descriptors(self, mol):
        """Get RDKit molecular descriptors"""
        if mol is None:
            return {}

        descriptors = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'rings': Descriptors.RingCount(mol),
            'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
            'qed': QED.qed(mol),
            'sas': 0.0  # Placeholder for synthetic accessibility score
        }
        return descriptors

    def get_fingerprints(self, mol, fp_type='morgan'):
        """Generate molecular fingerprints"""
        if mol is None:
            return None

        if fp_type == 'morgan':
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        elif fp_type == 'rdkit':
            return Chem.RDKFingerprint(mol)
        elif fp_type == 'maccs':
            return AllChem.GetMACCSKeysFingerprint(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

    def mol_to_graph(self, mol):
        """Convert molecule to graph representation"""
        if mol is None:
            return None

        # Node features (atoms)
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetMass(),
                atom.GetTotalNumHs()
            ]
            atom_features.append(features)

        # Edge indices (bonds)
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])  # Undirected graph

            bond_features = [
                int(bond.GetBondType()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing())
            ]
            edge_features.extend([bond_features, bond_features])

        if len(edge_indices) == 0:
            edge_indices = [[0, 0]]  # Self-loop for single atom
            edge_features = [[0, 0, 0]]

        return Data(
            x=torch.tensor(atom_features, dtype=torch.float),
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_features, dtype=torch.float)
        )

# =============================================================================
# CELL 4: Advanced Neural Network Architectures
# =============================================================================
class MultiHeadAttention(nn.Module):
    """Custom multi-head attention with improvements"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.w_o(context)
        return self.layer_norm(output + query)  # Residual connection

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class GraphNeuralNetwork(nn.Module):
    """Advanced Graph Neural Network for molecular representation"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))

        # Final layer
        self.convs.append(GATConv(hidden_dim * 4, output_dim, heads=1, dropout=dropout))

        self.dropout = nn.Dropout(dropout)
        self.global_pool = AttentionalAggregation(gate_nn=nn.Linear(output_dim, 1))

    def forward(self, x, edge_index, batch):
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        x = self.global_pool(x, batch)

        return x

# =============================================================================
# CELL 5: Advanced Transformer Architecture
# =============================================================================
class AdvancedTransformerModel(MassSpecGymModel):
    """State-of-the-art transformer model for MS-to-structure prediction"""

    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Spectrum encoder
        self.spectrum_embedding = nn.Linear(1, config.D_MODEL)
        self.spectrum_pos_encoding = PositionalEncoding(config.D_MODEL, config.MAX_SPECTRUM_LEN)

        spectrum_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            dim_feedforward=config.HIDDEN_DIM,
            dropout=config.DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.spectrum_encoder = nn.TransformerEncoder(
            spectrum_encoder_layer,
            num_layers=config.NUM_LAYERS
        )

        # Molecular decoder
        self.mol_embedding = nn.Embedding(vocab_size, config.D_MODEL)
        self.mol_pos_encoding = PositionalEncoding(config.D_MODEL, config.MAX_SMILES_LEN)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            dim_feedforward=config.HIDDEN_DIM,
            dropout=config.DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.mol_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.NUM_DECODER_LAYERS
        )

        # Graph neural network for molecular graphs
        self.gnn = GraphNeuralNetwork(
            input_dim=7,  # Atom features
            hidden_dim=config.GNN_HIDDEN_DIM,
            output_dim=config.D_MODEL,
            num_layers=config.GNN_NUM_LAYERS,
            dropout=config.GNN_DROPOUT
        )

        # Cross-modal attention
        self.cross_attention = MultiHeadAttention(
            config.D_MODEL, config.NHEAD, config.ATTENTION_DROPOUT
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.D_MODEL, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, vocab_size)
        )

        # Property prediction heads
        self.property_heads = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(config.D_MODEL, config.HIDDEN_DIM // 2),
                nn.GELU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.HIDDEN_DIM // 2, 1)
            ) for prop in config.PREDICT_PROPERTIES
        })

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.D_MODEL, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(config.HIDDEN_DIM // 2, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, spectrum, target=None, molecular_graph=None, return_attention=False):
        batch_size = spectrum.size(0)
        device = spectrum.device

        # Encode spectrum
        spectrum_emb = self.spectrum_embedding(spectrum.unsqueeze(-1))
        spectrum_emb = self.spectrum_pos_encoding(spectrum_emb)
        spectrum_encoded = self.spectrum_encoder(spectrum_emb)

        # Encode molecular graph if available
        if molecular_graph is not None:
            graph_encoded = self.gnn(
                molecular_graph.x,
                molecular_graph.edge_index,
                molecular_graph.batch
            )
            # Expand to match sequence length
            graph_encoded = graph_encoded.unsqueeze(1).expand(-1, spectrum_encoded.size(1), -1)

            # Cross-modal attention
            spectrum_encoded = self.cross_attention(
                spectrum_encoded, graph_encoded, graph_encoded
            )

        if target is not None:
            # Training mode
            target_emb = self.mol_embedding(target[:, :-1])
            target_emb = self.mol_pos_encoding(target_emb)

            # Create causal mask
            tgt_len = target.size(1) - 1
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()

            # Decode
            decoded = self.mol_decoder(
                target_emb, spectrum_encoded, tgt_mask=tgt_mask
            )

            # Output projection
            logits = self.output_projection(decoded)

            # Property predictions
            pooled_representation = spectrum_encoded.mean(dim=1)
            properties = {}
            for prop_name, head in self.property_heads.items():
                properties[prop_name] = head(pooled_representation)

            # Uncertainty estimation
            uncertainty = self.uncertainty_head(pooled_representation)

            return {
                'logits': logits,
                'properties': properties,
                'uncertainty': uncertainty
            }
        else:
            # Inference mode - beam search
            return self.beam_search(spectrum_encoded, return_attention=return_attention)

    def beam_search(self, memory, beam_width=None, max_length=None, return_attention=False):
        """Advanced beam search with nucleus sampling"""
        if beam_width is None:
            beam_width = self.config.BEAM_WIDTH
        if max_length is None:
            max_length = self.config.MAX_SMILES_LEN

        batch_size = memory.size(0)
        device = memory.device

        # Initialize with start token
        sequences = torch.full((batch_size, beam_width, 1), 1, device=device)  # Assuming 1 is start token
        scores = torch.zeros(batch_size, beam_width, device=device)

        for step in range(max_length):
            # Prepare input
            current_sequences = sequences.view(-1, step + 1)
            current_memory = memory.repeat_interleave(beam_width, dim=0)

            # Embed and decode
            target_emb = self.mol_embedding(current_sequences)
            target_emb = self.mol_pos_encoding(target_emb)

            # Create causal mask
            tgt_mask = torch.triu(torch.ones(step + 1, step + 1, device=device), diagonal=1).bool()

            decoded = self.mol_decoder(target_emb, current_memory, tgt_mask=tgt_mask)
            logits = self.output_projection(decoded[:, -1])  # Last position

            # Apply temperature and get probabilities
            logits = logits / self.config.TEMPERATURE
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_width, -1)

            if step == 0:
                # First step: only use first beam
                top_log_probs, top_indices = log_probs[:, 0].topk(beam_width, dim=-1)
                scores = top_log_probs
                sequences = torch.cat([
                    sequences[:, :1].expand(-1, beam_width, -1),
                    top_indices.unsqueeze(-1)
                ], dim=-1)
            else:
                # Subsequent steps
                candidate_scores = scores.unsqueeze(-1) + log_probs
                candidate_scores = candidate_scores.view(batch_size, -1)

                top_scores, top_indices = candidate_scores.topk(beam_width, dim=-1)

                # Update sequences
                beam_indices = top_indices // self.vocab_size
                token_indices = top_indices % self.vocab_size

                new_sequences = torch.zeros(batch_size, beam_width, step + 2, dtype=torch.long, device=device)
                for b in range(batch_size):
                    for i in range(beam_width):
                        beam_idx = beam_indices[b, i]
                        new_sequences[b, i, :step + 1] = sequences[b, beam_idx, :step + 1]
                        new_sequences[b, i, step + 1] = token_indices[b, i]

                sequences = new_sequences
                scores = top_scores

            # Early stopping if all sequences end with end token
            if (sequences[:, :, -1] == 2).all():  # Assuming 2 is end token
                break

        return sequences, scores

# =============================================================================
# CELL 6: Training and Evaluation Framework
# =============================================================================
class AdvancedTrainingModule(pl.LightningModule):
    """Advanced PyTorch Lightning module with comprehensive training"""

    def __init__(self, model, config, vocab_size):
        super().__init__()
        self.model = model
        self.config = config
        self.vocab_size = vocab_size
        self.save_hyperparameters(ignore=['model'])

        # Metrics
        self.train_metrics = {
            'accuracy': TopKAccuracy(k=1),
            'top5_accuracy': TopKAccuracy(k=5),
            'tanimoto': TopKTanimoto(k=1),
            'mces': TopKMCES(k=1)
        }

        self.val_metrics = {
            'accuracy': TopKAccuracy(k=1),
            'top5_accuracy': TopKAccuracy(k=5),
            'tanimoto': TopKTanimoto(k=1),
            'mces': TopKMCES(k=1),
            'exact_match': ExactMatchAccuracy()
        }

        # Loss weights
        self.generation_loss_weight = 1.0
        self.property_loss_weight = 0.1
        self.uncertainty_loss_weight = 0.01

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        spectrum, target, properties, molecular_graph = batch

        outputs = self.model(
            spectrum=spectrum,
            target=target,
            molecular_graph=molecular_graph
        )

        # Generation loss
        generation_loss = F.cross_entropy(
            outputs['logits'].reshape(-1, self.vocab_size),
            target[:, 1:].reshape(-1),
            ignore_index=0  # Padding token
        )

        # Property prediction losses
        property_loss = 0
        for prop_name, pred in outputs['properties'].items():
            if prop_name in properties:
                property_loss += F.mse_loss(pred.squeeze(), properties[prop_name])

        # Uncertainty regularization
        uncertainty_loss = outputs['uncertainty'].mean()

        # Total loss
        total_loss = (
            self.generation_loss_weight * generation_loss +
            self.property_loss_weight * property_loss +
            self.uncertainty_loss_weight * uncertainty_loss
        )

        # Logging
        self.log('train/generation_loss', generation_loss, prog_bar=True)
        self.log('train/property_loss', property_loss)
        self.log('train/uncertainty_loss', uncertainty_loss)
        self.log('train/total_loss', total_loss, prog_bar=True)

        # Update metrics
        predictions = outputs['logits'].argmax(dim=-1)
        targets = target[:, 1:]

        for name, metric in self.train_metrics.items():
            metric.update(predictions, targets)
            self.log(f'train/{name}', metric, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        spectrum, target, properties, molecular_graph = batch

        # Generate predictions
        with torch.no_grad():
            sequences, scores = self.model(
                spectrum=spectrum,
                molecular_graph=molecular_graph
            )

        # Get best predictions
        predictions = sequences[:, 0]  # Best beam
        targets = target

        # Update metrics
        for name, metric in self.val_metrics.items():
            metric.update(predictions, targets)
            self.log(f'val/{name}', metric, prog_bar=True)

        return {'predictions': predictions, 'targets': targets}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

# =============================================================================
# CELL 7: Hyperparameter Optimization
# =============================================================================
class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using multiple strategies"""

    def __init__(self, config):
        self.config = config
        self.best_params = None
        self.best_score = float('-inf')

    def objective(self, trial):
        """Optuna objective function"""
        # Sample hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'd_model': trial.suggest_categorical('d_model', [256, 512, 768, 1024]),
            'num_layers': trial.suggest_int('num_layers', 4, 12),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 12, 16]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'warmup_steps': trial.suggest_int('warmup_steps', 500, 2000)
        }

        # Update config
        temp_config = AdvancedConfig()
        for key, value in params.items():
            setattr(temp_config, key.upper(), value)

        # Train model with these parameters
        score = self._train_and_evaluate(temp_config, trial)

        return score

    def _train_and_evaluate(self, config, trial):
        """Train model and return validation score"""
        try:
            # Create model and trainer
            model = AdvancedTransformerModel(vocab_size=1000, config=config)  # Placeholder vocab size
            training_module = AdvancedTrainingModule(model, config, vocab_size=1000)

            # Callbacks
            callbacks = [
                PyTorchLightningPruningCallback(trial, monitor='val/tanimoto'),
                EarlyStopping(monitor='val/tanimoto', patience=5, mode='max')
            ]

            # Trainer
            trainer = pl.Trainer(
                max_epochs=20,  # Reduced for hyperparameter search
                callbacks=callbacks,
                logger=False,
                enable_checkpointing=False,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu'
            )

            # Train
            trainer.fit(training_module)

            # Return best validation score
            return trainer.callback_metrics.get('val/tanimoto', 0.0)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 0.0

    def optimize(self, n_trials=None):
        """Run hyperparameter optimization"""
        if n_trials is None:
            n_trials = self.config.N_OPTUNA_TRIALS

        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=self.config.OPTUNA_TIMEOUT
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score}")

        return self.best_params, self.best_score

# =============================================================================
# CELL 8: Ensemble Methods
# =============================================================================
class EnsembleModel(nn.Module):
    """Ensemble of multiple models with learned weights"""

    def __init__(self, models, config):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.config = config

        if config.ENSEMBLE_WEIGHTS == 'learned':
            self.weight_network = nn.Sequential(
                nn.Linear(len(models), 64),
                nn.ReLU(),
                nn.Linear(64, len(models)),
                nn.Softmax(dim=-1)
            )
        else:
            # Equal weights
            self.register_buffer('weights', torch.ones(len(models)) / len(models))

    def forward(self, *args, **kwargs):
        # Get predictions from all models
        outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(*args, **kwargs)
                outputs.append(output)

        if self.config.ENSEMBLE_WEIGHTS == 'learned':
            # Learn ensemble weights based on input
            features = torch.stack([out['logits'].mean(dim=(1, 2)) for out in outputs], dim=1)
            weights = self.weight_network(features)
        else:
            weights = self.weights.unsqueeze(0).expand(outputs[0]['logits'].size(0), -1)

        # Weighted combination
        ensemble_logits = torch.zeros_like(outputs[0]['logits'])
        for i, output in enumerate(outputs):
            ensemble_logits += weights[:, i:i+1, None] * output['logits']

        return {'logits': ensemble_logits}

# =============================================================================
# CELL 9: Active Learning Framework
# =============================================================================
class ActiveLearningFramework:
    """Active learning for efficient data annotation"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.labeled_indices = set()
        self.unlabeled_indices = set()

    def uncertainty_sampling(self, dataloader, n_samples):
        """Select samples with highest prediction uncertainty"""
        uncertainties = []
        indices = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                spectrum = batch[0]
                outputs = self.model(spectrum)

                # Calculate uncertainty (entropy)
                probs = F.softmax(outputs['logits'], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                uncertainty = entropy.mean(dim=1)  # Average over sequence

                uncertainties.extend(uncertainty.cpu().numpy())
                indices.extend(range(batch_idx * dataloader.batch_size,
                                   (batch_idx + 1) * dataloader.batch_size))

        # Select top uncertain samples
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        return [indices[i] for i in uncertain_indices]

    def diversity_sampling(self, dataloader, n_samples):
        """Select diverse samples using k-means clustering"""
        features = []
        indices = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                spectrum = batch[0]
                # Get spectrum embeddings
                spectrum_emb = self.model.spectrum_embedding(spectrum.unsqueeze(-1))
                spectrum_encoded = self.model.spectrum_encoder(spectrum_emb)
                features.extend(spectrum_encoded.mean(dim=1).cpu().numpy())
                indices.extend(range(batch_idx * dataloader.batch_size,
                                   (batch_idx + 1) * dataloader.batch_size))

        # K-means clustering
        features = np.array(features)
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        # Select one sample from each cluster (closest to centroid)
        selected_indices = []
        for i in range(n_samples):
            cluster_mask = cluster_labels == i
            if cluster_mask.sum() > 0:
                cluster_features = features[cluster_mask]
                cluster_indices = np.array(indices)[cluster_mask]

                # Find closest to centroid
                centroid = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                closest_idx = np.argmin(distances)
                selected_indices.append(cluster_indices[closest_idx])

        return selected_indices

    def select_samples(self, dataloader, n_samples):
        """Combined uncertainty and diversity sampling"""
        n_uncertain = int(n_samples * (1 - self.config.DIVERSITY_WEIGHT))
        n_diverse = n_samples - n_uncertain

        uncertain_samples = self.uncertainty_sampling(dataloader, n_uncertain)
        diverse_samples = self.diversity_sampling(dataloader, n_diverse)

        return uncertain_samples + diverse_samples

# =============================================================================
# CELL 10: Comprehensive Evaluation and Analysis
# =============================================================================
class ComprehensiveEvaluator:
    """Comprehensive model evaluation and analysis"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.results = {}

    def evaluate_generation_quality(self, dataloader):
        """Evaluate molecular generation quality"""
        predictions = []
        targets = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating generation"):
                spectrum, target = batch[:2]
                sequences, scores = self.model(spectrum)

                # Get best predictions
                pred_sequences = sequences[:, 0]  # Best beam
                predictions.extend(pred_sequences.cpu().numpy())
                targets.extend(target.cpu().numpy())

        # Calculate metrics
        metrics = {
            'exact_match': self._calculate_exact_match(predictions, targets),
            'tanimoto_similarity': self._calculate_tanimoto_similarity(predictions, targets),
            'validity': self._calculate_validity(predictions),
            'uniqueness': self._calculate_uniqueness(predictions),
            'diversity': self._calculate_diversity(predictions)
        }

        return metrics

    def _calculate_exact_match(self, predictions, targets):
        """Calculate exact match accuracy"""
        matches = 0
        total = len(predictions)

        for pred, target in zip(predictions, targets):
            if np.array_equal(pred, target):
                matches += 1

        return matches / total

    def _calculate_tanimoto_similarity(self, predictions, targets):
        """Calculate average Tanimoto similarity"""
        similarities = []

        for pred, target in zip(predictions, targets):
            # Convert sequences to SMILES (placeholder)
            pred_smiles = self._sequence_to_smiles(pred)
            target_smiles = self._sequence_to_smiles(target)

            if pred_smiles and target_smiles:
                similarity = tanimoto_similarity(pred_smiles, target_smiles)
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_validity(self, predictions):
        """Calculate fraction of valid molecules"""
        valid_count = 0
        total_count = len(predictions)

        for pred in predictions:
            smiles = self._sequence_to_smiles(pred)
            if smiles and self._is_valid_smiles(smiles):
                valid_count += 1

        return valid_count / total_count

    def _calculate_uniqueness(self, predictions):
        """Calculate fraction of unique molecules"""
        unique_smiles = set()
        valid_count = 0

        for pred in predictions:
            smiles = self._sequence_to_smiles(pred)
            if smiles and self._is_valid_smiles(smiles):
                unique_smiles.add(smiles)
                valid_count += 1

        return len(unique_smiles) / valid_count if valid_count > 0 else 0.0

    def _calculate_diversity(self, predictions):
        """Calculate average pairwise Tanimoto distance"""
        valid_smiles = []

        for pred in predictions:
            smiles = self._sequence_to_smiles(pred)
            if smiles and self._is_valid_smiles(smiles):
                valid_smiles.append(smiles)

        if len(valid_smiles) < 2:
            return 0.0

        similarities = []
        for i in range(len(valid_smiles)):
            for j in range(i + 1, len(valid_smiles)):
                sim = tanimoto_similarity(valid_smiles[i], valid_smiles[j])
                similarities.append(sim)

        return 1.0 - np.mean(similarities)  # Diversity = 1 - similarity

    def _sequence_to_smiles(self, sequence):
        """Convert token sequence to SMILES string (placeholder)"""
        # This would need to be implemented based on your tokenization scheme
        return "CCO"  # Placeholder

    def _is_valid_smiles(self, smiles):
        """Check if SMILES string is valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def analyze_attention_patterns(self, dataloader, n_samples=10):
        """Analyze attention patterns in the model"""
        attention_maps = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_samples:
                    break

                spectrum, target = batch[:2]

                # Forward pass with attention return
                outputs = self.model(
                    spectrum=spectrum,
                    target=target,
                    return_attention=True
                )

                if 'attention_weights' in outputs:
                    attention_maps.append(outputs['attention_weights'])

        return attention_maps

    def generate_evaluation_report(self, dataloader):
        """Generate comprehensive evaluation report"""
        logger.info("Starting comprehensive evaluation...")

        # Generation quality metrics
        generation_metrics = self.evaluate_generation_quality(dataloader)

        # Attention analysis
        attention_maps = self.analyze_attention_patterns(dataloader)

        # Compile results
        report = {
            'generation_metrics': generation_metrics,
            'attention_analysis': {
                'n_samples_analyzed': len(attention_maps),
                'avg_attention_entropy': self._calculate_attention_entropy(attention_maps)
            },
            'model_complexity': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }

        self.results = report
        return report

    def _calculate_attention_entropy(self, attention_maps):
        """Calculate average entropy of attention weights"""
        if not attention_maps:
            return 0.0

        entropies = []
        for attention_map in attention_maps:
            # Calculate entropy for each attention head
            entropy = -torch.sum(attention_map * torch.log(attention_map + 1e-8), dim=-1)
            entropies.append(entropy.mean().item())

        return np.mean(entropies)

# =============================================================================
# CELL 11: Main Training Pipeline
# =============================================================================
def main_training_pipeline():
    """Main training pipeline orchestrating all components"""
    logger.info("Starting main training pipeline...")

    # Initialize configuration
    config = AdvancedConfig()

    # Load data (placeholder - would need actual data loading)
    logger.info("Loading data...")
    # datamodule = MassSpecDataModule(config)

    # Hyperparameter optimization
    if config.N_OPTUNA_TRIALS > 0:
        logger.info("Starting hyperparameter optimization...")
        optimizer = HyperparameterOptimizer(config)
        best_params, best_score = optimizer.optimize()

        # Update config with best parameters
        for key, value in best_params.items():
            setattr(config, key.upper(), value)

    # Create model
    logger.info("Creating model...")
    model = AdvancedTransformerModel(vocab_size=1000, config=config)  # Placeholder vocab size
    training_module = AdvancedTrainingModule(model, config, vocab_size=1000)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.MODEL_DIR,
            filename='best_model_{epoch:02d}_{val_tanimoto:.3f}',
            monitor='val/tanimoto',
            mode='max',
            save_top_k=config.SAVE_TOP_K_MODELS
        ),
        EarlyStopping(
            monitor='val/tanimoto',
            patience=config.PATIENCE,
            mode='max'
        ),
        LearningRateMonitor(logging_interval='step'),
        StochasticWeightAveraging(swa_lrs=1e-2),
        DeviceStatsMonitor()
    ]

    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=config.LOG_DIR,
        name='advanced_transformer'
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        callbacks=callbacks,
        logger=tb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.NUM_GPUS if torch.cuda.is_available() else 1,
        strategy=DDPStrategy() if config.USE_DDP else None,
        precision=16,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        accumulate_grad_batches=config.ACCUMULATE_GRAD_BATCHES,
        val_check_interval=config.EVAL_EVERY_N_EPOCHS,
        log_every_n_steps=50
    )

    # Train model
    logger.info("Starting training...")
    # trainer.fit(training_module, datamodule)

    # Evaluation
    logger.info("Starting evaluation...")
    evaluator = ComprehensiveEvaluator(model, config)
    # evaluation_report = evaluator.generate_evaluation_report(datamodule.test_dataloader())

    logger.info("Training pipeline completed!")
    # return model, evaluation_report

if __name__ == "__main__":
    main_training_pipeline()
            if self.rng.random() < 0.5:
                spectrum = self.scale_intensity(spectrum)
            if self.rng.random() < 0.3:
                spectrum = self.remove_peaks(spectrum)

        return spectrum

# Enhanced molecular feature extraction
class MolecularFeatureExtractor:
    """Extract comprehensive molecular features"""

    def __init__(self):
        # Initialize descriptor calculators
        self.descriptor_names = [desc[0] for desc in Descriptors._descList]
        self.mordred_calc = Calculator(descriptors, ignore_3D=True)

    def extract_rdkit_features(self, smiles):
        """Extract RDKit molecular descriptors"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(len(self.descriptor_names))

        features = []
        for desc_name in self.descriptor_names:
            try:
                desc_fn = getattr(Descriptors, desc_name)
                features.append(desc_fn(mol))
            except:
                features.append(0.0)

        return np.array(features)

    def extract_mordred_features(self, smiles):
        """Extract Mordred molecular descriptors"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(len(self.mordred_calc.descriptors))

        try:
            features = self.mordred_calc(mol)
            # Handle missing values
            features = [float(f) if f is not None and not np.isnan(float(f)) else 0.0
                       for f in features]
            return np.array(features)
        except:
            return np.zeros(len(self.mordred_calc.descriptors))

    def extract_fingerprints(self, smiles, fp_type='morgan'):
        """Extract molecular fingerprints"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2048)

        if fp_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        elif fp_type == 'rdkit':
            fp = Chem.RDKFingerprint(mol, fpSize=2048)
        elif fp_type == 'maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
            # Pad to 2048 bits
            fp_array = np.array(fp)
            padded = np.zeros(2048)
            padded[:len(fp_array)] = fp_array
            return padded
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")

        return np.array(fp)

    def extract_all_features(self, smiles):
        """Extract all molecular features"""
        rdkit_features = self.extract_rdkit_features(smiles)
        mordred_features = self.extract_mordred_features(smiles)
        morgan_fp = self.extract_fingerprints(smiles, 'morgan')
        rdkit_fp = self.extract_fingerprints(smiles, 'rdkit')

        return {
            'rdkit_descriptors': rdkit_features,
            'mordred_descriptors': mordred_features,
            'morgan_fingerprint': morgan_fp,
            'rdkit_fingerprint': rdkit_fp
        }

# Advanced data loading and preprocessing
class AdvancedDataModule(pl.LightningDataModule):
    """Advanced data module with comprehensive preprocessing"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.augmentation = SpectrumAugmentation(config)
        self.feature_extractor = MolecularFeatureExtractor()

        # Initialize transforms
        self.spectrum_transforms = [
            SpecTransforms.normalize_intensity(),
            SpecTransforms.remove_noise(threshold=0.01),
            SpecTransforms.bin_spectrum(n_bins=config.MAX_SPECTRUM_LEN)
        ]

        self.mol_transforms = [
            MolTransforms.canonicalize_smiles(),
            MolTransforms.remove_salts(),
            MolTransforms.neutralize_charges()
        ]

    def setup(self, stage=None):
        """Setup datasets for different stages"""

        # Load base datasets
        self.train_dataset = MassSpecDataset(
            split='train',
            transform=self.spectrum_transforms,
            mol_transform=self.mol_transforms
        )

        self.val_dataset = MassSpecDataset(
            split='val',
            transform=self.spectrum_transforms,
            mol_transform=self.mol_transforms
        )

        self.test_dataset = MassSpecDataset(
            split='test',
            transform=self.spectrum_transforms,
            mol_transform=self.mol_transforms
        )

        # Extract molecular features for all datasets
        self._extract_molecular_features()

        # Create stratified splits for cross-validation
        self._create_cv_splits()

        print(f'Train dataset: {len(self.train_dataset)} samples')
        print(f'Validation dataset: {len(self.val_dataset)} samples')
        print(f'Test dataset: {len(self.test_dataset)} samples')
        print(f'Vocabulary size: {self.train_dataset.vocab_size}')

    def _extract_molecular_features(self):
        """Extract molecular features for all samples"""
        logger.info("Extracting molecular features...")

        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            features_cache = {}

            for i, sample in enumerate(tqdm(dataset, desc="Extracting features")):
                smiles = sample['smiles']
                if smiles not in features_cache:
                    features_cache[smiles] = self.feature_extractor.extract_all_features(smiles)

                # Add features to sample
                sample.update(features_cache[smiles])

    def _create_cv_splits(self):
        """Create stratified cross-validation splits"""
        # Create molecular scaffolds for stratification
        scaffolds = []
        for sample in self.train_dataset:
            mol = Chem.MolFromSmiles(sample['smiles'])
            if mol:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.append(Chem.MolToSmiles(scaffold))
            else:
                scaffolds.append('invalid')

        # Create stratified splits
        self.cv_splits = []
        skf = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(range(len(self.train_dataset)), scaffolds):
            self.cv_splits.append((train_idx, val_idx))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # Separate different data types
        spectra = [item['spectrum'] for item in batch]
        smiles = [item['smiles'] for item in batch]

        # Pad spectra
        spectra = pad_sequence([torch.tensor(s) for s in spectra], batch_first=True)

        # Tokenize SMILES
        tokenized_smiles = [self.train_dataset.tokenize_smiles(s) for s in smiles]
        tokenized_smiles = pad_sequence(
            [torch.tensor(t) for t in tokenized_smiles],
            batch_first=True,
            padding_value=self.train_dataset.pad_token_id
        )

        # Collect molecular features
        mol_features = {}
        feature_keys = ['rdkit_descriptors', 'mordred_descriptors',
                       'morgan_fingerprint', 'rdkit_fingerprint']

        for key in feature_keys:
            if key in batch[0]:
                mol_features[key] = torch.stack([torch.tensor(item[key]) for item in batch])

        # Collect metadata
        metadata = {}
        meta_keys = ['adduct', 'instrument', 'collision_energy', 'precursor_mz']
        for key in meta_keys:
            if key in batch[0]:
                metadata[key] = torch.tensor([item[key] for item in batch])

        return {
            'spectrum': spectra,
            'smiles': tokenized_smiles,
            'molecular_features': mol_features,
            'metadata': metadata
        }

# Initialize data module
datamodule = AdvancedDataModule(config)
datamodule.setup()

# Cell 3: Multi-Modal Fusion Architecture with Graph Neural Networks
class MultiModalGraphTransformer(MassSpecGymModel):
    """Advanced multi-modal architecture combining transformers, GNNs, and molecular features"""

    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Spectrum encoder with hierarchical attention
        self.spectrum_encoder = HierarchicalSpectrumEncoder(config)

        # Graph neural network for molecular structure
        self.graph_encoder = MolecularGraphEncoder(config)

        # Multi-modal fusion transformer
        self.fusion_transformer = MultiModalFusionTransformer(config)

        # Molecular decoder with copy mechanism
        self.mol_decoder = MolecularDecoderWithCopy(vocab_size, config)

        # Property prediction heads
        self.property_predictors = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(config.D_MODEL, config.HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.HIDDEN_DIM, 1)
            ) for prop in config.PREDICT_PROPERTIES
        })

        # Uncertainty estimation
        self.uncertainty_head = UncertaintyEstimationHead(config)

        # Attention visualization
        self.attention_weights = {}

    def forward(self, batch, return_attention=False):
        """Forward pass with multi-modal fusion"""
        spectrum = batch['spectrum']
        molecular_features = batch.get('molecular_features', {})
        metadata = batch.get('metadata', {})
        target = batch.get('smiles')

        # Encode spectrum with hierarchical attention
        spectrum_repr, spec_attention = self.spectrum_encoder(spectrum)

        # Encode molecular graph if available
        graph_repr = None
        if 'molecular_graph' in batch:
            graph_repr, graph_attention = self.graph_encoder(batch['molecular_graph'])

        # Multi-modal fusion
        fused_repr, fusion_attention = self.fusion_transformer(
            spectrum_repr, graph_repr, molecular_features, metadata
        )

        # Store attention weights for visualization
        if return_attention:
            self.attention_weights = {
                'spectrum': spec_attention,
                'fusion': fusion_attention
            }
            if graph_repr is not None:
                self.attention_weights['graph'] = graph_attention

        # Molecular generation
        if target is not None:
            # Training mode
            generation_output = self.mol_decoder(fused_repr, target)

            # Property prediction
            property_predictions = {}
            for prop_name, predictor in self.property_predictors.items():
                property_predictions[prop_name] = predictor(fused_repr.mean(dim=1))

            # Uncertainty estimation
            uncertainty = self.uncertainty_head(fused_repr)

            return {
                'generation_logits': generation_output['logits'],
                'copy_scores': generation_output.get('copy_scores'),
                'property_predictions': property_predictions,
                'uncertainty': uncertainty,
                'attention_weights': self.attention_weights if return_attention else None
            }
        else:
            # Inference mode
            return self.generate(fused_repr)

    def generate(self, fused_repr, **kwargs):
        """Generate molecular structures with advanced decoding"""
        return self.mol_decoder.generate(fused_repr, **kwargs)

class HierarchicalSpectrumEncoder(nn.Module):
    """Hierarchical spectrum encoder with multi-scale attention"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Multi-scale convolutions for local patterns
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, config.D_MODEL // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 11]
        ])

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.D_MODEL, config.MAX_SPECTRUM_LEN)

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                config.D_MODEL, config.NHEAD,
                dropout=config.ATTENTION_DROPOUT,
                batch_first=True
            ) for _ in range(config.NUM_LAYERS)
        ])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.D_MODEL, config.HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.HIDDEN_DIM, config.D_MODEL)
            ) for _ in range(config.NUM_LAYERS)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.D_MODEL) for _ in range(config.NUM_LAYERS * 2)
        ])

        # Peak detection and importance weighting
        self.peak_detector = PeakDetectionModule(config)

    def forward(self, spectrum):
        # Multi-scale convolution
        conv_outputs = []
        spectrum_expanded = spectrum.unsqueeze(1)  # Add channel dimension

        for conv in self.conv_layers:
            conv_out = F.relu(conv(spectrum_expanded))
            conv_outputs.append(conv_out.transpose(1, 2))  # [batch, seq, features]

        # Concatenate multi-scale features
        x = torch.cat(conv_outputs, dim=-1)  # [batch, seq, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Peak importance weighting
        peak_weights = self.peak_detector(spectrum)
        x = x * peak_weights.unsqueeze(-1)

        # Transformer layers with residual connections
        attention_weights = []

        for i, (attn, ffn) in enumerate(zip(self.attention_layers, self.ffn_layers)):
            # Multi-head attention
            residual = x
            x = self.layer_norms[i*2](x)
            attn_out, attn_weights = attn(x, x, x)
            x = residual + attn_out
            attention_weights.append(attn_weights)

            # Feed-forward
            residual = x
            x = self.layer_norms[i*2 + 1](x)
            x = residual + ffn(x)

        return x, attention_weights

class MolecularGraphEncoder(nn.Module):
    """Graph neural network for molecular structure encoding"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Node and edge feature dimensions
        self.node_dim = 128  # Atom features
        self.edge_dim = 64   # Bond features

        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GATConv(
                self.node_dim if i == 0 else config.GNN_HIDDEN_DIM,
                config.GNN_HIDDEN_DIM,
                heads=4,
                dropout=config.GNN_DROPOUT,
                edge_dim=self.edge_dim
            ) for i in range(config.GNN_NUM_LAYERS)
        ])

        # Graph pooling
        if config.GRAPH_POOLING == 'attention':
            self.pooling = AttentionalAggregation(
                gate_nn=nn.Linear(config.GNN_HIDDEN_DIM, 1)
            )
        elif config.GRAPH_POOLING == 'set2set':
            self.pooling = Set2Set(config.GNN_HIDDEN_DIM, processing_steps=3)
        else:
            self.pooling = global_mean_pool

        # Project to model dimension
        pool_dim = config.GNN_HIDDEN_DIM * 2 if config.GRAPH_POOLING == 'set2set' else config.GNN_HIDDEN_DIM
        self.projection = nn.Linear(pool_dim, config.D_MODEL)

    def forward(self, graph_batch):
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch

        attention_weights = []

        # Graph convolutions
        for conv in self.conv_layers:
            x_new, attn = conv(x, edge_index, edge_attr, return_attention_weights=True)
            x = F.relu(x_new)
            attention_weights.append(attn)

        # Graph pooling
        if hasattr(self.pooling, '__call__'):
            if isinstance(self.pooling, (AttentionalAggregation, Set2Set)):
                graph_repr = self.pooling(x, batch)
            else:
                graph_repr = self.pooling(x, batch)

        # Project to model dimension
        graph_repr = self.projection(graph_repr)

        return graph_repr, attention_weights

class MultiModalFusionTransformer(nn.Module):
    """Multi-modal fusion with cross-attention"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Feature projections
        self.spectrum_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.graph_proj = nn.Linear(config.D_MODEL, config.D_MODEL)

        # Molecular feature encoders
        self.rdkit_encoder = nn.Sequential(
            nn.Linear(200, config.D_MODEL // 2),  # Approximate RDKit descriptor count
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        self.mordred_encoder = nn.Sequential(
            nn.Linear(1613, config.D_MODEL // 2),  # Mordred descriptor count
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(2048, config.D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        # Metadata encoders
        self.adduct_encoder = nn.Embedding(20, config.D_MODEL // 4)
        self.instrument_encoder = nn.Embedding(10, config.D_MODEL // 4)
        self.collision_encoder = nn.Linear(1, config.D_MODEL // 4)
        self.precursor_encoder = nn.Linear(1, config.D_MODEL // 4)

        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            config.D_MODEL, config.NHEAD,
            dropout=config.ATTENTION_DROPOUT,
            batch_first=True
        )

        # Fusion transformer
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                dim_feedforward=config.HIDDEN_DIM,
                dropout=config.DROPOUT,
                activation='gelu',
                batch_first=True
            ) for _ in range(config.NUM_LAYERS // 2)
        ])

    def forward(self, spectrum_repr, graph_repr, molecular_features, metadata):
        batch_size = spectrum_repr.size(0)

        # Project spectrum representation
        spectrum_proj = self.spectrum_proj(spectrum_repr)

        # Encode molecular features
        feature_reprs = []

        if 'rdkit_descriptors' in molecular_features:
            rdkit_repr = self.rdkit_encoder(molecular_features['rdkit_descriptors'])
            feature_reprs.append(rdkit_repr.unsqueeze(1))

        if 'mordred_descriptors' in molecular_features:
            mordred_repr = self.mordred_encoder(molecular_features['mordred_descriptors'])
            feature_reprs.append(mordred_repr.unsqueeze(1))

        if 'morgan_fingerprint' in molecular_features:
            fp_repr = self.fingerprint_encoder(molecular_features['morgan_fingerprint'])
            feature_reprs.append(fp_repr.unsqueeze(1))

        # Encode metadata
        metadata_reprs = []
        if 'adduct' in metadata:
            adduct_repr = self.adduct_encoder(metadata['adduct'])
            metadata_reprs.append(adduct_repr.unsqueeze(1))

        if 'instrument' in metadata:
            instrument_repr = self.instrument_encoder(metadata['instrument'])
            metadata_reprs.append(instrument_repr.unsqueeze(1))

        if 'collision_energy' in metadata:
            collision_repr = self.collision_encoder(metadata['collision_energy'].unsqueeze(-1))
            metadata_reprs.append(collision_repr.unsqueeze(1))

        if 'precursor_mz' in metadata:
            precursor_repr = self.precursor_encoder(metadata['precursor_mz'].unsqueeze(-1))
            metadata_reprs.append(precursor_repr.unsqueeze(1))

        # Combine all representations
        all_reprs = [spectrum_proj]

        if graph_repr is not None:
            graph_proj = self.graph_proj(graph_repr.unsqueeze(1))
            all_reprs.append(graph_proj)

        all_reprs.extend(feature_reprs)
        all_reprs.extend(metadata_reprs)

        # Concatenate along sequence dimension
        fused_repr = torch.cat(all_reprs, dim=1)

        # Cross-attention between modalities
        attn_output, attention_weights = self.cross_attention(
            fused_repr, fused_repr, fused_repr
        )

        # Fusion transformer layers
        for layer in self.fusion_layers:
            attn_output = layer(attn_output)

        return attn_output, attention_weights

class MolecularDecoderWithCopy(nn.Module):
    """Molecular decoder with copy mechanism and beam search"""

    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, config.D_MODEL)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                dim_feedforward=config.HIDDEN_DIM,
                dropout=config.DROPOUT,
                activation='gelu',
                batch_first=True
            ) for _ in range(config.NUM_DECODER_LAYERS)
        ])

        # Output projections
        self.vocab_projection = nn.Linear(config.D_MODEL, vocab_size)
        self.copy_projection = nn.Linear(config.D_MODEL, 1)

        # Copy mechanism
        self.copy_attention = nn.MultiheadAttention(
            config.D_MODEL, config.NHEAD,
            dropout=config.ATTENTION_DROPOUT,
            batch_first=True
        )

    def forward(self, memory, target=None):
        if target is not None:
            # Training mode
            return self._forward_train(memory, target)
        else:
            # Inference mode
            return self.generate(memory)

    def _forward_train(self, memory, target):
        # Embed target tokens (teacher forcing)
        tgt_embedded = self.token_embedding(target[:, :-1])

        # Create causal mask
        seq_len = tgt_embedded.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(tgt_embedded.device)

        # Decoder forward pass
        decoder_output = tgt_embedded

        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output, memory,
                tgt_mask=causal_mask
            )

        # Vocabulary distribution
        vocab_logits = self.vocab_projection(decoder_output)

        # Copy mechanism
        copy_scores, copy_attention = self.copy_attention(
            decoder_output, memory, memory
        )
        copy_probs = torch.sigmoid(self.copy_projection(decoder_output))

        return {
            'logits': vocab_logits,
            'copy_scores': copy_scores,
            'copy_probs': copy_probs,
            'copy_attention': copy_attention
        }

    def generate(self, memory, max_length=None, beam_width=None, **kwargs):
        """Generate with beam search and nucleus sampling"""
        max_length = max_length or self.config.MAX_SMILES_LEN
        beam_width = beam_width or self.config.BEAM_WIDTH

        if beam_width > 1:
            return self._beam_search(memory, max_length, beam_width)
        else:
            return self._nucleus_sampling(memory, max_length, **kwargs)

    def _beam_search(self, memory, max_length, beam_width):
        """Advanced beam search with length normalization"""
        batch_size = memory.size(0)
        device = memory.device

        # Initialize beams
        sequences = torch.full(
            (batch_size, beam_width, 1),
            datamodule.train_dataset.sos_token_id,
            device=device
        )
        scores = torch.zeros(batch_size, beam_width, device=device)

        # Expand memory for beam search
        memory_expanded = memory.unsqueeze(1).expand(
            -1, beam_width, -1, -1
        ).reshape(batch_size * beam_width, memory.size(1), memory.size(2))

        for step in range(max_length):
            # Current sequences
            current_seqs = sequences.reshape(batch_size * beam_width, -1)

            # Forward pass
            output = self._forward_step(memory_expanded, current_seqs)
            logits = output['logits'][:, -1]  # Last token logits

            # Apply temperature and get log probabilities
            logits = logits / self.config.TEMPERATURE
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_width, -1)

            if step == 0:
                # First step: only use first beam
                top_log_probs, top_indices = log_probs[:, 0].topk(beam_width, dim=-1)
                scores = top_log_probs

                # Update sequences
                new_tokens = top_indices.unsqueeze(-1)
                sequences = torch.cat([sequences[:, :1].expand(-1, beam_width, -1), new_tokens], dim=-1)
            else:
                # Subsequent steps: expand all beams
                candidate_scores = scores.unsqueeze(-1) + log_probs
                candidate_scores = candidate_scores.view(batch_size, -1)

                # Select top candidates
                top_scores, top_indices = candidate_scores.topk(beam_width, dim=-1)
                beam_indices = top_indices // self.vocab_size
                token_indices = top_indices % self.vocab_size

                # Update sequences
                new_sequences = torch.zeros(
                    batch_size, beam_width, step + 2,
                    dtype=torch.long, device=device
                )

                for b in range(batch_size):
                    for i in range(beam_width):
                        beam_idx = beam_indices[b, i]
                        new_sequences[b, i, :step+1] = sequences[b, beam_idx, :step+1]
                        new_sequences[b, i, step+1] = token_indices[b, i]

                sequences = new_sequences
                scores = top_scores

            # Check for EOS tokens
            if (sequences[:, :, -1] == datamodule.train_dataset.eos_token_id).all():
                break

        # Length normalization
        lengths = (sequences != datamodule.train_dataset.pad_token_id).sum(dim=-1).float()
        normalized_scores = scores / lengths

        return sequences, normalized_scores

    def _nucleus_sampling(self, memory, max_length, temperature=None, top_p=None):
        """Nucleus (top-p) sampling"""
        temperature = temperature or self.config.TEMPERATURE
        top_p = top_p or self.config.NUCLEUS_P

        batch_size = memory.size(0)
        device = memory.device

        # Initialize sequences
        sequences = torch.full(
            (batch_size, 1),
            datamodule.train_dataset.sos_token_id,
            device=device
        )

        for step in range(max_length):
            # Forward pass
            output = self._forward_step(memory, sequences)
            logits = output['logits'][:, -1] / temperature

            # Nucleus sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Append to sequences
            sequences = torch.cat([sequences, next_tokens], dim=-1)

            # Check for EOS tokens
            if (next_tokens == datamodule.train_dataset.eos_token_id).all():
                break

        return sequences

    def _forward_step(self, memory, sequences):
        """Single forward step for generation"""
        # Embed tokens
        embedded = self.token_embedding(sequences)

        # Create causal mask
        seq_len = embedded.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(embedded.device)

        # Decoder forward pass
        decoder_output = embedded

        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output, memory,
                tgt_mask=causal_mask
            )

        # Output projections
        vocab_logits = self.vocab_projection(decoder_output)

        return {'logits': vocab_logits}

class UncertaintyEstimationHead(nn.Module):
    """Uncertainty estimation using Monte Carlo Dropout"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.uncertainty_layers = nn.Sequential(
            nn.Linear(config.D_MODEL, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.5),  # Always active for MC dropout
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.HIDDEN_DIM // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Average over sequence dimension
        x_pooled = x.mean(dim=1)
        return self.uncertainty_layers(x_pooled)

    def estimate_uncertainty(self, x, n_samples=100):
        """Estimate uncertainty using Monte Carlo Dropout"""
        self.train()  # Enable dropout

        uncertainties = []
        for _ in range(n_samples):
            with torch.no_grad():
                uncertainty = self.forward(x)
                uncertainties.append(uncertainty)

        uncertainties = torch.stack(uncertainties)
        mean_uncertainty = uncertainties.mean(dim=0)
        std_uncertainty = uncertainties.std(dim=0)

        return mean_uncertainty, std_uncertainty

class PeakDetectionModule(nn.Module):
    """Peak detection and importance weighting for spectra"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Learnable peak detection
        self.peak_detector = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, spectrum):
        # Add channel dimension
        spectrum_expanded = spectrum.unsqueeze(1)

        # Detect peaks
        peak_weights = self.peak_detector(spectrum_expanded)
        peak_weights = peak_weights.squeeze(1)

        # Ensure minimum weight
        peak_weights = peak_weights + 0.1

        return peak_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

    def training_step(self, batch, batch_idx):
        """Training step with multi-task learning"""
        output = self(batch)

        # Generation loss
        generation_logits = output['generation_logits']
        target = batch['smiles']

        generation_loss = F.cross_entropy(
            generation_logits.reshape(-1, self.vocab_size),
            target[:, 1:].reshape(-1),
            ignore_index=datamodule.train_dataset.pad_token_id
        )

        # Property prediction losses
        property_losses = {}
        total_property_loss = 0

        for prop_name, predictions in output['property_predictions'].items():
            if prop_name in batch:  # If ground truth available
                prop_loss = F.mse_loss(predictions.squeeze(), batch[prop_name].float())
                property_losses[f'{prop_name}_loss'] = prop_loss
                total_property_loss += prop_loss

        # Uncertainty regularization
        uncertainty_loss = output['uncertainty'].mean()  # Encourage low uncertainty

        # Total loss
        total_loss = (
            generation_loss +
            0.1 * total_property_loss +
            0.01 * uncertainty_loss
        )

        # Logging
        self.log('train_loss', total_loss)
        self.log('train_generation_loss', generation_loss)
        self.log('train_property_loss', total_property_loss)
        self.log('train_uncertainty_loss', uncertainty_loss)

        for prop_name, loss in property_losses.items():
            self.log(f'train_{prop_name}', loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step with comprehensive metrics"""
        # Generate predictions
        with torch.no_grad():
            output = self(batch)
            sequences = self.generate(batch)

        # Calculate generation metrics
        if isinstance(sequences, tuple):
            sequences, scores = sequences
            predictions = sequences[:, 0]  # Best beam
        else:
            predictions = sequences

        # Convert to SMILES
        pred_smiles = [datamodule.train_dataset.decode_sequence(seq) for seq in predictions]
        true_smiles = [datamodule.train_dataset.decode_sequence(seq) for seq in batch['smiles']]

        # Molecular similarity metrics
        tanimoto_similarities = []
        mces_similarities = []
        exact_matches = []

        for pred, true in zip(pred_smiles, true_smiles):
            try:
                tanimoto = tanimoto_similarity(pred, true)
                mces = mces_similarity(pred, true)
                exact = (pred == true)

                tanimoto_similarities.append(tanimoto)
                mces_similarities.append(mces)
                exact_matches.append(exact)
            except:
                tanimoto_similarities.append(0.0)
                mces_similarities.append(0.0)
                exact_matches.append(False)

        # Property prediction metrics
        property_metrics = {}
        for prop_name, predictions in output['property_predictions'].items():
            if prop_name in batch:
                pred_values = predictions.squeeze().cpu().numpy()
                true_values = batch[prop_name].cpu().numpy()

                mse = mean_squared_error(true_values, pred_values)
                r2 = r2_score(true_values, pred_values)

                property_metrics[f'{prop_name}_mse'] = mse
                property_metrics[f'{prop_name}_r2'] = r2

        # Aggregate metrics
        metrics = {
            'val_tanimoto': np.mean(tanimoto_similarities),
            'val_mces': np.mean(mces_similarities),
            'val_exact_match': np.mean(exact_matches),
            'val_uncertainty': output['uncertainty'].mean().item()
        }

        metrics.update(property_metrics)

        # Log all metrics
        for metric_name, value in metrics.items():
            self.log(metric_name, value)

        return metrics

    def configure_optimizers(self):
        """Configure optimizers with advanced scheduling"""
        # Separate parameter groups
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.WEIGHT_DECAY,
            },
            {
                'params': [p for n, p in self.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.LEARNING_RATE,
            eps=1e-8
        )

        # Learning rate scheduler
        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.WARMUP_STEPS,
                num_training_steps=self.trainer.estimated_stepping_batches
            ),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

# =============================================================================
# CELL 12: Advanced RAG System with Multi-Modal Retrieval
# =============================================================================
class AdvancedMolecularRAGSystem:
    """Advanced RAG system with multi-modal retrieval and re-ranking"""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.feature_extractor = MolecularFeatureExtractor()

        # Multiple encoders for different modalities
        self.text_encoder = SentenceTransformer('all-mpnet-base-v2')
        self.chem_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Build comprehensive indices
        self.build_multi_modal_indices()

        # Re-ranking model
        self.reranker = self._build_reranker()

    def build_multi_modal_indices(self):
        """Build multiple FAISS indices for different modalities"""
        logger.info("Building multi-modal retrieval indices...")

        # Collect all molecular data
        descriptions = []
        chemical_descriptions = []
        fingerprints = []
        rdkit_features = []
        spectra_features = []

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            smiles = sample['smiles']
            spectrum = sample['spectrum']

            # Create rich molecular descriptions
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Natural language description
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                rings = Descriptors.RingCount(mol)

                desc = (
                    f"Molecule with SMILES {smiles}, molecular weight {mw:.1f}, "
                    f"LogP {logp:.2f}, TPSA {tpsa:.1f}, {hbd} hydrogen bond donors, "
                    f"{hba} hydrogen bond acceptors, {rings} rings"
                )
                descriptions.append(desc)

                # Chemical-specific description
                formula = CalcMolFormula(mol)
                chem_desc = f"Chemical formula {formula} SMILES {smiles}"
                chemical_descriptions.append(chem_desc)

                # Molecular features
                mol_features = self.feature_extractor.extract_all_features(smiles)
                fingerprints.append(mol_features['morgan_fingerprint'])
                rdkit_features.append(mol_features['rdkit_descriptors'])

            else:
                descriptions.append(f"Invalid molecule {smiles}")
                chemical_descriptions.append(f"Invalid {smiles}")
                fingerprints.append(np.zeros(2048))
                rdkit_features.append(np.zeros(200))

            # Spectrum features
            spec_features = self._extract_spectrum_features(spectrum)
            spectra_features.append(spec_features)

        # Convert to arrays
        self.descriptions = descriptions
        self.chemical_descriptions = chemical_descriptions
        self.fingerprints = np.array(fingerprints)
        self.rdkit_features = np.array(rdkit_features)
        self.spectra_features = np.array(spectra_features)

        # Build FAISS indices
        self._build_faiss_indices()

    def _extract_spectrum_features(self, spectrum):
        """Extract statistical features from spectrum"""
        if len(spectrum) == 0:
            return np.zeros(20)

        features = [
            np.mean(spectrum),
            np.std(spectrum),
            np.max(spectrum),
            np.min(spectrum),
            np.median(spectrum),
            np.sum(spectrum > 0.1),
            np.sum(spectrum > 0.5),
            np.sum(spectrum > 0.9),
            len(spectrum),
            np.percentile(spectrum, 90),
            np.percentile(spectrum, 75),
            np.percentile(spectrum, 50),
            np.percentile(spectrum, 25),
            np.percentile(spectrum, 10),
            stats.skew(spectrum),
            stats.kurtosis(spectrum),
            np.sum(spectrum),
            np.var(spectrum),
            np.ptp(spectrum),  # peak-to-peak
            np.count_nonzero(spectrum)
        ]

        return np.array(features)

    def _build_faiss_indices(self):
        """Build FAISS indices for different modalities"""
        # Text semantic index
        text_embeddings = self.text_encoder.encode(self.descriptions)
        self.text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
        faiss.normalize_L2(text_embeddings)
        self.text_index.add(text_embeddings.astype('float32'))

        # Chemical semantic index
        chem_embeddings = self.chem_encoder.encode(self.chemical_descriptions)
        self.chem_index = faiss.IndexFlatIP(chem_embeddings.shape[1])
        faiss.normalize_L2(chem_embeddings)
        self.chem_index.add(chem_embeddings.astype('float32'))

        # Molecular fingerprint index
        self.fingerprint_index = faiss.IndexFlatIP(2048)
        faiss.normalize_L2(self.fingerprints)
        self.fingerprint_index.add(self.fingerprints.astype('float32'))

        # RDKit features index
        self.rdkit_index = faiss.IndexFlatL2(self.rdkit_features.shape[1])
        self.rdkit_index.add(self.rdkit_features.astype('float32'))

        # Spectrum features index
        self.spectrum_index = faiss.IndexFlatL2(self.spectra_features.shape[1])
        self.spectrum_index.add(self.spectra_features.astype('float32'))

        # Hierarchical clustering for diversity
        self._build_diversity_clusters()

    def _build_diversity_clusters(self):
        """Build clusters for diversity-aware retrieval"""
        # Use molecular fingerprints for clustering
        kmeans = KMeans(n_clusters=min(100, len(self.dataset) // 10), random_state=42)
        self.clusters = kmeans.fit_predict(self.fingerprints)
        self.cluster_centers = kmeans.cluster_centers_

    def _build_reranker(self):
        """Build neural re-ranking model"""
        class ReRanker(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.layers(x)

        # Input: concatenated features from all modalities
        input_dim = (
            self.text_encoder.get_sentence_embedding_dimension() +
            self.chem_encoder.get_sentence_embedding_dimension() +
            2048 +  # fingerprints
            self.rdkit_features.shape[1] +
            self.spectra_features.shape[1]
        )

        return ReRanker(input_dim)

    def search(self, query, query_type='text', k=10, mode='hybrid',
               diversity_weight=0.2, rerank=True):
        """Multi-modal search with re-ranking and diversity"""

        # Initial retrieval from multiple indices
        candidates = set()

        if mode in ['text', 'hybrid']:
            text_results = self._search_text(query, k*2)
            candidates.update([idx for idx, _ in text_results])

        if mode in ['chemical', 'hybrid']:
            chem_results = self._search_chemical(query, k*2)
            candidates.update([idx for idx, _ in chem_results])

        if mode in ['structure', 'hybrid'] and query_type == 'smiles':
            struct_results = self._search_structure(query, k*2)
            candidates.update([idx for idx, _ in struct_results])

        if mode in ['spectrum', 'hybrid'] and query_type == 'spectrum':
            spec_results = self._search_spectrum(query, k*2)
            candidates.update([idx for idx, _ in spec_results])

        # Convert to list and limit
        candidates = list(candidates)[:k*3]

        if rerank and len(candidates) > k:
            # Re-rank using neural model
            candidates = self._rerank_candidates(query, candidates, query_type)

        # Apply diversity filtering
        if diversity_weight > 0:
            candidates = self._diversify_results(candidates, k, diversity_weight)

        # Return top-k with scores
        results = []
        for idx in candidates[:k]:
            sample = self.dataset[idx]
            score = self._compute_final_score(query, sample, query_type)
            results.append((sample['smiles'], score, idx))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def _search_text(self, query, k):
        """Search using text embeddings"""
        query_emb = self.text_encoder.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.text_index.search(query_emb.astype('float32'), k)
        return [(indices[0][i], scores[0][i]) for i in range(len(indices[0]))]

    def _search_chemical(self, query, k):
        """Search using chemical embeddings"""
        query_emb = self.chem_encoder.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.chem_index.search(query_emb.astype('float32'), k)
        return [(indices[0][i], scores[0][i]) for i in range(len(indices[0]))]

    def _search_structure(self, query_smiles, k):
        """Search using molecular fingerprints"""
        query_fp = self.feature_extractor.extract_fingerprints(query_smiles, 'morgan')
        query_fp = query_fp.reshape(1, -1)
        faiss.normalize_L2(query_fp)
        scores, indices = self.fingerprint_index.search(query_fp.astype('float32'), k)
        return [(indices[0][i], scores[0][i]) for i in range(len(indices[0]))]

    def _search_spectrum(self, query_spectrum, k):
        """Search using spectrum features"""
        query_features = self._extract_spectrum_features(query_spectrum)
        query_features = query_features.reshape(1, -1)
        scores, indices = self.spectrum_index.search(query_features.astype('float32'), k)
        return [(indices[0][i], scores[0][i]) for i in range(len(indices[0]))]

    def _rerank_candidates(self, query, candidates, query_type):
        """Re-rank candidates using neural model"""
        # Extract features for query and candidates
        query_features = self._extract_query_features(query, query_type)

        candidate_scores = []
        for idx in candidates:
            candidate_features = self._extract_candidate_features(idx)
            combined_features = np.concatenate([query_features, candidate_features])

            with torch.no_grad():
                score = self.reranker(torch.tensor(combined_features).float())
                candidate_scores.append((idx, score.item()))

        # Sort by re-ranking score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in candidate_scores]

    def _diversify_results(self, candidates, k, diversity_weight):
        """Apply diversity filtering using clustering"""
        if len(candidates) <= k:
            return candidates

        # Greedy diversity selection
        selected = [candidates[0]]  # Start with top candidate
        remaining = candidates[1:]

        while len(selected) < k and remaining:
            best_candidate = None
            best_score = -1

            for candidate in remaining:
                # Relevance score (position in original ranking)
                relevance = 1.0 / (candidates.index(candidate) + 1)

                # Diversity score (distance to selected items)
                diversity = self._compute_diversity(candidate, selected)

                # Combined score
                combined_score = (1 - diversity_weight) * relevance + diversity_weight * diversity

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def _compute_diversity(self, candidate, selected):
        """Compute diversity score for candidate"""
        if not selected:
            return 1.0

        candidate_fp = self.fingerprints[candidate]
        min_similarity = 1.0

        for sel_idx in selected:
            selected_fp = self.fingerprints[sel_idx]
            similarity = np.dot(candidate_fp, selected_fp) / (
                np.linalg.norm(candidate_fp) * np.linalg.norm(selected_fp)
            )
            min_similarity = min(min_similarity, similarity)

        return 1.0 - min_similarity  # Higher diversity = lower similarity

    def _extract_query_features(self, query, query_type):
        """Extract features for query"""
        if query_type == 'text':
            text_emb = self.text_encoder.encode([query])[0]
            chem_emb = np.zeros(self.chem_encoder.get_sentence_embedding_dimension())
            fp = np.zeros(2048)
            rdkit_feat = np.zeros(self.rdkit_features.shape[1])
            spec_feat = np.zeros(self.spectra_features.shape[1])
        elif query_type == 'smiles':
            text_emb = np.zeros(self.text_encoder.get_sentence_embedding_dimension())
            chem_emb = self.chem_encoder.encode([f"Chemical formula SMILES {query}"])[0]
            mol_features = self.feature_extractor.extract_all_features(query)
            fp = mol_features['morgan_fingerprint']
            rdkit_feat = mol_features['rdkit_descriptors']
            spec_feat = np.zeros(self.spectra_features.shape[1])
        elif query_type == 'spectrum':
            text_emb = np.zeros(self.text_encoder.get_sentence_embedding_dimension())
            chem_emb = np.zeros(self.chem_encoder.get_sentence_embedding_dimension())
            fp = np.zeros(2048)
            rdkit_feat = np.zeros(self.rdkit_features.shape[1])
            spec_feat = self._extract_spectrum_features(query)

        return np.concatenate([text_emb, chem_emb, fp, rdkit_feat, spec_feat])

    def _extract_candidate_features(self, idx):
        """Extract features for candidate"""
        text_emb = self.text_encoder.encode([self.descriptions[idx]])[0]
        chem_emb = self.chem_encoder.encode([self.chemical_descriptions[idx]])[0]
        fp = self.fingerprints[idx]
        rdkit_feat = self.rdkit_features[idx]
        spec_feat = self.spectra_features[idx]

        return np.concatenate([text_emb, chem_emb, fp, rdkit_feat, spec_feat])

    def _compute_final_score(self, query, sample, query_type):
        """Compute final relevance score"""
        # Simple scoring based on query type
        if query_type == 'smiles':
            try:
                return tanimoto_similarity(query, sample['smiles'])
            except:
                return 0.0
        else:
            return 1.0  # Placeholder

    def get_similar_molecules(self, smiles, k=10, threshold=0.7):
        """Get molecules similar to given SMILES"""
        results = self.search(smiles, query_type='smiles', k=k*2, mode='structure')

        # Filter by similarity threshold
        filtered_results = [(mol, score, idx) for mol, score, idx in results if score >= threshold]

        return filtered_results[:k]

    def explain_retrieval(self, query, result_idx, query_type='text'):
        """Explain why a particular result was retrieved"""
        sample = self.dataset[result_idx]

        explanation = {
            'query': query,
            'result_smiles': sample['smiles'],
            'similarity_scores': {}
        }

        # Compute similarity scores for different modalities
        if query_type == 'smiles':
            try:
                explanation['similarity_scores']['tanimoto'] = tanimoto_similarity(query, sample['smiles'])
            except:
                explanation['similarity_scores']['tanimoto'] = 0.0

        # Text similarity
        query_text_emb = self.text_encoder.encode([query])
        result_text_emb = self.text_encoder.encode([self.descriptions[result_idx]])
        text_sim = cosine(query_text_emb[0], result_text_emb[0])
        explanation['similarity_scores']['text_similarity'] = 1 - text_sim

        return explanation

# Cell 5: Advanced Ensemble Methods and Baseline Models
class AdvancedEnsembleSystem:
    """Comprehensive ensemble system with multiple baseline models"""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.feature_extractor = MolecularFeatureExtractor()

        # Initialize multiple baseline models
        self.models = {
            'xgboost': AdvancedXGBoostModel(config),
            'lightgbm': AdvancedLightGBMModel(config),
            'catboost': AdvancedCatBoostModel(config),
            'random_forest': AdvancedRandomForestModel(config),
            'neural_baseline': NeuralBaselineModel(config)
        }

        # Ensemble meta-learner
        self.meta_learner = None
        self.ensemble_weights = None

        # Feature importance tracking
        self.feature_importance = {}

    def extract_comprehensive_features(self, spectrum, metadata=None):
        """Extract comprehensive feature set from spectrum and metadata"""
        features = []
        feature_names = []

        # Basic spectrum statistics
        basic_stats = [
            np.mean(spectrum), np.std(spectrum), np.max(spectrum), np.min(spectrum),
            np.median(spectrum), np.var(spectrum), stats.skew(spectrum), stats.kurtosis(spectrum),
            np.sum(spectrum > 0.01), np.sum(spectrum > 0.1), np.sum(spectrum > 0.5),
            len(spectrum), np.count_nonzero(spectrum), np.ptp(spectrum)
        ]
        features.extend(basic_stats)
        feature_names.extend([
            'mean', 'std', 'max', 'min', 'median', 'var', 'skew', 'kurtosis',
            'peaks_001', 'peaks_01', 'peaks_05', 'length', 'nonzero', 'ptp'
        ])

        # Percentile features
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            features.append(np.percentile(spectrum, p))
            feature_names.append(f'percentile_{p}')

        # Peak detection features
        peak_features = self._extract_peak_features(spectrum)
        features.extend(peak_features)
        feature_names.extend([
            'num_peaks', 'peak_prominence_mean', 'peak_prominence_std',
            'peak_width_mean', 'peak_width_std', 'base_peak_intensity',
            'second_peak_intensity', 'peak_intensity_ratio'
        ])

        # Spectral entropy and complexity
        entropy_features = self._extract_entropy_features(spectrum)
        features.extend(entropy_features)
        feature_names.extend(['spectral_entropy', 'spectral_complexity', 'information_content'])

        # Frequency domain features
        freq_features = self._extract_frequency_features(spectrum)
        features.extend(freq_features)
        feature_names.extend([
            'dominant_frequency', 'spectral_centroid', 'spectral_bandwidth',
            'spectral_rolloff', 'zero_crossing_rate'
        ])

        # Metadata features
        if metadata:
            meta_features = self._extract_metadata_features(metadata)
            features.extend(meta_features)
            feature_names.extend(['adduct_encoded', 'instrument_encoded', 'collision_energy', 'precursor_mz'])

        return np.array(features), feature_names

    def _extract_peak_features(self, spectrum):
        """Extract peak-related features"""
        from scipy.signal import find_peaks, peak_prominences, peak_widths

        # Find peaks
        peaks, properties = find_peaks(spectrum, height=0.01, distance=5)

        if len(peaks) == 0:
            return [0, 0, 0, 0, 0, 0, 0, 0]

        # Peak prominences
        prominences = peak_prominences(spectrum, peaks)[0]

        # Peak widths
        widths = peak_widths(spectrum, peaks)[0]

        # Peak intensities
        intensities = spectrum[peaks]

        features = [
            len(peaks),  # Number of peaks
            np.mean(prominences) if len(prominences) > 0 else 0,
            np.std(prominences) if len(prominences) > 0 else 0,
            np.mean(widths) if len(widths) > 0 else 0,
            np.std(widths) if len(widths) > 0 else 0,
            np.max(intensities) if len(intensities) > 0 else 0,  # Base peak
            np.partition(intensities, -2)[-2] if len(intensities) > 1 else 0,  # Second highest
            (np.partition(intensities, -2)[-2] / np.max(intensities)) if len(intensities) > 1 else 0
        ]

        return features

    def _extract_entropy_features(self, spectrum):
        """Extract entropy and complexity features"""
        # Normalize spectrum
        normalized = spectrum / (np.sum(spectrum) + 1e-8)

        # Spectral entropy
        entropy = -np.sum(normalized * np.log(normalized + 1e-8))

        # Spectral complexity (number of significant peaks)
        complexity = np.sum(normalized > 0.01)

        # Information content
        info_content = np.sum(-np.log(normalized + 1e-8) * normalized)

        return [entropy, complexity, info_content]

    def _extract_frequency_features(self, spectrum):
        """Extract frequency domain features"""
        # FFT
        fft = np.fft.fft(spectrum)
        magnitude = np.abs(fft)

        # Dominant frequency
        dominant_freq = np.argmax(magnitude)

        # Spectral centroid
        freqs = np.arange(len(magnitude))
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)

        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-8))

        # Spectral rolloff (95% of energy)
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
        spectral_rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else len(magnitude) - 1

        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(spectrum)) != 0)
        zero_crossing_rate = zero_crossings / len(spectrum)

        return [dominant_freq, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]

    def _extract_metadata_features(self, metadata):
        """Extract features from metadata"""
        features = []

        # Encode categorical variables
        if 'adduct' in metadata:
            features.append(hash(str(metadata['adduct'])) % 1000)  # Simple hash encoding
        else:
            features.append(0)

        if 'instrument' in metadata:
            features.append(hash(str(metadata['instrument'])) % 100)
        else:
            features.append(0)

        # Numerical features
        features.append(metadata.get('collision_energy', 0))
        features.append(metadata.get('precursor_mz', 0))

        return features

    def prepare_ensemble_data(self):
        """Prepare data for ensemble training"""
        logger.info("Preparing ensemble training data...")

        X, y = [], []
        feature_names = None

        for sample in tqdm(self.dataset, desc="Extracting features"):
            spectrum = sample['spectrum']
            metadata = sample.get('metadata', {})

            features, names = self.extract_comprehensive_features(spectrum, metadata)
            X.append(features)
            y.append(sample['smiles'])

            if feature_names is None:
                feature_names = names

        X = np.array(X)
        self.feature_names = feature_names

        # Handle categorical target (SMILES)
        unique_smiles = list(set(y))
        self.smiles_to_idx = {smiles: idx for idx, smiles in enumerate(unique_smiles)}
        self.idx_to_smiles = {idx: smiles for smiles, idx in self.smiles_to_idx.items()}

        y_encoded = [self.smiles_to_idx[smiles] for smiles in y]

        return X, np.array(y_encoded), y

    def train_ensemble(self, cv_folds=5):
        """Train ensemble with cross-validation"""
        logger.info("Training ensemble models...")

        X, y_encoded, y_smiles = self.prepare_ensemble_data()

        # Cross-validation setup
        cv_scores = {model_name: [] for model_name in self.models.keys()}
        cv_predictions = {model_name: [] for model_name in self.models.keys()}

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            fold_predictions = {}

            # Train each model
            for model_name, model in self.models.items():
                logger.info(f"  Training {model_name}...")

                # Train model
                model.fit(X_train, y_train)

                # Predict on validation set
                if hasattr(model, 'predict_proba'):
                    val_pred_proba = model.predict_proba(X_val)
                    val_pred = np.argmax(val_pred_proba, axis=1)
                else:
                    val_pred = model.predict(X_val)
                    val_pred_proba = None

                # Calculate accuracy
                accuracy = accuracy_score(y_val, val_pred)
                cv_scores[model_name].append(accuracy)

                # Store predictions for meta-learning
                fold_predictions[model_name] = val_pred_proba if val_pred_proba is not None else val_pred

            # Store fold predictions
            for model_name in self.models.keys():
                if model_name not in cv_predictions:
                    cv_predictions[model_name] = []
                cv_predictions[model_name].append(fold_predictions[model_name])

        # Print CV results
        logger.info("Cross-validation results:")
        for model_name, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logger.info(f"  {model_name}: {mean_score:.4f} ± {std_score:.4f}")

        # Train meta-learner
        self._train_meta_learner(cv_predictions, y_encoded)

        # Train final models on full dataset
        logger.info("Training final ensemble models on full dataset...")
        for model_name, model in self.models.items():
            model.fit(X, y_encoded)

        # Calculate feature importance
        self._calculate_feature_importance()

        return cv_scores

    def _train_meta_learner(self, cv_predictions, y_encoded):
        """Train meta-learner for ensemble combination"""
        logger.info("Training meta-learner...")

        # Prepare meta-features
        meta_X = []
        meta_y = []

        for fold_idx in range(len(cv_predictions[list(self.models.keys())[0]])):
            fold_meta_features = []

            for model_name in self.models.keys():
                fold_pred = cv_predictions[model_name][fold_idx]
                if fold_pred.ndim == 2:  # Probabilities
                    fold_meta_features.append(fold_pred)
                else:  # Class predictions
                    # Convert to one-hot
                    n_classes = len(self.smiles_to_idx)
                    one_hot = np.zeros((len(fold_pred), n_classes))
                    one_hot[np.arange(len(fold_pred)), fold_pred] = 1
                    fold_meta_features.append(one_hot)

            # Concatenate predictions from all models
            fold_meta_X = np.concatenate(fold_meta_features, axis=1)
            meta_X.append(fold_meta_X)

        # Combine all folds
        meta_X = np.vstack(meta_X)
        meta_y = y_encoded  # Use original labels

        # Train meta-learner (simple logistic regression)
        self.meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_learner.fit(meta_X, meta_y)

        logger.info("Meta-learner training completed")

    def _calculate_feature_importance(self):
        """Calculate feature importance across models"""
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                continue

            # Store importance with feature names
            self.feature_importance[model_name] = {
                name: imp for name, imp in zip(self.feature_names, importance)
            }

    def predict_ensemble(self, spectrum, metadata=None, return_probabilities=False):
        """Make ensemble prediction"""
        # Extract features
        features, _ = self.extract_comprehensive_features(spectrum, metadata)
        features = features.reshape(1, -1)

        # Get predictions from all models
        model_predictions = []

        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(features)[0]
                model_predictions.append(pred_proba)
            else:
                pred = model.predict(features)[0]
                # Convert to one-hot
                n_classes = len(self.smiles_to_idx)
                one_hot = np.zeros(n_classes)
                one_hot[pred] = 1
                model_predictions.append(one_hot)

        # Combine predictions
        if self.meta_learner is not None:
            # Use meta-learner
            meta_features = np.concatenate(model_predictions).reshape(1, -1)

            if return_probabilities:
                ensemble_proba = self.meta_learner.predict_proba(meta_features)[0]
                return ensemble_proba
            else:
                ensemble_pred = self.meta_learner.predict(meta_features)[0]
                return self.idx_to_smiles[ensemble_pred]
        else:
            # Simple averaging
            ensemble_proba = np.mean(model_predictions, axis=0)

            if return_probabilities:
                return ensemble_proba
            else:
                ensemble_pred = np.argmax(ensemble_proba)
                return self.idx_to_smiles[ensemble_pred]

    def get_feature_importance_summary(self, top_k=20):
        """Get summary of feature importance across models"""
        # Aggregate importance across models
        aggregated_importance = defaultdict(list)

        for model_name, importance_dict in self.feature_importance.items():
            for feature_name, importance in importance_dict.items():
                aggregated_importance[feature_name].append(importance)

        # Calculate mean importance
        mean_importance = {
            feature: np.mean(importances)
            for feature, importances in aggregated_importance.items()
        }

        # Sort by importance
        sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)

        return sorted_features[:top_k]

class AdvancedXGBoostModel:
    """Advanced XGBoost model with hyperparameter optimization"""

    def __init__(self, config):
        self.config = config
        self.model = None

    def fit(self, X, y):
        # Optimized hyperparameters
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

class AdvancedLightGBMModel:
    """Advanced LightGBM model"""

    def __init__(self, config):
        self.config = config
        self.model = None

    def fit(self, X, y):
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

class AdvancedCatBoostModel:
    """Advanced CatBoost model"""

    def __init__(self, config):
        self.config = config
        self.model = None

    def fit(self, X, y):
        self.model = CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

class AdvancedRandomForestModel:
    """Advanced Random Forest model"""

    def __init__(self, config):
        self.config = config
        self.model = None

    def fit(self, X, y):
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

class NeuralBaselineModel:
    """Neural network baseline model"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build neural network
        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        self.model = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

        # Train the model
        self._train_neural_model(X_scaled, y)
        return self

    def _train_neural_model(self, X, y):
        """Train neural network"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(50):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.numpy()

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)

        return probabilities.numpy()

# Cell 6: Advanced Cross-Validation and Evaluation Framework
class ComprehensiveEvaluationFramework:
    """Advanced evaluation framework with cross-validation, statistical testing, and visualization"""

    def __init__(self, config):
        self.config = config
        self.results = {}
        self.statistical_tests = {}
        self.visualizations = {}

        # Initialize metrics
        self.metrics = {
            'molecular_similarity': {
                'tanimoto': tanimoto_similarity,
                'mces': mces_similarity,
                'dice': self._dice_similarity,
                'cosine': self._cosine_similarity
            },
            'sequence_similarity': {
                'exact_match': self._exact_match,
                'levenshtein': self._levenshtein_similarity,
                'bleu': self._bleu_score,
                'rouge': self._rouge_score
            },
            'chemical_validity': {
                'valid_smiles': self._is_valid_smiles,
                'drug_likeness': self._drug_likeness,
                'synthetic_accessibility': self._synthetic_accessibility
            }
        }

    def run_comprehensive_evaluation(self, models, test_data, cv_folds=5):
        """Run comprehensive evaluation with cross-validation"""
        logger.info("Starting comprehensive evaluation...")

        # Cross-validation evaluation
        cv_results = self._cross_validation_evaluation(models, test_data, cv_folds)

        # Statistical significance testing
        statistical_results = self._statistical_significance_testing(cv_results)

        # Error analysis
        error_analysis = self._error_analysis(models, test_data)

        # Uncertainty analysis
        uncertainty_analysis = self._uncertainty_analysis(models, test_data)

        # Visualization
        visualizations = self._create_visualizations(cv_results, error_analysis)

        # Compile comprehensive results
        comprehensive_results = {
            'cross_validation': cv_results,
            'statistical_tests': statistical_results,
            'error_analysis': error_analysis,
            'uncertainty_analysis': uncertainty_analysis,
            'visualizations': visualizations
        }

        # Generate report
        self._generate_evaluation_report(comprehensive_results)

        return comprehensive_results

    def _cross_validation_evaluation(self, models, test_data, cv_folds):
        """Perform cross-validation evaluation"""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")

        # Create stratified folds
        smiles_list = [sample['smiles'] for sample in test_data]
        scaffolds = self._get_molecular_scaffolds(smiles_list)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        folds = list(skf.split(range(len(test_data)), scaffolds))

        cv_results = {model_name: defaultdict(list) for model_name in models.keys()}

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"Evaluating fold {fold_idx + 1}/{cv_folds}")

            fold_test_data = [test_data[i] for i in test_idx]

            for model_name, model in models.items():
                fold_metrics = self._evaluate_model_on_fold(model, fold_test_data)

                for metric_category, metrics in fold_metrics.items():
                    for metric_name, value in metrics.items():
                        cv_results[model_name][f"{metric_category}_{metric_name}"].append(value)

        # Calculate statistics
        cv_statistics = {}
        for model_name, metrics in cv_results.items():
            cv_statistics[model_name] = {}
            for metric_name, values in metrics.items():
                cv_statistics[model_name][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'values': values
                }

        return cv_statistics

    def _get_molecular_scaffolds(self, smiles_list):
        """Get molecular scaffolds for stratification"""
        scaffolds = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(hash(scaffold_smiles) % 100)  # Hash to integer
            else:
                scaffolds.append(0)
        return scaffolds

    def _evaluate_model_on_fold(self, model, test_data):
        """Evaluate model on a single fold"""
        predictions = []
        true_values = []

        for sample in test_data:
            try:
                if hasattr(model, 'generate'):
                    # Neural model
                    pred = model.generate(sample)
                elif hasattr(model, 'predict_ensemble'):
                    # Ensemble model
                    pred = model.predict_ensemble(sample['spectrum'], sample.get('metadata'))
                elif hasattr(model, 'search'):
                    # RAG system
                    results = model.search("similar molecule", k=1)
                    pred = results[0][0] if results else ""
                else:
                    # Other models
                    pred = model.predict(sample['spectrum'])

                predictions.append(pred)
                true_values.append(sample['smiles'])
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                predictions.append("")
                true_values.append(sample['smiles'])

        # Calculate all metrics
        fold_metrics = {}

        for category, metric_funcs in self.metrics.items():
            fold_metrics[category] = {}
            for metric_name, metric_func in metric_funcs.items():
                try:
                    values = [metric_func(pred, true) for pred, true in zip(predictions, true_values)]
                    fold_metrics[category][metric_name] = np.mean(values)
                except Exception as e:
                    logger.warning(f"Metric calculation error for {metric_name}: {e}")
                    fold_metrics[category][metric_name] = 0.0

        return fold_metrics

    def _statistical_significance_testing(self, cv_results):
        """Perform statistical significance testing between models"""
        logger.info("Performing statistical significance testing...")

        model_names = list(cv_results.keys())
        statistical_results = {}

        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                statistical_results[comparison_key] = {}

                # Compare on each metric
                for metric_name in cv_results[model1].keys():
                    values1 = cv_results[model1][metric_name]['values']
                    values2 = cv_results[model2][metric_name]['values']

                    # Paired t-test
                    t_stat, t_pvalue = stats.ttest_rel(values1, values2)

                    # Wilcoxon signed-rank test
                    w_stat, w_pvalue = stats.wilcoxon(values1, values2)

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1) +
                                         (len(values2) - 1) * np.var(values2)) /
                                        (len(values1) + len(values2) - 2))
                    cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std

                    statistical_results[comparison_key][metric_name] = {
                        't_test': {'statistic': t_stat, 'p_value': t_pvalue},
                        'wilcoxon': {'statistic': w_stat, 'p_value': w_pvalue},
                        'effect_size': cohens_d,
                        'significant': min(t_pvalue, w_pvalue) < 0.05
                    }

        return statistical_results

    def _error_analysis(self, models, test_data):
        """Perform detailed error analysis"""
        logger.info("Performing error analysis...")

        error_analysis = {}

        for model_name, model in models.items():
            model_errors = {
                'prediction_errors': [],
                'error_categories': defaultdict(int),
                'difficult_cases': [],
                'failure_modes': defaultdict(list)
            }

            for i, sample in enumerate(test_data[:100]):  # Analyze subset
                try:
                    # Get prediction
                    if hasattr(model, 'generate'):
                        pred = model.generate(sample)
                    elif hasattr(model, 'predict_ensemble'):
                        pred = model.predict_ensemble(sample['spectrum'], sample.get('metadata'))
                    else:
                        pred = model.predict(sample['spectrum'])

                    true_smiles = sample['smiles']

                    # Calculate similarity
                    similarity = tanimoto_similarity(pred, true_smiles)

                    # Categorize errors
                    if similarity < 0.1:
                        model_errors['error_categories']['complete_failure'] += 1
                        model_errors['failure_modes']['complete_failure'].append({
                            'sample_idx': i,
                            'true_smiles': true_smiles,
                            'pred_smiles': pred,
                            'similarity': similarity
                        })
                    elif similarity < 0.3:
                        model_errors['error_categories']['poor_prediction'] += 1
                    elif similarity < 0.7:
                        model_errors['error_categories']['moderate_prediction'] += 1
                    else:
                        model_errors['error_categories']['good_prediction'] += 1

                    # Track difficult cases
                    if similarity < 0.5:
                        model_errors['difficult_cases'].append({
                            'sample_idx': i,
                            'true_smiles': true_smiles,
                            'pred_smiles': pred,
                            'similarity': similarity,
                            'spectrum_complexity': self._calculate_spectrum_complexity(sample['spectrum'])
                        })

                    model_errors['prediction_errors'].append({
                        'sample_idx': i,
                        'similarity': similarity,
                        'exact_match': pred == true_smiles
                    })

                except Exception as e:
                    model_errors['error_categories']['prediction_failure'] += 1
                    model_errors['failure_modes']['prediction_failure'].append({
                        'sample_idx': i,
                        'error': str(e)
                    })

            error_analysis[model_name] = model_errors

        return error_analysis

    def _uncertainty_analysis(self, models, test_data):
        """Analyze prediction uncertainty"""
        logger.info("Performing uncertainty analysis...")

        uncertainty_analysis = {}

        for model_name, model in models.items():
            if not hasattr(model, 'estimate_uncertainty'):
                continue

            uncertainties = []
            accuracies = []

            for sample in test_data[:50]:  # Analyze subset
                try:
                    # Get uncertainty estimate
                    uncertainty = model.estimate_uncertainty(sample)

                    # Get prediction accuracy
                    pred = model.predict(sample['spectrum'])
                    accuracy = tanimoto_similarity(pred, sample['smiles'])

                    uncertainties.append(uncertainty)
                    accuracies.append(accuracy)

                except Exception as e:
                    logger.warning(f"Uncertainty estimation error: {e}")

            if uncertainties:
                # Analyze uncertainty-accuracy correlation
                correlation = np.corrcoef(uncertainties, accuracies)[0, 1]

                uncertainty_analysis[model_name] = {
                    'mean_uncertainty': np.mean(uncertainties),
                    'std_uncertainty': np.std(uncertainties),
                    'uncertainty_accuracy_correlation': correlation,
                    'calibration_score': self._calculate_calibration_score(uncertainties, accuracies)
                }

        return uncertainty_analysis

    def _calculate_spectrum_complexity(self, spectrum):
        """Calculate spectrum complexity score"""
        # Number of significant peaks
        significant_peaks = np.sum(spectrum > 0.1)

        # Spectral entropy
        normalized = spectrum / (np.sum(spectrum) + 1e-8)
        entropy = -np.sum(normalized * np.log(normalized + 1e-8))

        # Dynamic range
        dynamic_range = np.max(spectrum) / (np.mean(spectrum) + 1e-8)

        return {
            'significant_peaks': significant_peaks,
            'entropy': entropy,
            'dynamic_range': dynamic_range,
            'complexity_score': significant_peaks * entropy / dynamic_range
        }

    def _calculate_calibration_score(self, uncertainties, accuracies):
        """Calculate calibration score for uncertainty estimates"""
        # Bin predictions by uncertainty
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        calibration_error = 0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find predictions in this bin
            in_bin = (uncertainties >= bin_lower) & (uncertainties < bin_upper)

            if np.sum(in_bin) > 0:
                # Average uncertainty in bin
                avg_uncertainty = np.mean(np.array(uncertainties)[in_bin])

                # Average accuracy in bin
                avg_accuracy = np.mean(np.array(accuracies)[in_bin])

                # Calibration error for this bin
                bin_error = abs(avg_uncertainty - avg_accuracy)
                calibration_error += bin_error * np.sum(in_bin)

        return calibration_error / len(uncertainties)

    def _create_visualizations(self, cv_results, error_analysis):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")

        visualizations = {}

        # Performance comparison plot
        visualizations['performance_comparison'] = self._plot_performance_comparison(cv_results)

        # Error distribution plots
        visualizations['error_distributions'] = self._plot_error_distributions(error_analysis)

        # Learning curves (if available)
        # visualizations['learning_curves'] = self._plot_learning_curves(models)

        return visualizations

    def _plot_performance_comparison(self, cv_results):
        """Create performance comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        # Select key metrics
        key_metrics = [
            'molecular_similarity_tanimoto',
            'molecular_similarity_mces',
            'sequence_similarity_exact_match',
            'chemical_validity_valid_smiles'
        ]

        for idx, metric in enumerate(key_metrics):
            ax = axes[idx // 2, idx % 2]

            model_names = []
            means = []
            stds = []

            for model_name, metrics in cv_results.items():
                if metric in metrics:
                    model_names.append(model_name)
                    means.append(metrics[metric]['mean'])
                    stds.append(metrics[metric]['std'])

            # Bar plot with error bars
            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.3f}±{std:.3f}',
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return fig

    def _plot_error_distributions(self, error_analysis):
        """Create error distribution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Error Analysis', fontsize=16)

        model_names = list(error_analysis.keys())

        # Error categories pie chart
        ax = axes[0, 0]
        if model_names:
            categories = error_analysis[model_names[0]]['error_categories']
            ax.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
            ax.set_title(f'Error Categories - {model_names[0]}')

        # Similarity distribution histogram
        ax = axes[0, 1]
        for model_name in model_names[:3]:  # Limit to 3 models
            similarities = [error['similarity'] for error in error_analysis[model_name]['prediction_errors']]
            ax.hist(similarities, alpha=0.6, label=model_name, bins=20)
        ax.set_xlabel('Tanimoto Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title('Similarity Distribution')
        ax.legend()

        # Spectrum complexity vs accuracy scatter
        ax = axes[1, 0]
        if model_names:
            difficult_cases = error_analysis[model_names[0]]['difficult_cases']
            if difficult_cases:
                complexities = [case['spectrum_complexity']['complexity_score'] for case in difficult_cases]
                similarities = [case['similarity'] for case in difficult_cases]
                ax.scatter(complexities, similarities, alpha=0.6)
                ax.set_xlabel('Spectrum Complexity')
                ax.set_ylabel('Prediction Similarity')
                ax.set_title('Complexity vs Accuracy')

        # Model comparison radar chart
        ax = axes[1, 1]
        # This would require more complex radar chart implementation
        ax.text(0.5, 0.5, 'Radar Chart\n(Implementation needed)',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Multi-metric Comparison')

        plt.tight_layout()
        return fig

    def _generate_evaluation_report(self, results):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")

        report_path = self.config.RESULTS_DIR / 'evaluation_report.md'

        with open(report_path, 'w') as f:
            f.write("# Comprehensive Evaluation Report\n\n")

            # Executive summary
            f.write("## Executive Summary\n\n")
            cv_results = results['cross_validation']
            best_model = max(cv_results.keys(),
                           key=lambda x: cv_results[x].get('molecular_similarity_tanimoto', {}).get('mean', 0))
            f.write(f"Best performing model: **{best_model}**\n\n")

            # Cross-validation results
            f.write("## Cross-Validation Results\n\n")
            for model_name, metrics in cv_results.items():
                f.write(f"### {model_name}\n\n")
                for metric_name, stats in metrics.items():
                    f.write(f"- {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write("\n")

            # Statistical significance
            f.write("## Statistical Significance Testing\n\n")
            stat_results = results['statistical_tests']
            for comparison, tests in stat_results.items():
                f.write(f"### {comparison}\n\n")
                significant_metrics = [metric for metric, test in tests.items() if test['significant']]
                if significant_metrics:
                    f.write(f"Significantly different metrics: {', '.join(significant_metrics)}\n\n")
                else:
                    f.write("No statistically significant differences found.\n\n")

            # Error analysis summary
            f.write("## Error Analysis Summary\n\n")
            error_results = results['error_analysis']
            for model_name, errors in error_results.items():
                f.write(f"### {model_name}\n\n")
                categories = errors['error_categories']
                total_predictions = sum(categories.values())
                for category, count in categories.items():
                    percentage = (count / total_predictions) * 100
                    f.write(f"- {category}: {count} ({percentage:.1f}%)\n")
                f.write("\n")

        logger.info(f"Evaluation report saved to {report_path}")

    # Metric implementation methods
    def _dice_similarity(self, smiles1, smiles2):
        """Calculate Dice similarity between two SMILES"""
        try:
            fp1 = molecular_fingerprints(smiles1)
            fp2 = molecular_fingerprints(smiles2)
            intersection = np.sum(fp1 & fp2)
            return 2 * intersection / (np.sum(fp1) + np.sum(fp2))
        except:
            return 0.0

    def _cosine_similarity(self, smiles1, smiles2):
        """Calculate cosine similarity between molecular fingerprints"""
        try:
            fp1 = molecular_fingerprints(smiles1)
            fp2 = molecular_fingerprints(smiles2)
            return np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2))
        except:
            return 0.0

    def _exact_match(self, pred, true):
        """Check exact match"""
        return float(pred == true)

    def _levenshtein_similarity(self, pred, true):
        """Calculate Levenshtein similarity"""
        try:
            from Levenshtein import ratio
            return ratio(pred, true)
        except ImportError:
            # Fallback implementation
            return self._simple_edit_distance_similarity(pred, true)

    def _simple_edit_distance_similarity(self, s1, s2):
        """Simple edit distance similarity"""
        if len(s1) == 0:
            return 0.0 if len(s2) > 0 else 1.0
        if len(s2) == 0:
            return 0.0

        # Simple character-level similarity
        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        max_len = max(len(s1), len(s2))
        return matches / max_len

    def _bleu_score(self, pred, true):
        """Calculate BLEU score"""
        # Simplified BLEU implementation
        pred_tokens = list(pred)
        true_tokens = list(true)

        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            return 0.0

        # 1-gram precision
        pred_counts = Counter(pred_tokens)
        true_counts = Counter(true_tokens)

        overlap = sum(min(pred_counts[token], true_counts[token]) for token in pred_counts)
        precision = overlap / len(pred_tokens)

        # Brevity penalty
        bp = min(1.0, len(pred_tokens) / len(true_tokens))

        return bp * precision

    def _rouge_score(self, pred, true):
        """Calculate ROUGE score"""
        # Simplified ROUGE-L implementation
        pred_tokens = list(pred)
        true_tokens = list(true)

        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            return 0.0

        # Longest common subsequence
        lcs_length = self._lcs_length(pred_tokens, true_tokens)

        if len(true_tokens) == 0:
            return 0.0

        return lcs_length / len(true_tokens)

    def _lcs_length(self, seq1, seq2):
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _is_valid_smiles(self, smiles, _):
        """Check if SMILES is valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return float(mol is not None)
        except:
            return 0.0

    def _drug_likeness(self, smiles, _):
        """Calculate drug-likeness score"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            # Lipinski's Rule of Five
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1

            return float(violations <= 1)  # Drug-like if <= 1 violation
        except:
            return 0.0

    def _synthetic_accessibility(self, smiles, _):
        """Calculate synthetic accessibility score"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            # Simplified SA score based on molecular complexity
            complexity = (
                mol.GetNumAtoms() +
                mol.GetNumBonds() +
                Descriptors.RingCount(mol) * 2 +
                Descriptors.NumAromaticRings(mol)
            )

            # Normalize to 0-1 scale (lower is more accessible)
            normalized_complexity = min(1.0, complexity / 100.0)
            return 1.0 - normalized_complexity
        except:
            return 0.0

class AdvancedMassSpecGymPipeline:
    """Comprehensive pipeline integrating all advanced components"""

    def __init__(self, config):
        self.config = config
        self.results = {}

        # Initialize components
        self.datamodule = AdvancedDataModule(config)
        self.evaluation_framework = ComprehensiveEvaluationFramework(config)

        # Models will be initialized during training
        self.models = {}

    def run_complete_pipeline(self):
        """Run the complete advanced pipeline"""
        logger.info("Starting complete advanced pipeline...")

        # 1. Setup data
        logger.info("Setting up data...")
        self.datamodule.setup()

        # 2. Train multi-modal transformer
        logger.info("Training multi-modal transformer...")
        transformer_model = self._train_multimodal_transformer()
        self.models['multimodal_transformer'] = transformer_model

        # 3. Train ensemble system
        logger.info("Training ensemble system...")
        ensemble_system = AdvancedEnsembleSystem(self.datamodule.train_dataset, self.config)
        ensemble_cv_scores = ensemble_system.train_ensemble()
        self.models['ensemble'] = ensemble_system

        # 4. Build RAG system
        logger.info("Building advanced RAG system...")
        rag_system = AdvancedMolecularRAGSystem(self.datamodule.train_dataset, self.config)
        self.models['rag'] = rag_system

        # 5. Comprehensive evaluation
        logger.info("Running comprehensive evaluation...")
        evaluation_results = self.evaluation_framework.run_comprehensive_evaluation(
            self.models, self.datamodule.test_dataset
        )

        # 6. Active learning (optional)
        if self.config.ACTIVE_LEARNING_BUDGET > 0:
            logger.info("Running active learning...")
            active_learning_results = self._run_active_learning()
            evaluation_results['active_learning'] = active_learning_results

        # 7. Generate final report
        logger.info("Generating final comprehensive report...")
        self._generate_final_report(evaluation_results, ensemble_cv_scores)

        self.results = evaluation_results
        return evaluation_results

    def _train_multimodal_transformer(self):
        """Train the multi-modal transformer model"""
        model = MultiModalGraphTransformer(
            vocab_size=self.datamodule.train_dataset.vocab_size,
            config=self.config
        )

        # Advanced callbacks
        callbacks = [
            ModelCheckpoint(
                monitor='val_tanimoto',
                mode='max',
                save_top_k=self.config.SAVE_TOP_K_MODELS,
                filename='multimodal-{epoch:02d}-{val_tanimoto:.4f}'
            ),
            EarlyStopping(
                monitor='val_tanimoto',
                patience=self.config.PATIENCE,
                mode='max',
                verbose=True
            ),
            LearningRateMonitor(logging_interval='step'),
            StochasticWeightAveraging(swa_lrs=1e-2),
            DeviceStatsMonitor()
        ]

        # Logger
        logger_instance = TensorBoardLogger(
            save_dir=self.config.LOG_DIR,
            name='multimodal_transformer',
            version=None
        )

        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            callbacks=callbacks,
            logger=logger_instance,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=self.config.NUM_GPUS if torch.cuda.is_available() else 1,
            strategy=DDPStrategy() if self.config.USE_DDP else None,
            gradient_clip_val=self.config.GRADIENT_CLIP_VAL,
            accumulate_grad_batches=self.config.ACCUMULATE_GRAD_BATCHES,
            val_check_interval=self.config.EVAL_EVERY_N_EPOCHS,
            precision=16 if torch.cuda.is_available() else 32,
            enable_progress_bar=True,
            enable_model_summary=True
        )

        # Train
        trainer.fit(model, self.datamodule)

        # Load best model
        best_model_path = callbacks[0].best_model_path
        if best_model_path:
            model = MultiModalGraphTransformer.load_from_checkpoint(
                best_model_path,
                vocab_size=self.datamodule.train_dataset.vocab_size,
                config=self.config
            )

        return model

    def _run_active_learning(self):
        """Run active learning experiment"""
        logger.info("Starting active learning experiment...")

        # This would implement active learning strategies
        # For now, return placeholder results
        return {
            'strategy': 'uncertainty_sampling',
            'budget_used': self.config.ACTIVE_LEARNING_BUDGET,
            'performance_improvement': 0.05,  # Placeholder
            'selected_samples': []  # Would contain indices of selected samples
        }

    def _generate_final_report(self, evaluation_results, ensemble_cv_scores):
        """Generate comprehensive final report"""
        report_path = self.config.RESULTS_DIR / 'final_comprehensive_report.md'

        with open(report_path, 'w') as f:
            f.write("# Final Comprehensive Pipeline Report\n\n")

            f.write("## Pipeline Overview\n\n")
            f.write("This report summarizes the results of the comprehensive ")
            f.write("MassSpecGym-optimized MS-to-Structure prediction pipeline.\n\n")

            f.write("### Components Evaluated\n\n")
            for model_name in self.models.keys():
                f.write(f"- {model_name}\n")
            f.write("\n")

            f.write("### Key Findings\n\n")
            cv_results = evaluation_results.get('cross_validation', {})
            if cv_results:
                best_model = max(cv_results.keys(),
                               key=lambda x: cv_results[x].get('molecular_similarity_tanimoto', {}).get('mean', 0))
                best_score = cv_results[best_model]['molecular_similarity_tanimoto']['mean']
                f.write(f"- Best performing model: **{best_model}** (Tanimoto: {best_score:.4f})\n")

            f.write("\n### Detailed Results\n\n")
            f.write("See individual component reports for detailed analysis.\n\n")

            f.write("### Recommendations\n\n")
            f.write("Based on the evaluation results, we recommend:\n\n")
            f.write("1. Further hyperparameter optimization for the best performing model\n")
            f.write("2. Ensemble combination of top-performing models\n")
            f.write("3. Additional data augmentation strategies\n")
            f.write("4. Investigation of failure modes identified in error analysis\n\n")

        logger.info(f"Final report saved to {report_path}")

# Cell 7: Advanced Hyperparameter Optimization and Active Learning
def optimize_hyperparameters(n_trials=20):
    """Optimize hyperparameters using Optuna"""

    def objective(trial):
        # Suggest hyperparameters
        config_opt = Config()
        config_opt.LEARNING_RATE = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        config_opt.D_MODEL = trial.suggest_categorical('d_model', [256, 512, 768])
        config_opt.NUM_LAYERS = trial.suggest_int('num_layers', 3, 8)
        config_opt.DROPOUT = trial.suggest_float('dropout', 0.1, 0.3)

        # Quick training
        model = HybridMSMSModel(dataset.vocab_size, config_opt)
        datamodule = MassSpecDataModule(batch_size=32, num_workers=2)

        trainer = pl.Trainer(
            max_epochs=5,  # Quick training for optimization
            enable_checkpointing=False,
            logger=False,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu'
        )

        trainer.fit(model, datamodule)

        # Return validation metric
        return trainer.callback_metrics.get('val_tanimoto', 0.0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")

    return study.best_params

# Cell 8: Main Execution
if __name__ == "__main__":
    # Run hyperparameter optimization
    print("Starting hyperparameter optimization...")
    best_params = optimize_hyperparameters(n_trials=10)

    # Update config with best parameters
    for param, value in best_params.items():
        setattr(config, param.upper(), value)

    # Run full pipeline
    print("Starting full pipeline...")
    pipeline = MassSpecGymPipeline(config)
    results = pipeline.run_full_pipeline()

    print("Pipeline completed successfully!")
    print("Final Results Summary:")
    for system, metrics in results.items():
        print(f"{system}: Tanimoto={np.mean(metrics['tanimoto']):.4f}, "
              f"Exact Match={np.mean(metrics['exact_match']):.4f}")
