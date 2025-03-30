#chiral 

# 3D-Stereo Graph Neural Network - Chiral Center Resolution

import torch
from torch_geometric.nn import GATv2Conv
from rdkit import Chem

class StereoNet(torch.nn.Module):
    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 32,
                 hidden_dim: int = 256,
                 num_heads: int = 8):
        super().__init__()

        # Graph attention layers
        self.conv1 = GATv2Conv(node_dim, hidden_dim, edge_dim=edge_dim, heads=num_heads)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, edge_dim=edge_dim)

        # Stereo prediction head
        self.stereo_head = torch.nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # R/S probability
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)

        return self.stereo_head(x)

def smiles_to_graph(smiles: str, device: torch.device):
    """Convert SMILES to PyTorch Geometric graph"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    mol = Chem.AddHs(mol)
    
    # Node features (atom type as one-hot)
    num_atoms = mol.GetNumAtoms()
    atom_types = ['C', 'H', 'O', 'N', 'S', 'Fe', 'Mg', 'Co']
    x = torch.zeros((num_atoms, len(atom_types)), dtype=torch.float, device=device)
    for i, atom in enumerate(mol.GetAtoms()):
        idx = atom_types.index(atom.GetSymbol()) if atom.GetSymbol() in atom_types else 0
        x[i, idx] = 1.0

    # Edge indices and attributes
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([float(bond.GetBondTypeAsDouble())] * 2)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t() if edge_index else torch.empty((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=device).unsqueeze(1) if edge_attr else torch.empty((0, 1), dtype=torch.float, device=device)

    return type('Data', (), {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr})

# Training function
def train_stereonet(model, train_loader, epochs=10, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:  # Expects graph data with y (R/S labels)
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y)  # y = tensor of 0/1 labels per atom
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(train_loader)}")