#moltransformer
# Molecular Transformer for Structure Assembly - 12-Layer Hardware-Optimized Architecture

import torch
import torch.nn as nn

class MolFormer(nn.Module):
    def __init__(self,
                 d_model: int = 768,
                 nhead: int = 12,
                 num_layers: int = 12,
                 max_seq_len: int = 512):
        super().__init__()

        # Molecular token embeddings
        self.token_embed = nn.Embedding(300, d_model)  # SMILES vocabulary
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer layers with optimization
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])

        # Output projection
        self.fc = nn.Linear(d_model, 300)  # Back to vocabulary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embed(x) + self.pos_embed(positions)

        for layer in self.layers:
            x = layer(x)

        return self.fc(x)

# Training function
def train_molformer(model, train_loader, epochs=10, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:  # Expects (inputs, targets) from DataLoader
            inputs, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 300), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(train_loader)}")