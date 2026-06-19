import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

# ============================================================
# WEEK 4 ASSIGNMENT — Node Classification on Cora
#
# You will implement a 2-layer Graph Convolutional Network
# (GCN) using message passing from scratch.
#
# Your job: fill in the three functions marked TODO.
#   1. normalize_adjacency  — build the normalised adj matrix
#   2. GCNLayer.forward     — one round of message passing
#   3. train_step           — loss + backprop + optimiser step
#
# Architecture:
#   Input (1433,) → GCNLayer(1433→64) → ReLU
#              → GCNLayer(64→7) → log-softmax → prediction
#
# Target: ~80% validation accuracy.
# ============================================================


# ------------------------------------------------------------
# Data loading  (nothing to change here)
# ------------------------------------------------------------

def load_cora():
    """
    Downloads Cora automatically on first run.

    Cora is a citation network:
      - 2708 nodes   (research papers)
      - 5429 edges   (citation links)
      - 1433 features per node (bag-of-words of the paper)
      - 7 classes    (research topics)

    Returns a single PyG Data object. The standard train/val/test
    split is already provided via data.train_mask etc.
    """
    dataset = Planetoid(root='./data', name='Cora')
    data    = dataset[0]
    print(f"Nodes: {data.num_nodes}  |  Edges: {data.num_edges}  "
          f"|  Features: {data.num_node_features}  |  Classes: {dataset.num_classes}")
    return data, dataset.num_classes


# ------------------------------------------------------------
# TODO 1 — Build the normalised adjacency matrix
# ------------------------------------------------------------

def normalize_adjacency(edge_index, num_nodes):
    """
    Build the symmetrically normalised adjacency matrix used in
    Kipf & Welling (2017):

        A_hat = D^{-1/2} * (A + I) * D^{-1/2}

    where A is the raw adjacency, I is the identity (self-loops),
    and D is the degree matrix of (A + I).

    Args:
        edge_index : LongTensor, shape (2, num_edges)
                     edge_index[0] = source nodes
                     edge_index[1] = target nodes
        num_nodes  : int

    Returns:
        A_hat : FloatTensor, shape (num_nodes, num_nodes)

    Steps:
        1. Build a dense (num_nodes x num_nodes) adjacency matrix A
           from edge_index.
        2. Add self-loops: A = A + I
        3. Compute the degree of each node: d[i] = sum of row i.
        4. Build D^{-1/2} as a diagonal matrix.
        5. Return D^{-1/2} @ A @ D^{-1/2}.

    Hint: torch.diag, torch.eye, and matrix multiplication (@) are enough.
    """
    # 1. Build a dense adjacency matrix A from edge_index
    A = torch.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1.0

    # 2. Add self-loops: A = A + I
    A = A + torch.eye(num_nodes)

    # 3. Degree of each node (row sums of A+I)
    d = A.sum(dim=1)

    # 4. D^{-1/2} as a diagonal matrix
    d_inv_sqrt = torch.diag(d.pow(-0.5))

    # 5. D^{-1/2} @ A @ D^{-1/2}
    A_hat = d_inv_sqrt @ A @ d_inv_sqrt
    return A_hat


# ------------------------------------------------------------
# TODO 2 — GCN layer (one round of message passing)
# ------------------------------------------------------------

class GCNLayer(nn.Module):
    """
    One GCN layer as defined in Kipf & Welling (2017):

        H_out = A_hat @ H_in @ W

    where:
        H_in  is the current node feature matrix  (num_nodes, in_features)
        W     is the learnable weight matrix       (in_features, out_features)
        A_hat is the normalised adjacency matrix   (num_nodes, num_nodes)
        H_out is the updated node features         (num_nodes, out_features)

    Concretely, multiplying A_hat @ H_in is the "message passing"
    step: each node aggregates the (normalised) features of its
    neighbours. The subsequent @ W is a linear transformation.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        # A single linear layer — no bias, following the original paper.
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, A_hat):
        """
        Args:
            x     : node features, shape (num_nodes, in_features)
            A_hat : normalised adjacency, shape (num_nodes, num_nodes)

        Returns:
            out : updated node features, shape (num_nodes, out_features)

        TODO:
            1. Apply self.linear to x  → transforms features (H @ W)
            2. Multiply A_hat by the result  → aggregates neighbours
            Return the aggregated result.

        Note: step 2 (A_hat @) is the message passing operation.
        Each node's new representation is a weighted sum of its
        neighbours' (transformed) features.
        """
        # 1. Linear transform: H @ W
        h = self.linear(x)
        # 2. Aggregate neighbours: A_hat @ (H @ W)
        out = A_hat @ h
        return out


# ------------------------------------------------------------
# Full 2-layer GCN model  (nothing to change here)
# ------------------------------------------------------------

class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, x, A_hat):
        x = self.layer1(x, A_hat)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer2(x, A_hat)
        return F.log_softmax(x, dim=1)


# ------------------------------------------------------------
# TODO 3 — Single training step
# ------------------------------------------------------------

def train_step(model, data, A_hat, optimiser):
    """
    Run one forward pass and one gradient update.

    Args:
        model     : GCN instance
        data      : PyG Data object (use data.train_mask to select
                    only the labelled training nodes for the loss)
        A_hat     : normalised adjacency matrix
        optimiser : torch optimiser

    Returns:
        loss : scalar float

    Steps:
        1. Set the model to training mode.
        2. Zero the gradients.
        3. Forward pass through the model → log-probabilities for all nodes.
        4. Compute NLL loss only on the training nodes
           (use data.train_mask and data.y).
        5. Backpropagate and step the optimiser.
        6. Return the loss value as a Python float.

    Hint: F.nll_loss takes (input, target). The model outputs
    log-softmax values, so NLL loss is the right choice.
    """
    model.train()
    optimiser.zero_grad()

    log_probs = model(data.x, A_hat)
    loss = F.nll_loss(log_probs[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimiser.step()

    return loss.item()


# ------------------------------------------------------------
# Evaluation  (nothing to change here)
# ------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, A_hat):
    model.eval()
    log_probs = model(data.x, A_hat)
    pred      = log_probs.argmax(dim=1)

    results = {}
    for split in ['train', 'val', 'test']:
        mask = data[f'{split}_mask']
        correct = (pred[mask] == data.y[mask]).sum().item()
        results[split] = correct / mask.sum().item()
    return results


# ------------------------------------------------------------
# Main training loop  (nothing to change here)
# ------------------------------------------------------------

def main():
    data, num_classes = load_cora()

    # Pre-compute A_hat once — it doesn't change during training.
    A_hat = normalize_adjacency(data.edge_index, data.num_nodes)

    model     = GCN(in_features=data.num_node_features,
                    hidden_dim=64,
                    num_classes=num_classes)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        loss = train_step(model, data, A_hat, optimiser)
        if epoch % 20 == 0:
            acc = evaluate(model, data, A_hat)
            print(f"Epoch {epoch:3d}  loss: {loss:.4f}  "
                  f"train: {acc['train']:.3f}  "
                  f"val: {acc['val']:.3f}  "
                  f"test: {acc['test']:.3f}")

    acc = evaluate(model, data, A_hat)
    print(f"\nFinal test accuracy: {acc['test']:.3f}")


if __name__ == '__main__':
    main()