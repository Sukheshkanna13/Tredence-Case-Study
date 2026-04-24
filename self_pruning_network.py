import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import time
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
def get_dataloaders(batch_size_train: int = 128,
                    batch_size_test:  int = 256,
                    num_workers:      int = 2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2470, 0.2435, 0.2616)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size_train,
        shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=batch_size_test,
        shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Train : {len(train_ds):,} samples | {len(train_loader)} batches")
    print(f"Test  : {len(test_ds):,} samples  | {len(test_loader)} batches")
    return train_loader, test_loader
class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), -2.0))
        nn.init.kaiming_uniform_(self.weight, a=0.01)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)
    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()
    def sparsity(self, threshold: float = 1e-2) -> float:
        return (self.get_gates() < threshold).float().mean().item()
    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"sparsity={self.sparsity():.1%}")
class SelfPruningNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.prunable_layers = nn.ModuleList([
            PrunableLinear(256 * 4 * 4, 512),   
            PrunableLinear(512, num_classes),    
        ])
        self.dropout = nn.Dropout(0.4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                        
        x = x.view(x.size(0), -1)                  
        x = F.relu(self.prunable_layers[0](x))      
        x = self.dropout(x)
        x = self.prunable_layers[1](x)              
        return x
    def get_all_gates(self) -> list:
        return [layer.get_gates() for layer in self.prunable_layers]
    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        all_gates = torch.cat([g.flatten() for g in self.get_all_gates()])
        return (all_gates < threshold).float().mean().item()
    def per_layer_sparsity(self, threshold: float = 1e-2) -> dict:
        return {
            f"PrunableLinear_{i}": layer.sparsity(threshold)
            for i, layer in enumerate(self.prunable_layers)
        }
def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    total   = torch.tensor(0.0, device=device)
    total_n = 0
    for layer in model.prunable_layers:
        gates   = torch.sigmoid(layer.gate_scores)   
        total   = total + gates.sum()
        total_n += layer.gate_scores.numel()
    return total / total_n                            
class LambdaScheduler:
    def __init__(self, lambda_max: float,
                 warmup_epochs: int,
                 total_epochs:  int):
        self.lambda_max    = lambda_max
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
    def get(self, epoch: int) -> float:
        return self.lambda_max * min(1.0, epoch / max(self.warmup_epochs, 1))
    def __repr__(self) -> str:
        return (f"LambdaScheduler(max={self.lambda_max}, "
                f"warmup={self.warmup_epochs}/{self.total_epochs})")
def train_one_epoch(model, loader, optimizer,
                    lambda_scheduler, epoch: int) -> dict:
    model.train()
    lam = lambda_scheduler.get(epoch)
    total_loss    = 0.0
    cls_loss_sum  = 0.0
    spar_loss_sum = 0.0
    correct = 0
    total   = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        cls_loss  = F.cross_entropy(logits, targets)
        spar      = sparsity_loss(model)
        loss      = cls_loss + lam * spar
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        cls_loss_sum  += cls_loss.item()
        spar_loss_sum += spar.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(targets).sum().item()
        total   += targets.size(0)
    n = len(loader)
    return {
        "loss"     : total_loss    / n,
        "cls_loss" : cls_loss_sum  / n,
        "spar_loss": spar_loss_sum / n,
        "accuracy" : 100.0 * correct / total,
        "lambda"   : lam,
        "sparsity" : model.overall_sparsity(),
    }
@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    correct = 0
    total   = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, predicted = model(inputs).max(1)
        correct += predicted.eq(targets).sum().item()
        total   += targets.size(0)
    return {
        "accuracy" : 100.0 * correct / total,
        "sparsity" : model.overall_sparsity(),
        "per_layer": model.per_layer_sparsity(),
    }
def run_experiment(lambda_max:     float,
                   epochs:         int   = 40,
                   warmup_epochs:  int   = 5,
                   seed:           int   = 42,
                   train_loader=None,
                   test_loader=None) -> tuple:
    torch.manual_seed(seed)
    print(f"\n{'='*55}")
    print(f"  λ_max = {lambda_max}  |  warmup = {warmup_epochs} epochs")
    print(f"{'='*55}")
    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    lambda_sched = LambdaScheduler(
        lambda_max=lambda_max,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs)
    history = defaultdict(list)
    for epoch in range(epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, lambda_sched, epoch)
        test_stats = evaluate(model, test_loader)
        lr_scheduler.step()
        for k, v in train_stats.items():
            history[f"train_{k}"].append(v)
        history["test_accuracy"].append(test_stats["accuracy"])
        history["test_sparsity"].append(test_stats["sparsity"])
        if (epoch + 1) % 5 == 0:
            print(f"Ep {epoch+1:3d} | "
                  f"Acc: {test_stats['accuracy']:5.1f}% | "
                  f"Sparsity: {test_stats['sparsity']:5.1%} | "
                  f"λ: {train_stats['lambda']:.2e} | "
                  f"t: {time.time()-t0:.1f}s")
    final = evaluate(model, test_loader)
    print(f"\nFinal Accuracy  : {final['accuracy']:.2f}%")
    print(f"Final Sparsity  : {final['sparsity']:.2%}")
    print("Per-Layer       :")
    for name, val in final["per_layer"].items():
        print(f"  {name}: {val:.2%}")
    return model, history, final
def plot_results(results: dict, save_path: str = "gate_distribution.png"):
    lambdas = sorted(results.keys())
    n       = len(lambdas)
    colors  = ['#2ecc71', '#e67e22', '#e74c3c', '#3498db', '#9b59b6', '#1abc9c']
    fig = plt.figure(figsize=(6 * n, 10))
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.35)
    accs      = []
    sparsities = []
    for idx, lam in enumerate(lambdas):
        res   = results[lam]
        color = colors[idx % len(colors)]
        all_gates = torch.cat([
            g.flatten() for g in res["model"].get_all_gates()
        ]).cpu().numpy()
        ax = fig.add_subplot(gs[0, idx])
        ax.hist(all_gates, bins=80, color=color, alpha=0.85, edgecolor='white')
        ax.axvline(x=0.01, color='black', linestyle='--',
                   linewidth=1.2, label='Prune threshold (0.01)')
        ax.set_xlabel("Gate Value", fontsize=9)
        ax.set_ylabel("Count",      fontsize=9)
        ax.legend(fontsize=7)
        acc  = res["final"]["accuracy"]
        spar = res["final"]["sparsity"] * 100
        ax.set_title(f"λ = {lam:.0e}\nAcc: {acc:.1f}%  |  Sparse: {spar:.1f}%",
                     fontsize=10, fontweight='bold')
        accs.append(acc)
        sparsities.append(spar)
    ax_curve = fig.add_subplot(gs[1, :])
    ax_curve.plot(sparsities, accs, 'o-', color='#2c3e50',
                  linewidth=2.5, markersize=10)
    for i, lam in enumerate(lambdas):
        ax_curve.annotate(
            f"λ={lam:.0e}",
            (sparsities[i], accs[i]),
            textcoords="offset points",
            xytext=(8, 4), fontsize=9)
    ax_curve.set_xlabel("Sparsity Level (%)", fontsize=11)
    ax_curve.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax_curve.set_title("Accuracy vs Sparsity Trade-off",
                       fontsize=12, fontweight='bold')
    ax_curve.grid(True, alpha=0.3)
    fig.suptitle("Self-Pruning Neural Network — Gate Distributions & Trade-off",
                 fontsize=13, fontweight='bold', y=1.01)
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", save_path)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved → {out}")
def print_results_table(results: dict):
    print("\n" + "="*58)
    print(f"{'Lambda':<10} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("="*58)
    for lam in sorted(results.keys()):
        f = results[lam]["final"]
        print(f"{lam:<10.1e} "
              f"{f['accuracy']:>13.2f}% "
              f"{f['sparsity']*100:>13.2f}%")
    print("="*58)
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    LAMBDAS = [1e-2, 5e-2, 1e-1]
    results = {}
    for lam in LAMBDAS:
        model, history, final = run_experiment(
            lambda_max    = lam,
            epochs        = 40,
            warmup_epochs = 5,
            train_loader  = train_loader,
            test_loader   = test_loader,
        )
        results[lam] = {
            "model"  : model,
            "history": history,
            "final"  : final,
        }
    print_results_table(results)
    plot_results(results, save_path="gate_distribution.png")
    print("\nDone. Upload results/gate_distribution.png to your GitHub repo.")
