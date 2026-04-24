import os
import time
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
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
            torch.full((out_features, in_features), 0.5))
        nn.init.kaiming_uniform_(self.weight, a=0.01)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.gate_scores.clamp(0.0, 1.0)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)
    def get_gates(self) -> torch.Tensor:
        return self.gate_scores.clamp(0.0, 1.0).detach()
    def sparsity(self, threshold: float = 0.05) -> float:
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
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1     = PrunableLinear(256 * 4 * 4, 512)
        self.fc2     = PrunableLinear(512, num_classes)
        self.dropout = nn.Dropout(0.4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    def prunable_list(self) -> list:
        return [self.fc1, self.fc2]
    def overall_sparsity(self, threshold: float = 0.05) -> float:
        all_g = torch.cat([l.get_gates().flatten() for l in self.prunable_list()])
        return (all_g < threshold).float().mean().item()
    def per_layer_sparsity(self, threshold: float = 0.05) -> dict:
        return {f"FC_{i}": l.sparsity(threshold)
                for i, l in enumerate(self.prunable_list())}
def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    total, n = torch.tensor(0.0, device=device), 0
    for layer in model.prunable_list():
        gates  = layer.gate_scores.clamp(0.0, 1.0)   
        total += gates.sum()
        n     += layer.gate_scores.numel()
    return total / n
class LambdaScheduler:
    def __init__(self, lambda_max: float, warmup_epochs: int):
        self.lambda_max    = lambda_max
        self.warmup_epochs = warmup_epochs
    def get(self, epoch: int) -> float:
        return self.lambda_max * min(1.0, epoch / max(self.warmup_epochs, 1))
def train_one_epoch(model, loader, optimizer,
                    lambda_scheduler, epoch: int) -> dict:
    model.train()
    lam = lambda_scheduler.get(epoch)
    total_loss, cls_sum, spar_sum = 0.0, 0.0, 0.0
    correct, total_n = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits   = model(inputs)
        cls_loss = F.cross_entropy(logits, targets)
        spar     = sparsity_loss(model)
        loss     = cls_loss + lam * spar   
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        cls_sum    += cls_loss.item()
        spar_sum   += spar.item()
        _, pred    = logits.max(1)
        correct   += pred.eq(targets).sum().item()
        total_n   += targets.size(0)
    n = len(loader)
    return {
        "loss"     : total_loss / n,
        "cls_loss" : cls_sum    / n,
        "spar_loss": spar_sum   / n,
        "accuracy" : 100.0 * correct / total_n,
        "lambda"   : lam,
        "sparsity" : model.overall_sparsity(),
    }
@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    correct, total_n = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, pred   = model(inputs).max(1)
        correct  += pred.eq(targets).sum().item()
        total_n  += targets.size(0)
    return {
        "accuracy" : 100.0 * correct / total_n,
        "sparsity" : model.overall_sparsity(),
        "per_layer": model.per_layer_sparsity(),
    }
def run_experiment(lambda_max:    float,
                   train_loader,
                   test_loader,
                   epochs:        int = 40,
                   warmup_epochs: int = 5,
                   seed:          int = 42) -> tuple:
    torch.manual_seed(seed)
    print(f"\n{'='*50}\n  lambda_max = {lambda_max}\n{'='*50}")
    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    lam_sched = LambdaScheduler(lambda_max, warmup_epochs)
    g0 = model.fc1.get_gates()
    print(f"Gate init mean : {g0.mean().item():.3f}  (expected 0.500)")
    history = defaultdict(list)
    for epoch in range(epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, lam_sched, epoch)
        test_stats  = evaluate(model, test_loader)
        scheduler.step()
        for k, v in train_stats.items():
            history[f"train_{k}"].append(v)
        history["test_accuracy"].append(test_stats["accuracy"])
        history["test_sparsity"].append(test_stats["sparsity"])
        if (epoch + 1) % 5 == 0:
            print(f"Ep {epoch+1:3d} | "
                  f"Acc: {test_stats['accuracy']:5.1f}% | "
                  f"Sparsity: {test_stats['sparsity']:5.1%} | "
                  f"lambda: {train_stats['lambda']:.2e} | "
                  f"t: {time.time()-t0:.1f}s")
    final = evaluate(model, test_loader)
    print(f"\nFinal Accuracy  : {final['accuracy']:.2f}%")
    print(f"Final Sparsity  : {final['sparsity']:.2%}")
    for name, val in final["per_layer"].items():
        print(f"  {name}: {val:.2%}")
    return model, history, final
def plot_results(results: dict,
                 save_path: str = "results/gate_distribution.png"):
    lambdas = sorted(results.keys())
    n       = len(lambdas)
    colors  = ['#2ecc71', '#e67e22', '#e74c3c', '#3498db']
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for idx, lam in enumerate(lambdas):
        res   = results[lam]
        color = colors[idx % len(colors)]
        all_gates = torch.cat([
            res["model"].fc1.get_gates().flatten(),
            res["model"].fc2.get_gates().flatten()
        ]).cpu().numpy()
        ax = axes[idx]
        ax.hist(all_gates, bins=80, color=color, alpha=0.85, edgecolor='white')
        ax.axvline(x=0.05, color='black', linestyle='--',
                   linewidth=1.5, label='Threshold (0.05)')
        ax.set_xlabel("Gate Value", fontsize=10)
        ax.set_ylabel("Count",      fontsize=10)
        ax.legend(fontsize=8)
        acc  = res["final"]["accuracy"]
        spar = res["final"]["sparsity"] * 100
        ax.set_title(
            f"lambda = {lam:.0e}\nAcc: {acc:.1f}%  |  Sparsity: {spar:.1f}%",
            fontsize=11, fontweight='bold')
    plt.suptitle(
        "Self-Pruning Neural Network — Gate Distributions (CIFAR-10)",
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved -> {save_path}")
def print_results_table(results: dict):
    print("\n" + "="*55)
    print(f"{'Lambda':<10} {'Accuracy (%)':>14} {'Sparsity (%)':>14}")
    print("="*55)
    for lam in sorted(results.keys()):
        f = results[lam]["final"]
        print(f"{lam:<10.1e} "
              f"{f['accuracy']:>13.2f}% "
              f"{f['sparsity']*100:>13.2f}%")
    print("="*55)
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    LAMBDAS = [1e-2, 5e-2, 1e-1]
    results = {}
    for lam in LAMBDAS:
        model, history, final = run_experiment(
            lambda_max    = lam,
            train_loader  = train_loader,
            test_loader   = test_loader,
            epochs        = 40,
            warmup_epochs = 5,
        )
        results[lam] = {"model": model, "history": history, "final": final}
    print_results_table(results)
    plot_results(results, save_path="results/gate_distribution.png")
    print("\nDone. Upload results/gate_distribution.png to your GitHub repo.")
