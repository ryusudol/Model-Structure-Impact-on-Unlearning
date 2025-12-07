"""
Shared utilities for machine unlearning notebooks.
Provides models, data loading, evaluation, UMAP, attack metrics, and result generation.
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from torchvision import datasets, transforms
import timm
import umap

# ============================================================================
# Configuration
# ============================================================================

SEED = 2048
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
NUM_CLASSES = 10
UMAP_N_NEIGHBORS = 9
UMAP_MIN_DIST = 0.45
UMAP_DATA_SIZE = 2000

# Layer names for CKA computation (10 layers each to match)
RESNET_LAYERS = [
    'conv1',      # Initial convolution
    'layer1.0',   # ResNet block 1, sub-block 0
    'layer1.1',   # ResNet block 1, sub-block 1
    'layer2.0',   # ResNet block 2, sub-block 0
    'layer2.1',   # ResNet block 2, sub-block 1
    'layer3.0',   # ResNet block 3, sub-block 0
    'layer3.1',   # ResNet block 3, sub-block 1
    'layer4.0',   # ResNet block 4, sub-block 0
    'layer4.1',   # ResNet block 4, sub-block 1
    'fc'          # Final classifier
]

VGG_LAYERS = [
    'features.2',   # After first conv block (ReLU after conv1)
    'features.5',   # After second conv block
    'features.9',   # After third conv block
    'features.12',  # Mid third block
    'features.16',  # After fourth conv block
    'features.19',  # Mid fourth block
    'features.22',  # After fifth conv block
    'features.26',  # Mid fifth block
    'features.29',  # End of features
    'head.fc'       # Final classifier
]

# ============================================================================
# Models
# ============================================================================

def get_resnet18():
    """Load pretrained ResNet-18 for CIFAR-10 from Hugging Face Hub."""
    from torchvision import models
    from huggingface_hub import hf_hub_download
    
    # Create CIFAR-10 adapted ResNet-18 (smaller input, no maxpool)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    # Load pretrained weights from Hugging Face Hub
    repo_id = "jaeunglee/resnet18-cifar10-unlearning"
    weights_path = hf_hub_download(repo_id=repo_id, filename="resnet18_cifar10_full.pth")
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    
    return model


def get_vgg16bn():
    """Load pretrained VGG-16-BN for CIFAR-10 from timm."""
    # Use timm's standard VGG-16-BN and adapt for CIFAR-10
    model = timm.create_model("vgg16_bn", pretrained=True, num_classes=NUM_CLASSES)
    return model


# ============================================================================
# Data Loading
# ============================================================================

def get_data_loaders(
    batch_size: int,
    forget_class: int,
    data_dir: str = './data'
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 data and create data loaders.

    Returns:
        Tuple of (train_loader, test_loader, retain_loader, forget_loader)
    """
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Load datasets
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # Create indices for retain and forget sets
    train_targets = torch.tensor(train_set.targets)
    retain_indices = (train_targets != forget_class).nonzero(as_tuple=True)[0].tolist()
    forget_indices = (train_targets == forget_class).nonzero(as_tuple=True)[0].tolist()

    # Create subsets
    retain_set = Subset(train_set, retain_indices)
    forget_set = Subset(train_set, forget_indices)

    # Create loaders
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)
    retain_loader = DataLoader(retain_set, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)
    forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)

    return train_loader, test_loader, retain_loader, forget_loader, train_set, test_set


def get_umap_subset(
    train_set,
    test_set,
    num_samples: int = UMAP_DATA_SIZE,
    use_train: bool = True
) -> Tuple[Subset, DataLoader, List[int]]:
    """
    Create a balanced subset for UMAP visualization.

    Returns:
        Tuple of (subset, loader, selected_indices)
    """
    dataset = train_set if use_train else test_set
    targets = torch.tensor(dataset.targets)

    samples_per_class = num_samples // NUM_CLASSES
    generator = torch.Generator()
    generator.manual_seed(SEED)

    selected_indices = []
    for class_idx in range(NUM_CLASSES):
        class_indices = (targets == class_idx).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_indices), generator=generator)
        selected_indices.extend(class_indices[perm[:samples_per_class]].tolist())

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=num_samples, shuffle=False)

    return subset, loader, selected_indices


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict[int, float]]:
    """
    Evaluate model on a data loader.

    Returns:
        Tuple of (loss, accuracy, per_class_accuracies)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total = {i: 0 for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for label, pred in zip(labels, predicted):
                label = label.item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    avg_loss = total_loss / total
    accuracy = correct / total
    per_class_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
                     for i in range(NUM_CLASSES)}

    return avg_loss, accuracy, per_class_acc


def evaluate_model_with_distributions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict[int, float], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Evaluate model and compute prediction/confidence distributions.

    Returns:
        Tuple of (loss, accuracy, per_class_accs, label_dist, conf_dist)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total = {i: 0 for i in range(NUM_CLASSES)}

    # Distribution matrices: [ground_truth][predicted]
    label_counts = {i: {j: 0 for j in range(NUM_CLASSES)} for i in range(NUM_CLASSES)}
    conf_sums = {i: {j: 0.0 for j in range(NUM_CLASSES)} for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for i, (label, pred, prob) in enumerate(zip(labels, predicted, probs)):
                gt = label.item()
                p = pred.item()
                class_total[gt] += 1
                if p == gt:
                    class_correct[gt] += 1
                label_counts[gt][p] += 1
                conf_sums[gt][p] += prob[p].item()

    avg_loss = total_loss / total
    accuracy = correct / total
    per_class_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
                     for i in range(NUM_CLASSES)}

    # Normalize distributions
    label_dist = {}
    conf_dist = {}
    for gt in range(NUM_CLASSES):
        total_gt = class_total[gt] if class_total[gt] > 0 else 1
        label_dist[f"gt_{gt}"] = [label_counts[gt][p] / total_gt for p in range(NUM_CLASSES)]
        conf_dist[f"gt_{gt}"] = [conf_sums[gt][p] / max(label_counts[gt][p], 1) for p in range(NUM_CLASSES)]

    return avg_loss, accuracy, per_class_acc, label_dist, conf_dist


# ============================================================================
# UMAP Visualization
# ============================================================================

def get_layer_activations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract penultimate layer activations, predictions, and probabilities.

    Returns:
        Tuple of (activations, predicted_labels, probabilities)
    """
    model.eval()
    activations = []
    predictions = []
    all_probs = []

    # Register hook for penultimate layer
    activation_output = {}

    def get_activation(name):
        def hook(model, input, output):
            activation_output[name] = output.detach()
        return hook

    # Try to find the right layer for activation extraction
    # For ResNet: layer before fc
    # For VGG: pre_logits or features
    hook_handle = None

    # Try ResNet-style (timm)
    if hasattr(model, 'global_pool'):
        hook_handle = model.global_pool.register_forward_hook(get_activation('penultimate'))
    # Try ResNet-style (torchvision)
    elif hasattr(model, 'avgpool'):
        hook_handle = model.avgpool.register_forward_hook(get_activation('penultimate'))
    # Try VGG-style (timm) - pre_logits gives rich 4096-dim features
    elif hasattr(model, 'pre_logits'):
        hook_handle = model.pre_logits.register_forward_hook(get_activation('penultimate'))
    # Fallback for VGG: try features (conv output) or head.global_pool
    elif hasattr(model, 'features'):
        hook_handle = model.features.register_forward_hook(get_activation('penultimate'))
    elif hasattr(model, 'head') and hasattr(model.head, 'global_pool'):
        hook_handle = model.head.global_pool.register_forward_hook(get_activation('penultimate'))

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            if 'penultimate' in activation_output:
                act = activation_output['penultimate']
                if act.dim() > 2:
                    act = act.view(act.size(0), -1)
                activations.append(act.cpu().numpy())
            else:
                # Fallback: use logits as features
                activations.append(outputs.cpu().numpy())

            predictions.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if hook_handle:
        hook_handle.remove()

    return np.concatenate(activations), np.concatenate(predictions), np.concatenate(all_probs)


def compute_umap_embedding(
    activations: np.ndarray,
    labels: np.ndarray,
    forget_class: int,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST
) -> np.ndarray:
    """
    Compute UMAP embedding from activations.

    Returns:
        UMAP embedding array of shape (n_samples, 2)
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=SEED,
        n_jobs=-1
    )
    embedding = reducer.fit_transform(activations)
    return embedding


def prepare_umap_points(
    umap_subset: Subset,
    selected_indices: List[int],
    predicted_labels: np.ndarray,
    umap_embedding: np.ndarray,
    probs: np.ndarray,
    forget_class: int
) -> List[List[Any]]:
    """
    Prepare UMAP points data for JSON output.

    Returns:
        List of [gt, pred, idx, is_forget, x, y, compressed_probs]
    """
    points = []
    for i, (idx, (_, gt_label)) in enumerate(zip(selected_indices, umap_subset)):
        gt = gt_label if isinstance(gt_label, int) else gt_label.item()
        pred = int(predicted_labels[i])
        is_forget = 1 if gt == forget_class else 0
        x, y = float(umap_embedding[i, 0]), float(umap_embedding[i, 1])

        # Compress probabilities (only keep those > 0.001)
        prob_dict = {}
        for c in range(NUM_CLASSES):
            if probs[i, c] > 0.001:
                prob_dict[str(c)] = round(float(probs[i, c]), 4)

        points.append([gt, pred, idx, is_forget, round(x, 4), round(y, 4), prob_dict])

    return points


# ============================================================================
# Privacy Attack Metrics
# ============================================================================

def compute_attack_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forget_class: int
) -> Tuple[List[Dict], Dict, float]:
    """
    Compute privacy attack metrics (entropy and confidence-based).

    Returns:
        Tuple of (values, results, fqs)
        - values: per-sample entropy and confidence
        - results: attack scores at various thresholds
        - fqs: Forgetting Quality Score
    """
    model.eval()
    values = []
    forget_entropies = []
    forget_confidences = []
    retain_entropies = []
    retain_confidences = []

    sample_idx = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            for i in range(len(labels)):
                p = probs[i].cpu().numpy()
                entropy = -np.sum(p * np.log(p + 1e-10))
                confidence = float(probs[i].max().item())

                values.append({
                    'img': sample_idx,
                    'entropy': round(entropy, 4),
                    'confidence': round(confidence, 4)
                })

                if labels[i].item() == forget_class:
                    forget_entropies.append(entropy)
                    forget_confidences.append(confidence)
                else:
                    retain_entropies.append(entropy)
                    retain_confidences.append(confidence)

                sample_idx += 1

    # Calculate attack scores at different thresholds
    results = {
        'entropy_above_unlearn': [],
        'confidence_above_unlearn': []
    }

    # Generate thresholds
    thresholds = np.linspace(0, 2.5, 51)

    for thresh in thresholds:
        # Entropy-based: higher entropy on forget class = better forgetting
        forget_above = sum(1 for e in forget_entropies if e >= thresh)
        retain_above = sum(1 for e in retain_entropies if e >= thresh)

        fpr = retain_above / len(retain_entropies) if retain_entropies else 0
        fnr = 1 - (forget_above / len(forget_entropies)) if forget_entropies else 1
        attack_score = 1 - (fpr + fnr) / 2

        results['entropy_above_unlearn'].append({
            'threshold': round(float(thresh), 3),
            'fpr': round(fpr, 4),
            'fnr': round(fnr, 4),
            'attack_score': round(attack_score, 4)
        })

    # Confidence-based thresholds
    conf_thresholds = np.linspace(0, 1, 51)
    for thresh in conf_thresholds:
        # Lower confidence on forget class = better forgetting
        forget_below = sum(1 for c in forget_confidences if c <= thresh)
        retain_below = sum(1 for c in retain_confidences if c <= thresh)

        fpr = retain_below / len(retain_confidences) if retain_confidences else 0
        tpr = forget_below / len(forget_confidences) if forget_confidences else 0
        attack_score = 1 - abs(tpr - fpr)

        results['confidence_above_unlearn'].append({
            'threshold': round(float(thresh), 3),
            'fpr': round(fpr, 4),
            'tpr': round(tpr, 4),
            'attack_score': round(attack_score, 4)
        })

    # Forgetting Quality Score (FQS)
    # Based on entropy: higher avg entropy on forget class = better forgetting
    avg_forget_entropy = np.mean(forget_entropies) if forget_entropies else 0
    avg_retain_entropy = np.mean(retain_entropies) if retain_entropies else 0
    # Normalize to 0-1 range (assuming max entropy for 10 classes is ~2.3)
    max_entropy = np.log(NUM_CLASSES)
    fqs = min(1.0, avg_forget_entropy / max_entropy)

    return values, results, round(fqs, 4)


# ============================================================================
# CKA Similarity (Layer-wise)
# ============================================================================

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment between two feature matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    dot_XX = np.sum(X * X)
    dot_YY = np.sum(Y * Y)
    dot_XY = np.sum(X * Y)

    if dot_XX == 0 or dot_YY == 0:
        return 0.0

    return dot_XY / np.sqrt(dot_XX * dot_YY)


def get_layer_activations_for_cka(
    model: nn.Module,
    loader: DataLoader,
    layer_names: List[str],
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Extract activations from specified layers for CKA computation.

    Args:
        model: The model to extract activations from
        loader: DataLoader with input data
        layer_names: List of layer names to extract activations from
        device: Device to run on

    Returns:
        Dictionary mapping layer names to activation arrays
    """
    model.eval()
    activations = {name: [] for name in layer_names}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            act = output.detach()
            if act.dim() > 2:
                act = act.view(act.size(0), -1)
            activations[name].append(act.cpu().numpy())
        return hook

    # Register hooks for each layer
    for name in layer_names:
        try:
            # Navigate to the layer by name
            parts = name.split('.')
            module = model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            hooks.append(module.register_forward_hook(get_hook(name)))
        except (AttributeError, IndexError, KeyError):
            print(f"Warning: Layer '{name}' not found in model, skipping")
            continue

    # Forward pass to collect activations
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate activations
    result = {}
    for name in layer_names:
        if activations[name]:
            result[name] = np.concatenate(activations[name], axis=0)

    return result


def filter_loader_by_class(
    loader: DataLoader,
    target_class: int,
    include_class: bool = True,
    max_samples: int = None
) -> DataLoader:
    """
    Filter a DataLoader to include/exclude a specific class.

    Args:
        loader: Original DataLoader
        target_class: Class to filter
        include_class: If True, keep only target_class; if False, exclude it
        max_samples: Maximum number of samples to include (None for all)

    Returns:
        New DataLoader with filtered data
    """
    dataset = loader.dataset

    # Get indices based on filter condition
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_val = label if isinstance(label, int) else label.item()

        if include_class and label_val == target_class:
            indices.append(i)
        elif not include_class and label_val != target_class:
            indices.append(i)

    # Apply max_samples limit
    if max_samples is not None and len(indices) > max_samples:
        generator = torch.Generator()
        generator.manual_seed(SEED + target_class)
        perm = torch.randperm(len(indices), generator=generator)
        indices = [indices[i] for i in perm[:max_samples].tolist()]

    if not indices:
        return None

    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=False, num_workers=0)


def compute_cka_similarity(
    model_unlearned: nn.Module,
    model_original: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    forget_class: int,
    device: torch.device,
    model_type: str = "resnet"
) -> Dict[str, Any]:
    """
    Compute layer-wise CKA similarity between unlearned and original model.

    Args:
        model_unlearned: The unlearned model
        model_original: The original model (before unlearning)
        train_loader: Training data loader
        test_loader: Test data loader
        forget_class: The class that was unlearned
        device: Device to run on
        model_type: "resnet" or "vgg" to select appropriate layer names

    Returns:
        Dictionary with structure:
        {
            "layers": [...],
            "train": {"forget_class": [[...], ...], "other_classes": [[...], ...]},
            "test": {"forget_class": [[...], ...], "other_classes": [[...], ...]}
        }
    """
    # Select layer names based on model type
    layer_names = RESNET_LAYERS if model_type.lower() == "resnet" else VGG_LAYERS

    # Create filtered loaders
    # Training data: use 1/10 of samples for efficiency
    train_forget = filter_loader_by_class(train_loader, forget_class, include_class=True,
                                          max_samples=len(train_loader.dataset) // 10)
    train_other = filter_loader_by_class(train_loader, forget_class, include_class=False,
                                         max_samples=len(train_loader.dataset) // 10)

    # Test data: use 1/2 of samples
    test_forget = filter_loader_by_class(test_loader, forget_class, include_class=True,
                                         max_samples=len(test_loader.dataset) // 2)
    test_other = filter_loader_by_class(test_loader, forget_class, include_class=False,
                                        max_samples=len(test_loader.dataset) // 2)

    def compute_cka_for_loader(loader):
        """Compute layer-wise CKA for a single loader."""
        if loader is None:
            return [[0.0] for _ in layer_names]

        # Get activations from both models
        act_unlearned = get_layer_activations_for_cka(model_unlearned, loader, layer_names, device)
        act_original = get_layer_activations_for_cka(model_original, loader, layer_names, device)

        # Compute CKA for each layer
        cka_scores = []
        for layer_name in layer_names:
            if layer_name in act_unlearned and layer_name in act_original:
                score = linear_cka(act_unlearned[layer_name], act_original[layer_name])
                cka_scores.append([round(score, 3)])
            else:
                cka_scores.append([0.0])

        return cka_scores

    # Compute CKA for all four combinations
    print("  Computing CKA for train/forget_class...")
    train_forget_cka = compute_cka_for_loader(train_forget)

    print("  Computing CKA for train/other_classes...")
    train_other_cka = compute_cka_for_loader(train_other)

    print("  Computing CKA for test/forget_class...")
    test_forget_cka = compute_cka_for_loader(test_forget)

    print("  Computing CKA for test/other_classes...")
    test_other_cka = compute_cka_for_loader(test_other)

    return {
        "layers": layer_names,
        "train": {
            "forget_class": train_forget_cka,
            "other_classes": train_other_cka
        },
        "test": {
            "forget_class": test_forget_cka,
            "other_classes": test_other_cka
        }
    }


# ============================================================================
# Result Generation
# ============================================================================

def format_distribution(dist: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Format distribution dictionary with rounded values."""
    return {k: [round(v, 4) for v in vals] for k, vals in dist.items()}


def create_results_json(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    umap_subset: Subset,
    umap_loader: DataLoader,
    selected_indices: List[int],
    forget_class: int,
    method_name: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    runtime: float,
    device: torch.device,
    original_model: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """
    Create full results JSON matching existing codebase format.
    """
    criterion = nn.CrossEntropyLoss()
    remain_classes = [i for i in range(NUM_CLASSES) if i != forget_class]

    # Evaluate on train set
    print("Evaluating on train set...")
    train_loss, train_acc, train_class_accs, train_label_dist, train_conf_dist = \
        evaluate_model_with_distributions(model, train_loader, criterion, device)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc, test_class_accs, test_label_dist, test_conf_dist = \
        evaluate_model_with_distributions(model, test_loader, criterion, device)

    # Calculate metrics
    ua = train_class_accs[forget_class]
    ra = sum(train_class_accs[i] for i in remain_classes) / len(remain_classes)
    tua = test_class_accs[forget_class]
    tra = sum(test_class_accs[i] for i in remain_classes) / len(remain_classes)

    # UMAP computation
    print("Computing UMAP embeddings...")
    activations, predicted_labels, probs = get_layer_activations(model, umap_loader, device)
    umap_embedding = compute_umap_embedding(activations, predicted_labels, forget_class)
    points = prepare_umap_points(umap_subset, selected_indices, predicted_labels,
                                  umap_embedding, probs, forget_class)

    # Attack metrics
    print("Computing attack metrics...")
    values, attack_results, fqs = compute_attack_metrics(model, train_loader, device, forget_class)

    # CKA similarity (if original model provided)
    cka_result = None
    if original_model is not None:
        print("Computing CKA similarity...")
        # Determine model type from model_name
        model_type = "vgg" if "vgg" in model_name.lower() else "resnet"
        cka_result = compute_cka_similarity(
            model, original_model, train_loader, test_loader,
            forget_class, device, model_type
        )

    # Generate unique ID
    result_id = uuid.uuid4().hex[:4]

    # Create results dictionary
    results = {
        "CreatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ID": result_id,
        "FC": forget_class,           # int, not str
        "Type": "Unlearned",
        "Model": model_name,
        "Base": "pretrained",
        "Method": method_name,
        "Epoch": epochs,              # int, not str
        "BS": batch_size,             # int, not str
        "LR": learning_rate,          # float, not str
        "UA": round(ua, 3),
        "RA": round(ra, 3),
        "TUA": round(tua, 3),
        "TRA": round(tra, 3),
        "RTE": round(runtime, 1),
        "FQS": fqs,
        "accs": [round(train_class_accs[i], 3) for i in range(NUM_CLASSES)],
        "t_accs": [round(test_class_accs[i], 3) for i in range(NUM_CLASSES)],
        "label_dist": format_distribution(train_label_dist),
        "t_label_dist": format_distribution(test_label_dist),
        "conf_dist": format_distribution(train_conf_dist),
        "t_conf_dist": format_distribution(test_conf_dist),
        "points": points,
        "attack": {
            "values": values[:200],  # Limit to first 200 for UMAP subset
            "results": attack_results
        }
    }

    if cka_result:
        results["cka"] = cka_result

    return results


def save_results(
    result: Dict[str, Any],
    model: nn.Module,
    output_dir: str = "notebook_results"
) -> str:
    """
    Save results JSON and model weights.

    Returns:
        Path to saved JSON file
    """
    model_name = result.get("Model", "Unknown")
    method_name = result.get("Method", "Unknown")
    result_id = result.get("ID", "0000")

    # Create output directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(model_dir, f"{method_name}_{result_id}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Save model weights
    weights_path = os.path.join(model_dir, f"{method_name}_{result_id}.pth")
    torch.save(model.state_dict(), weights_path)

    print(f"Results saved to: {json_path}")
    print(f"Weights saved to: {weights_path}")

    return json_path
