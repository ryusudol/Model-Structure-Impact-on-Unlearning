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
from torch_cka import CKA

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

RESNET_LAYERS = [
    'conv1',
    'layer1.0',
    'layer1.1',
    'layer2.0',
    'layer2.1',
    'layer3.0',
    'layer3.1',
    'layer4.0',
    'layer4.1',
    'fc'
]

VGG_LAYERS = [
    'features.2',
    'features.5',
    'features.9',
    'features.12',
    'features.16',
    'features.19',
    'features.22',
    'features.26',
    'features.29',
    'head.fc' 
]

# ============================================================================
# Models
# ============================================================================

def get_resnet18():
    from torchvision import models
    from huggingface_hub import hf_hub_download

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    repo_id = "jaeunglee/resnet18-cifar10-unlearning"
    print(f"  Downloading weights from {repo_id}...")
    weights_path = hf_hub_download(repo_id=repo_id, filename="resnet18_cifar10_full.pth")
    print(f"  Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    print("  Model loaded successfully.")

    return model


def get_vgg16bn():
    print("  Loading VGG16-BN pretrained on CIFAR-10 (via timm)...")
    model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
    print("  Model loaded successfully.")
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

        # Include all 10 class probabilities as a dict
        prob_dict = {str(c): round(float(probs[i, c]), 3) for c in range(NUM_CLASSES)}

        points.append([gt, pred, idx, is_forget, round(x, 2), round(y, 2), prob_dict])

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

                max_prob = float(probs[i].max().item())
                other_prob = 1 - max_prob
                confidence = np.log(max_prob + 1e-45) - np.log(other_prob + 1e-45)

                values.append({
                    'img': sample_idx,
                    'entropy': round(entropy, 4),
                    'confidence': round(confidence, 2)
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
        'entropy_above_retrain': [],
        'confidence_above_unlearn': [],
        'confidence_above_retrain': []
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

    # Placeholder retrain entropy scores (zeros since no retrain model)
    for thresh in thresholds:
        results['entropy_above_retrain'].append({
            'threshold': round(float(thresh), 3),
            'fpr': 0.0,
            'fnr': 0.0,
            'attack_score': 0.0
        })

    # Confidence-based thresholds (log-scaled range matching backend)
    conf_thresholds = np.linspace(-2.50, 10.00, 51)
    for thresh in conf_thresholds:
        # Higher confidence on retain class vs forget class
        forget_above = sum(1 for c in forget_confidences if c >= thresh)
        retain_above = sum(1 for c in retain_confidences if c >= thresh)

        tpr = forget_above / len(forget_confidences) if forget_confidences else 0
        fpr = retain_above / len(retain_confidences) if retain_confidences else 0
        fnr = 1.0 - tpr
        attack_score = 1 - abs(tpr - fpr)

        results['confidence_above_unlearn'].append({
            'threshold': round(float(thresh), 3),
            'fpr': round(fpr, 3),
            'fnr': round(fnr, 3),
            'attack_score': round(attack_score, 3)
        })

    # Placeholder retrain confidence scores (zeros since no retrain model)
    for thresh in conf_thresholds:
        results['confidence_above_retrain'].append({
            'threshold': round(float(thresh), 3),
            'fpr': 0.0,
            'fnr': 0.0,
            'attack_score': 0.0
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

def compute_cka_similarity(
    model_unlearned: nn.Module,
    model_original: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    forget_class: int,
    device: torch.device,
    model_type: str = "resnet",
    retrain_model: Optional[nn.Module] = None,
    batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Compute layer-wise CKA similarity between unlearned and original model.
    Uses torch-cka library to match backend implementation.

    Args:
        model_unlearned: The unlearned model
        model_original: The original model (before unlearning)
        train_loader: Training data loader
        test_loader: Test data loader
        forget_class: The class that was unlearned
        device: Device to run on
        model_type: "resnet" or "vgg" to select appropriate layer names
        retrain_model: Optional retrained model for additional comparison
        batch_size: Batch size for CKA computation

    Returns:
        Dictionary with structure matching backend:
        {
            "similarity": {"layers": [...], "train": {...}, "test": {...}},
            "similarity_retrain": {...} or None
        }
    """
    # Select layer names based on model type
    layer_names = RESNET_LAYERS if model_type.lower() == "resnet" else VGG_LAYERS

    def filter_loader(loader, is_train=False):
        """Filter loader to create forget_class and other_classes loaders."""
        targets = torch.tensor(loader.dataset.targets) if hasattr(loader.dataset, 'targets') else None

        if targets is None:
            # For Subset datasets, extract targets manually
            targets = []
            for i in range(len(loader.dataset)):
                _, label = loader.dataset[i]
                targets.append(label if isinstance(label, int) else label.item())
            targets = torch.tensor(targets)

        forget_indices = (targets == forget_class).nonzero(as_tuple=True)[0]
        other_indices = (targets != forget_class).nonzero(as_tuple=True)[0]

        if is_train:
            forget_samples = len(forget_indices) // 10
            other_samples = len(other_indices) // 10
        else:
            forget_samples = len(forget_indices) // 2
            other_samples = len(other_indices) // 2

        # Fix random seed for consistent CKA sampling - use forget_class as part of seed
        seed = 42 + forget_class
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Sort indices for complete determinism (matching backend)
        forget_indices_sorted = torch.sort(forget_indices)[0]
        other_indices_sorted = torch.sort(other_indices)[0]

        forget_sampled = forget_indices_sorted[:forget_samples]
        other_sampled = other_indices_sorted[:other_samples]

        forget_loader = DataLoader(
            Subset(loader.dataset, forget_sampled.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        other_loader = DataLoader(
            Subset(loader.dataset, other_sampled.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        return forget_loader, other_loader

    print("  Creating filtered data loaders...")
    forget_class_train_loader, other_classes_train_loader = filter_loader(train_loader, is_train=True)
    forget_class_test_loader, other_classes_test_loader = filter_loader(test_loader, is_train=False)

    def format_cka_results(results):
        """Format CKA results to match backend structure."""
        if results is None:
            return None
        return [[round(float(value), 3) for value in layer_results] for layer_results in results['CKA'].tolist()]

    # Set models to eval mode
    model_original.eval()
    model_unlearned.eval()

    print("  Computing CKA similarity (original vs unlearned)...")

    # Create CKA object for original vs unlearned comparison
    cka = CKA(
        model_original,
        model_unlearned,
        model1_name="Original",
        model2_name="Unlearned",
        model1_layers=layer_names,
        model2_layers=layer_names,
        device=device
    )

    with torch.no_grad():
        # Compare on forget class train data
        cka.compare(forget_class_train_loader, forget_class_train_loader)
        results_forget_train = cka.export()

        # Compare on other classes train data
        cka.compare(other_classes_train_loader, other_classes_train_loader)
        results_other_train = cka.export()

        # Compare on forget class test data
        cka.compare(forget_class_test_loader, forget_class_test_loader)
        results_forget_test = cka.export()

        # Compare on other classes test data
        cka.compare(other_classes_test_loader, other_classes_test_loader)
        results_other_test = cka.export()

    # Build similarity result
    similarity = {
        "layers": layer_names,
        "train": {
            "forget_class": format_cka_results(results_forget_train),
            "other_classes": format_cka_results(results_other_train)
        },
        "test": {
            "forget_class": format_cka_results(results_forget_test),
            "other_classes": format_cka_results(results_other_test)
        }
    }

    # Compute retrain comparison if retrain model provided
    similarity_retrain = None
    if retrain_model is not None:
        print("  Computing CKA similarity (retrain vs unlearned)...")
        retrain_model.eval()

        cka_retrain = CKA(
            retrain_model,
            model_unlearned,
            model1_name="Retrain",
            model2_name="Unlearned",
            model1_layers=layer_names,
            model2_layers=layer_names,
            device=device
        )

        with torch.no_grad():
            cka_retrain.compare(forget_class_train_loader, forget_class_train_loader)
            retrain_results_forget_train = cka_retrain.export()

            cka_retrain.compare(other_classes_train_loader, other_classes_train_loader)
            retrain_results_other_train = cka_retrain.export()

            cka_retrain.compare(forget_class_test_loader, forget_class_test_loader)
            retrain_results_forget_test = cka_retrain.export()

            cka_retrain.compare(other_classes_test_loader, other_classes_test_loader)
            retrain_results_other_test = cka_retrain.export()

        similarity_retrain = {
            "layers": layer_names,
            "train": {
                "forget_class": format_cka_results(retrain_results_forget_train),
                "other_classes": format_cka_results(retrain_results_other_train)
            },
            "test": {
                "forget_class": format_cka_results(retrain_results_forget_test),
                "other_classes": format_cka_results(retrain_results_other_test)
            }
        }

    return {
        "similarity": similarity,
        "similarity_retrain": similarity_retrain
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
    original_model: Optional[nn.Module] = None,
    retrain_model: Optional[nn.Module] = None
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

    # Attack metrics (computed on UMAP subset to match points array)
    print("Computing attack metrics...")
    values, attack_results, fqs = compute_attack_metrics(model, umap_loader, device, forget_class)

    # CKA similarity (if original model provided)
    cka_result = None
    if original_model is not None:
        print("Computing CKA similarity...")
        # Determine model type from model_name
        model_type = "vgg" if "vgg" in model_name.lower() else "resnet"
        cka_result = compute_cka_similarity(
            model, original_model, train_loader, test_loader,
            forget_class, device, model_type, retrain_model=retrain_model
        )

    # Generate unique ID
    result_id = uuid.uuid4().hex[:4]

    # Compute PA (Privacy Attack) score - max attack score from entropy-based results
    pa_score = max(r['attack_score'] for r in attack_results['entropy_above_unlearn'])

    # Create results dictionary
    results = {
        "CreatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ID": result_id,
        "FC": int(forget_class),           # Ensure int
        "Type": "Unlearned",
        "Base": "pretrained",
        "Method": method_name,
        "Epoch": int(epochs),              # Ensure int
        "BS": int(batch_size),             # Ensure int
        "LR": float(learning_rate),        # Ensure float
        "UA": round(float(ua), 3),         # Ensure float, rounded
        "RA": round(float(ra), 3),
        "TUA": round(float(tua), 3),
        "TRA": round(float(tra), 3),
        "RTE": round(float(runtime), 1),
        "FQS": float(fqs),                 # Ensure float
        "PA": round(float(pa_score), 4),
        "accs": [round(float(train_class_accs[i]), 3) for i in range(NUM_CLASSES)],
        "t_accs": [round(float(test_class_accs[i]), 3) for i in range(NUM_CLASSES)],
        "label_dist": format_distribution(train_label_dist),
        "t_label_dist": format_distribution(test_label_dist),
        "conf_dist": format_distribution(train_conf_dist),
        "t_conf_dist": format_distribution(test_conf_dist),
        "points": points,
        "attack": {
            "values": values,  # All 2000 values from UMAP subset
            "results": attack_results
        }
    }

    if cka_result:
        # Extract similarity results from new format
        results["cka"] = cka_result["similarity"]
        # Use actual retrain CKA if available, otherwise placeholder
        if cka_result["similarity_retrain"] is not None:
            results["cka_retrain"] = cka_result["similarity_retrain"]
        else:
            # Add placeholder retrain CKA (zeros since no retrain model available)
            results["cka_retrain"] = {
                "layers": cka_result["similarity"]["layers"],
                "train": {
                    "forget_class": [[0.0] for _ in cka_result["similarity"]["layers"]],
                    "other_classes": [[0.0] for _ in cka_result["similarity"]["layers"]]
                },
                "test": {
                    "forget_class": [[0.0] for _ in cka_result["similarity"]["layers"]],
                    "other_classes": [[0.0] for _ in cka_result["similarity"]["layers"]]
                }
            }
    else:
        # Provide default CKA structure for frontend compatibility
        model_type = "vgg" if "vgg" in model_name.lower() else "resnet"
        layer_names = VGG_LAYERS if model_type == "vgg" else RESNET_LAYERS
        results["cka"] = {
            "layers": layer_names,
            "train": {
                "forget_class": [[0.0] for _ in layer_names],
                "other_classes": [[0.0] for _ in layer_names]
            },
            "test": {
                "forget_class": [[0.0] for _ in layer_names],
                "other_classes": [[0.0] for _ in layer_names]
            }
        }
        results["cka_retrain"] = {
            "layers": layer_names,
            "train": {
                "forget_class": [[0.0] for _ in layer_names],
                "other_classes": [[0.0] for _ in layer_names]
            },
            "test": {
                "forget_class": [[0.0] for _ in layer_names],
                "other_classes": [[0.0] for _ in layer_names]
            }
        }

    # Validate points array structure before returning
    if points:
        for i, p in enumerate(points):
            if not isinstance(p, list) or len(p) != 7:
                raise ValueError(f"Invalid point at index {i}: expected 7 elements, got {len(p) if isinstance(p, list) else 'not a list'}")
            # Ensure numeric types are JSON-serializable
            p[0] = int(p[0])  # gt
            p[1] = int(p[1])  # pred
            p[2] = int(p[2])  # idx
            p[3] = int(p[3])  # is_forget
            p[4] = float(p[4])  # x
            p[5] = float(p[5])  # y
            # p[6] is prob_dict, already correct type

    return results


def save_results(
    result: Dict[str, Any],
    model: nn.Module,
    model_name: str,
    forget_class: int,
    output_dir: str = "backend/data"
) -> str:
    """
    Save results JSON and model weights.

    Args:
        result: Results dictionary
        model: The model to save weights for
        model_name: Name of the model (e.g., "ResNet-18", "VGG-16-BN")
        forget_class: The forget class (0-9)
        output_dir: Output directory path

    Returns:
        Path to saved JSON file
    """
    result_id = result.get("ID", "0000")
    output_dir = os.path.abspath(output_dir)

    # Create output directory: backend/data/{forget_class}/
    class_dir = os.path.join(output_dir, str(forget_class))
    os.makedirs(class_dir, exist_ok=True)

    # Save JSON as {id}.json
    json_path = os.path.join(class_dir, f"{result_id}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)

    # Save model weights with model name for identification
    weights_path = os.path.join(class_dir, f"{model_name}_{result_id}.pth")
    torch.save(model.state_dict(), weights_path)

    print(f"Results saved to: {json_path}")
    print(f"Weights saved to: {weights_path}")

    return json_path
