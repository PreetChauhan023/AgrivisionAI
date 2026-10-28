"""Train MobileNetV3 on image folders: data/leaf_images/{train,val,test}/{class}/*.jpg
Example: python train_image_model.py --data data/leaf_images --epochs 8
Requires: torch torchvision pillow scikit-learn numpy
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from image_detection import _network


def run(args):
    random.seed(args.seed);
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    train_tf = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomResizedCrop(224, scale=(.75, 1.0)),
                                   transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                                   transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    root = Path(args.data)
    datasets_by_split = {s: datasets.ImageFolder(root / s, transform=train_tf if s == 'train' else eval_tf)
                         for s in ('train', 'val', 'test')}
    classes = datasets_by_split['train'].classes
    if len(classes) < 2: raise ValueError('Need at least two disease classes.')
    if any(ds.classes != classes for ds in datasets_by_split.values()):
        raise ValueError('Train, val and test must have identical class folders in identical order.')
    loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == 'train'), num_workers=0)
               for s, ds in datasets_by_split.items()}
    model = _network(len(classes))
    if not args.scratch:
        pretrained = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.features.load_state_dict(pretrained.features.state_dict())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best = float('-inf');
    best_state = None;
    history = []
    for epoch in range(args.epochs):
        record = {'epoch': epoch + 1}
        for split in ('train', 'val'):
            model.train(split == 'train')
            total = 0;
            correct = 0;
            running = 0
            for images, labels in loaders[split]:
                images = images.to(device);
                labels = labels.to(device)
                with torch.set_grad_enabled(split == 'train'):
                    outputs = model(images);
                    loss = loss_fn(outputs, labels)
                    if split == 'train':
                        optimizer.zero_grad();
                        loss.backward();
                        optimizer.step()
                total += len(labels);
                correct += (outputs.argmax(1) == labels).sum().item()
                running += loss.item() * len(labels)
            record[split + '_loss'] = running / max(1, total)
            record[split + '_accuracy'] = correct / max(1, total)
        history.append(record);
        print(record)
        if record['val_accuracy'] > best:
            best = record['val_accuracy'];
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state);
    model.eval()
    actual = [];
    predicted = []
    with torch.inference_mode():
        for images, labels in loaders['test']:
            outputs = model(images.to(device));
            predicted.extend(outputs.argmax(1).cpu().tolist());
            actual.extend(labels.tolist())
    metrics = {'accuracy': accuracy_score(actual, predicted),
               'balanced_accuracy': balanced_accuracy_score(actual, predicted),
               'classification_report': classification_report(actual, predicted, labels=list(range(len(classes))),
                                                              target_names=classes, zero_division=0, output_dict=True),
               'n_test': len(actual), 'best_validation_accuracy': best, 'history': history, 'classes': classes,
               'warning': 'Ensure no plant/image duplicates cross splits; test images must remain untouched during tuning.'}
    out = Path(args.output);
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'classes': classes, 'architecture': 'mobilenet_v3_small'}, out)
    out.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Saved:', out, 'Test accuracy:', metrics['accuracy'], 'Balanced accuracy:', metrics['balanced_accuracy'])


if __name__ == '__main__':
    p = argparse.ArgumentParser();
    p.add_argument('--data', required=True);
    p.add_argument('--output', default='artifacts/leaf_disease_mobilenet.pt')
    p.add_argument('--epochs', type=int, default=8);
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.0003);
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true');
    p.add_argument('--scratch', action='store_true');
    run(p.parse_args())
