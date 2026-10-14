# DEVIATIONS — Federated Input-Level Privacy

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Datasets | PubFig, LFW, PINS | CelebA (Fed train) + LFW (verify) | PubFig/PINS not in monorepo |
| D2 | Backbone | Paper FR backbone | ResNet-18 classifier | Same role; paper-compatible FedAvg |
| D3 | MIA | Oriented max-softmax ROC-AUC | Same score family, train vs val probe | Protocol adapted to CelebA splits |
| D4 | Clients / rounds | Paper FL schedule | 4 clients, 20 rounds (full) | Practical full-train budget |
| D5 | CIFAR privacy suite | Mentioned in PDF | Omitted | Face-biometric focus |
