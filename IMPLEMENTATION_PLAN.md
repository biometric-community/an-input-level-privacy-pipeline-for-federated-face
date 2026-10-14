# Implementation Plan

## Pipeline
1. Client: n-CutMix soft targets → keyed G×G block permutation → inverse decode → backbone
2. Server: FedAvg of client weights
3. Eval: LFW verification, max-softmax MIA AUC, clean vs obfuscated top-1

## Figures
- train_loss.svg — FedAvg round loss
- privacy_metrics.svg — obfuscation gap + MIA AUC
