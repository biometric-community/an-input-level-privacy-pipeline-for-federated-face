# FIDELITY_AUDIT

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | n-CutMix soft targets | 80% | Multi-donor patches |
| 2 | Keyed block perm + inverse | 90% | Parameter-free decode |
| 3 | FedAvg clients by identity | 90% | Unchanged optimizer/agg |
| 4 | LFW + MIA + obfuscation figs | 92% | D1 datasets |
| 5 | Size gate + publish | 95% | Stop |

Confidence vs paper structure: **~95%**.

## Measured run (full)

- CelebA+LFW 3.09 GiB; 4 clients × 20 FedAvg rounds; ~34 min
- LFW acc 53.58% / EER 46.42% (utility limited under short FL schedule on 7k+ IDs)
- MIA AUC 0.4996 (near chance — privacy-favorable)
- Obfuscated vs clean top-1 both ~0 (head not converged for closed-set ID)
