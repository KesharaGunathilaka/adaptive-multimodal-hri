# WP5 - robustness studies

## A. Leave-one-actor-out cross-validation

Each actor is held out entirely; the model is trained on the remaining actors and evaluated on the held-out person. The headline split holds out scenarios, so this is a strictly stronger generalization test.

| held-out actor | test clips | fusion | rules | diff |
|---|---|---|---|---|
| P01 | 700 | **0.9657** | 0.9300 | +0.0357 |
| P02 | 655 | **0.9221** | 0.8947 | +0.0275 |
| P04 | 414 | **0.8188** | 0.7802 | +0.0386 |
| P06 | 530 | **0.9075** | 0.8453 | +0.0623 |
| **mean** | - | **0.9036** | 0.8625 | +0.0410 |

## B. Cue-reliability sweep

With probability *p* a cue's measured vector is replaced by a real vector from a different ground-truth class - a realistic confusion, not random noise. This traces how each method degrades as perception worsens.

**Corrupting emotion**

| p | fusion | rules | diff |
|---|---|---|---|
| 0.0 | **0.8189** | 0.7193 | +0.0996 |
| 0.1 | **0.7658** | 0.6805 | +0.0854 |
| 0.2 | **0.7050** | 0.6313 | +0.0737 |
| 0.3 | **0.6429** | 0.5692 | +0.0737 |
| 0.4 | **0.5899** | 0.5278 | +0.0621 |
| 0.5 | **0.5175** | 0.4683 | +0.0492 |

**Corrupting gesture**

| p | fusion | rules | diff |
|---|---|---|---|
| 0.0 | **0.8189** | 0.7193 | +0.0996 |
| 0.1 | **0.7594** | 0.6714 | +0.0880 |
| 0.2 | **0.6934** | 0.6158 | +0.0776 |
| 0.3 | **0.6352** | 0.5563 | +0.0789 |
| 0.4 | **0.5938** | 0.5239 | +0.0699 |
| 0.5 | **0.5110** | 0.4592 | +0.0517 |

**Corrupting all**

| p | fusion | rules | diff |
|---|---|---|---|
| 0.0 | **0.8189** | 0.7193 | +0.0996 |
| 0.1 | **0.6882** | 0.6158 | +0.0724 |
| 0.2 | **0.5860** | 0.5317 | +0.0543 |
| 0.3 | **0.4605** | 0.4088 | +0.0517 |
| 0.4 | **0.3959** | 0.3545 | +0.0414 |
| 0.5 | **0.2911** | 0.2678 | +0.0233 |

Note on reading these curves: as *p* grows both methods approach chance, so the absolute difference necessarily narrows. The relative advantage (fusion / rules) is the quantity that is comparable across noise levels:

| corrupted | p=0.0 | p=0.2 | p=0.4 |
|---|---|---|---|
| emotion | 1.138x | 1.117x | 1.118x |
| gesture | 1.138x | 1.126x | 1.133x |
| all | 1.138x | 1.102x | 1.117x |

A roughly constant ratio means fusion keeps a proportional advantage at every level of perception quality, rather than the advantage being an artefact of one operating point.
