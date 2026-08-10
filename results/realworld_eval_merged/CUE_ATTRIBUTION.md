# Cue attribution — the CAM-analogue for AttentionFusion

Generated 2026-08-07 01:03 · `data/final_merged` · headline test · self-attention `full` model, 3-seed ensemble mean · CLS token's last-layer attention weight onto each modality token (`fusion/model/attribution.py`).

**Note on 'CAM' terminology**: this is NOT Class Activation Mapping (that needs spatial feature maps a 24-dim vector doesn't have). It is the model's own attention weights — the mixing coefficients the classifier head actually reads from, extracted directly rather than approximated. `fusion_zoo.py`'s `ChannelAttentionFusion` is the unrelated SE/CBAM-style head that happened to share the initials.

## Overall — which cue does the model lean on, on average?

| Modality | Mean attention |
|---|---|
| emotion | 0.2952 |
| gesture | 0.3176 |
| motion | 0.1755 |
| context | 0.1006 |

## Per predicted intent

| Intent | emotion | gesture | motion | context |
|---|---|---|---|---|
| F01 | 0.3678 | 0.3124 | 0.1041 | 0.1035 |
| F02 | 0.3597 | 0.2141 | 0.2244 | 0.0814 |
| F03 | 0.2087 | 0.3396 | 0.2334 | 0.1031 |
| F04 | 0.2791 | 0.4028 | 0.0957 | 0.1079 |
| F05 | 0.1879 | 0.457 | 0.2022 | 0.0772 |
| F06 | 0.1887 | 0.3263 | 0.2729 | 0.0898 |
| F07 | 0.3748 | 0.2437 | 0.1458 | 0.1194 |
| F08 | 0.335 | 0.3157 | 0.1313 | 0.1197 |
| F10 | 0.2218 | 0.2726 | 0.2925 | 0.0968 |

## Qualitative spotlight — same gesture, different emotion

Does the model's attention visibly shift toward the disambiguating cue when the same gesture means different things?

| Gesture | Row | Context | Emotion | Motion | Intent | n | attn(emo) | attn(ges) | attn(mot) | attn(ctx) |
|---|---|---|---|---|---|---|---|---|---|---|
| both hands up | #26 | classroom | neutral | sit | F05 | 52 | 0.187 | 0.51 | 0.144 | 0.083 |
| wave | #22 | classroom | [missing] | walk | F01 | 52 | 0.356 | 0.349 | 0.087 | 0.1 |