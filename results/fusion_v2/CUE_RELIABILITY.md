# Phase 1 — cue-model reliability (frozen models)

Measured on the **train+val** recorded clips only (858 clips); the test split is untouched. Ground truth = the clip's annotated cue class; prediction = argmax of the deployed cue model's clip-pooled output. Rows where the cue is designed-missing or unobserved are excluded.

## emotion — accuracy 0.882 (95% CI 0.858–0.903, n=806)

| | Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|---|
| **true_Surprise** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **true_Fear** | 0 | 48 | 0 | 0 | 0 | 0 | 0 |
| **true_Disgust** | 0 | 0 | 130 | 0 | 3 | 19 | 0 |
| **true_Happy** | 0 | 0 | 0 | 65 | 0 | 0 | 1 |
| **true_Sad** | 0 | 10 | 2 | 13 | 216 | 2 | 0 |
| **true_Anger** | 0 | 1 | 0 | 0 | 0 | 138 | 0 |
| **true_Neutral** | 1 | 13 | 3 | 2 | 23 | 2 | 114 |

| class | n | recall | precision |
|---|---|---|---|
| Surprise | 0 | nan | 0.000 |
| Fear | 48 | 1.000 | 0.667 |
| Disgust | 152 | 0.855 | 0.963 |
| Happy | 66 | 0.985 | 0.812 |
| Sad | 243 | 0.889 | 0.893 |
| Anger | 139 | 0.993 | 0.857 |
| Neutral | 158 | 0.722 | 0.991 |

**Most-confused distinctions** (true → predicted; these must not be the sole discriminator between two intents):

| true | predicted | n | rate |
|---|---|---|---|
| Neutral | Sad | 23 | 0.146 |
| Disgust | Anger | 19 | 0.125 |
| Sad | Happy | 13 | 0.053 |
| Neutral | Fear | 13 | 0.082 |
| Sad | Fear | 10 | 0.041 |
| Disgust | Sad | 3 | 0.020 |
| Neutral | Disgust | 3 | 0.019 |
| Sad | Disgust | 2 | 0.008 |

**Superclass granularity** {'alarm': ['Fear', 'Surprise'], 'hostile': ['Anger', 'Disgust'], 'low': ['Sad'], 'positive': ['Happy'], 'neutral': ['Neutral']}

- accuracy **0.906** (95% CI 0.884–0.924) vs 0.882 fine-grained → **+0.024**

| merge | classes | accuracy gain |
|---|---|---|
| hostile | Anger+Disgust | +0.024 |
| alarm | Fear+Surprise | +0.000 |

## gesture — accuracy 0.999 (95% CI 0.993–1.000, n=813)

| | idle | wave | point | thumbs_down | beckoning | raise_hand | both_hands_up |
|---|---|---|---|---|---|---|---|
| **true_idle** | 256 | 0 | 0 | 0 | 0 | 0 | 0 |
| **true_wave** | 0 | 149 | 0 | 0 | 0 | 0 | 0 |
| **true_point** | 0 | 0 | 96 | 0 | 0 | 0 | 0 |
| **true_thumbs_down** | 1 | 0 | 0 | 170 | 0 | 0 | 0 |
| **true_beckoning** | 0 | 0 | 0 | 0 | 51 | 0 | 0 |
| **true_raise_hand** | 0 | 0 | 0 | 0 | 0 | 42 | 0 |
| **true_both_hands_up** | 0 | 0 | 0 | 0 | 0 | 0 | 48 |

| class | n | recall | precision |
|---|---|---|---|
| idle | 256 | 1.000 | 0.996 |
| wave | 149 | 1.000 | 1.000 |
| point | 96 | 1.000 | 1.000 |
| thumbs_down | 171 | 0.994 | 1.000 |
| beckoning | 51 | 1.000 | 1.000 |
| raise_hand | 42 | 1.000 | 1.000 |
| both_hands_up | 48 | 1.000 | 1.000 |

**Most-confused distinctions** (true → predicted; these must not be the sole discriminator between two intents):

| true | predicted | n | rate |
|---|---|---|---|
| thumbs_down | idle | 1 | 0.006 |

**Superclass granularity** {'summon': ['beckoning', 'raise_hand'], 'greet': ['wave'], 'deictic': ['point'], 'appraise_pos': ['thumbs_up'], 'appraise_neg': ['thumbs_down'], 'alarm_pose': ['both_hands_up'], 'none': ['idle']}

- accuracy **0.999** (95% CI 0.993–1.000) vs 0.999 fine-grained → **+0.000**

| merge | classes | accuracy gain |
|---|---|---|
| summon | beckoning+raise_hand | +0.000 |

## motion — accuracy 0.435 (95% CI 0.402–0.468, n=858)

| | sitting | standing | walking | stepping_back |
|---|---|---|---|---|
| **true_sitting** | 165 | 24 | 47 | 0 |
| **true_standing** | 38 | 88 | 165 | 84 |
| **true_walking** | 9 | 26 | 105 | 6 |
| **true_stepping_back** | 1 | 54 | 31 | 15 |

| class | n | recall | precision |
|---|---|---|---|
| sitting | 236 | 0.699 | 0.775 |
| standing | 375 | 0.235 | 0.458 |
| walking | 146 | 0.719 | 0.302 |
| stepping_back | 101 | 0.149 | 0.143 |

**Most-confused distinctions** (true → predicted; these must not be the sole discriminator between two intents):

| true | predicted | n | rate |
|---|---|---|---|
| standing | walking | 165 | 0.440 |
| standing | stepping_back | 84 | 0.224 |
| stepping_back | standing | 54 | 0.535 |
| sitting | walking | 47 | 0.199 |
| standing | sitting | 38 | 0.101 |
| stepping_back | walking | 31 | 0.307 |
| walking | standing | 26 | 0.178 |
| sitting | standing | 24 | 0.102 |

**Superclass granularity** {'static': ['sitting', 'standing'], 'approach': ['walking'], 'retreat': ['stepping_back']}

- accuracy **0.507** (95% CI 0.474–0.540) vs 0.435 fine-grained → **+0.072**

| merge | classes | accuracy gain |
|---|---|---|
| static | sitting+standing | +0.072 |

## context — accuracy 0.972 (95% CI 0.959–0.981, n=858)

| | classroom | kitchen |
|---|---|---|
| **true_classroom** | 338 | 0 |
| **true_kitchen** | 24 | 496 |

| class | n | recall | precision |
|---|---|---|---|
| classroom | 338 | 1.000 | 0.934 |
| kitchen | 520 | 0.954 | 1.000 |

**Most-confused distinctions** (true → predicted; these must not be the sole discriminator between two intents):

| true | predicted | n | rate |
|---|---|---|---|
| kitchen | classroom | 24 | 0.046 |

**Superclass granularity** {'classroom': ['classroom'], 'kitchen': ['kitchen']}

- accuracy **0.972** (95% CI 0.959–0.981) vs 0.972 fine-grained → **+0.000**

## Design implications (plan P1/P2)

Distinctions listed under *most-confused* above are the ones an intent boundary must not rest on. Merges with a large accuracy gain are the superclasses the new rubric should key on instead.
