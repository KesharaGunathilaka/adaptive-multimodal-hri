# Policy demo — deployed fusion + intent->action layer (test subjects)

tau=0.5, tau_emergency=0.3

| scenario | intent | top actions (window share) | fallback% |
|---|---|---|---|
| S05_F02 | F07 | A08 97%, A14 3% | 0% |
| S06_F08 | F08 | A07 100% | 0% |
| S08_F06 | F06 | A11 79%, A10 7%, A12 6% | 6% |
| S09_F02 | F02 | A14 99%, A08 1% | 0% |
| S18_F01 | F01 | A01 100% | 0% |
| S19_F02 | F02 | A02 100% | 0% |
| S24_F07 | F07 | A11 58%, A02 33%, A08 8% | 0% |
| S25_F09 | F09 | A10 48%, A04 48%, A06 4% | 4% |
| S26_F02 | F02 | A02 89%, A06 7%, A10 4% | 1% |
| S29_F03 | F03 | A04 93%, A06 6%, A10 1% | 0% |

## F02 emergency safety check

- window-level F02 recall: **0.968** (278 windows)
- clip-level 'any emergency fired': **1.000** (22/22 clips)
- false-emergency rate on non-F02 windows: **0.013**
