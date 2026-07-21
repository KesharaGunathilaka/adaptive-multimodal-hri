# G3 spotlight — same gesture, different intent

Model: jetson_deploy/fusion/fusion_attn.pt (attn exclude, deployed).

| family | case | cues | expected | predicted | agreement | source |
|---|---|---|---|---|---|---|
| raise hand | S01_F04 | classroom + neutral + sit | F04 | F04 (18/18 clips) | 1.00 | real (val subjects) |
| raise hand | S11_F05 | classroom + happy + sit | F05 | F05 (52/54 clips) | 0.96 | real (train subjects!) |
| thumbs down | S04_F04 | classroom + sad + sit | F04 | F04 (59/61 clips) | 0.97 | real (train subjects!) |
| thumbs down | V3 #15 | classroom + angry + sit | F07 | F07 (196/200 samples) | 0.98 | synthetic |
| thumbs down | V3 #17 | classroom + disgust + sit | F08 | F08 (200/200 samples) | 1.00 | synthetic |
| thumbs down | V3 #33 | kitchen + happy + stand | F01 | F01 (200/200 samples) | 1.00 | synthetic |
| thumbs down | S21_F04 | kitchen + sad + sit | F04 | F04 (2/2 clips) | 1.00 | real (val subjects) |
| wave | S02_F01 | classroom + happy + walk | F01 | F01 (2/2 clips) | 1.00 | real (val subjects) |
| wave | S25_F09 | kitchen + happy + walk (exit) | F09 | F09 (2/3 clips) | 0.67 | real (test subjects) |
| wave | V3 #13 | classroom + angry + stand | F06 | F06 (200/200 samples) | 1.00 | synthetic |
| both hands up | S09_F02 | classroom + surprise + stand | F02 | F02 (12/12 clips) | 1.00 | real (test subjects) |
| both hands up | S05_F02 | classroom + angry + stand | F07 | F07 (14/14 clips) | 1.00 | real (test subjects) |
| both hands up | S24_F07 | kitchen + angry + stand | F07 | F06 (1/2 clips) | 0.50 | real (test subjects) |
| both hands up | S19_F02 | kitchen + fear + step back | F02 | F02 (3/3 clips) | 1.00 | real (test subjects) |
