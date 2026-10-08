# Intent annotation - instructions

You will watch 200 short video clips. In each one a person is
interacting with (or ignoring) a robot in a room.

**Your task:** for each clip, decide what the person most likely WANTS
from the robot, and write the matching code in `responses_<you>.csv`.

Please:

- Watch each clip once, at normal speed. Trust your first impression.
- Judge only what you SEE. There is no audio and no context beyond the clip.
- Work through them in the order given; do not go back and revise.
- Do not discuss clips with the other annotators until everyone is finished.
- If you genuinely cannot tell, write `unclear` rather than guessing.
- Also give a confidence from 1 (guessing) to 5 (certain).

## The intent codes

| code | short name | what it means |
|---|---|---|
| **F01** | Greeting or positive acknowledgement | The person is greeting the robot, saying hello or goodbye, or signalling approval / that something went well. |
| **F02** | Emergency or danger | Something is wrong or unsafe - the person is alarmed, startled, panicking, or recoiling from a hazard. The robot should treat this as urgent. |
| **F03** | Asking the robot to do a task | The person wants the robot to come over, fetch something, or attend to an object or place. A request directed outward, at a task. |
| **F04** | Asking for help | The person is struggling or in difficulty and wants assistance for THEMSELVES. Distress or being stuck, rather than delegating a task. |
| **F05** | Busy - does not want to be interrupted | The person is absorbed in their own activity, or dealing with someone else. The robot should stay out of the way. |
| **F06** | Wants the robot to move / give space | The person needs to get past, or is waving the robot away. About physical space, not about a task. |
| **F07** | Frustration or agitation | The person is annoyed, irritated or venting - but nothing is physically dangerous. The robot should calm things down. |
| **F08** | Wants a break or relief | The person is fed up with, or repulsed by, the current activity and wants it to stop or change. |
| **F10** | Discouraged / has given up | The person has disengaged - low energy, defeated, no longer trying. They are not asking for anything directly. |
| **unclear** | Cannot tell | The clip does not let you decide. |

## Distinctions people find tricky

- **F03 vs F04** - both are requests. F03 asks the robot to do something
  about a TASK or object; F04 asks for help for the PERSON themselves.
- **F05 vs F10** - both mean 'not asking for anything'. F05 is busy and
  engaged; F10 is defeated and has given up.
- **F06 vs F03** - F06 is about physical space (move aside); F03 is about
  a task (come and do this).
- **F07 vs F08** - F07 is annoyance at the situation; F08 is wanting the
  current activity to stop.
- **F02** is only for genuine urgency or danger, not ordinary annoyance.

## Files

- `videos/` - the clips, named `clip_001.mp4` and so on.
- `responses_annotator_A.csv` - your answer sheet (use YOUR letter).

Thank you - this takes about 60-90 minutes.
