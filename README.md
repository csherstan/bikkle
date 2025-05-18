Current python version: 3.10.16
`pip install -r requirements.txt`

Train a model (6.5 hours on my computer)

```
sh ./exp_ppo_self_bikkle2.sh
```

---

This project was implemented as a first attempt at "vibe coding". This is the idea of doing most of the coding through
an LLM agent. Also, my focus was more on speed of development rather than production quality.
Thus, the overall layout of the project is not overly clean.
When I wanted to make some changes, such as using a different model, I just copy-pasted rather than trying to implement
a more general purpose system. In particular, I generally prefer a system closer to what I've implemented in other
projects such as the experiment file
https://github.com/csherstan/triangles/blob/main/scripts/experiments/gym_envs/mountain_car.py

The primary files to consider are:

- `leanrl_ppo_selfattention.py`: this is the main script for training the RL agent
- `drive_eye_tracking.py`: drive the agent around the arena using arrow keys while
displaying the eye tracking data
- `env.py`: All of the various definitions for the environment and the various environment wrappers I used to work with
it.
- `model.py`: All of the various models I experimented with
- `eval_selfattention.py`: Loads models from the checkpoint file(s) and runs evals in 3 different settings:
  1. `fake`: Uses fake eye tracking data that was used during training. This is just Gaussian noise centered on the high
  value block.
  2. `fake-uniform`: fake eye tracking data is provided, but it is just uniform noise sampled over the whole arena. Not
  correlated with the high value block.
  3. `no_eye_tracking`: No eye tracking tokens are provided.
- `view_eval_selfattention.py`: Used for human driven interaction with actual eye tracking data given a trained model.
