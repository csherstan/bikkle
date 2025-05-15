2025-05-15

- I have wired in an approach that freezes the rest of the network and otherwise only trains
the eyetracking embedding and tokenizer
- I also created an ObservationWrapper with fake eyetracking data.
- I am attempting to train by continuing from an existing model: https://wandb.ai/csherstan-team/ppo_continuous_action/runs/e0gcsfz5

2025-05-12

Overnight 2 runs crashed, it looks like they crashed at the same wallclock time. I'm thinking there is a GPU
issue - it may have overheated.

- The CNN based run didn't show signs of learning the problem.
- The self-attention run looked like it was learning well actually: https://wandb.ai/csherstan-team/ppo_continuous_action/runs/oh278s94
  - I do note that the mean_entropy_loss was pinned, I'm not sure what that would mean.

 Best self-attention model so far: https://wandb.ai/csherstan-team/ppo_continuous_action/runs/5asd1v8c

 Trying for a smaller self-attention model: https://wandb.ai/csherstan-team/ppo_continuous_action/runs/7mytmpxd
 - This one looks very similar to oh278s94, maybe slightly better.

 Tried with `num_envs=4` https://wandb.ai/csherstan-team/ppo_continuous_action/runs/6o2dr3jo. This one is not as good
   - I suspect I may need to change some other hyperparams such as the batch size.
   - The learning curve is slower, but consistently upward. It looks pretty nice.

Ok, now I need to integrate this with the eyetracking data. The plan is to take a pretrained model, then train with
human eyetracking data. I think I will try to run only one environment at a time, so I don't need to stack.
When training the model we will only optimize over the linear embedding and the offset embedding.
- modify the PPO training script to take in a trained model
- set the optimizer to only consider the params of interest
- create some sort of modifier for the environment


2025-05-11

Thinking about this more.
- I could try doing some sort of human training + behavior cloning from existing policy for PPO
training. This might not work very well though.
- From a personal perspective, I really want to at least get the self-attention model working.
- ChatGPT suggest that the self-attention model would train better with SAC than PPO.

2 things I want to do:
1. finish the integration of human data collection with model training.
2. Get the self-attention model working. I really want to figure this out. I'm going to change my
approach though:
 - Use an observation wrapper as the preprocessor.
 - Use a single encoder for all the token types. Really, I should just remove the blue tokens.

Another idea that James suggested was to use a semantic image and then just apply a CNN.
**how should I initialize the self-attention?**
**where should I apply LayerNorm?**

2025-05-10

- I've been working on this a ton, but I keep hitting walls on getting the basic system working - I haven't even gotten
to the difficult part here. At present it works if I use PPO, terminate the episode when a token is touched, ensure
that I use reward normalization, and use an MLP.
- If I use SAC it's just not working. I
- Adapting PPO to work with the self-attention model has been a pain. Still not working.
- Is there a way I could simplify further?
	- Don't use a self-attention model?
	- Use a well known environment?
- I have PPO running with the self-attention model running, but not working.
- If I try using cudagraphs with PPO it doesn't work -> throws an error.

Next steps:
- Simplify the environment further. Drop cyan completely.
- Οrder the tokens by how close they are. This seems to have helped the PPO case significantly.
- Drop the self-attention approach completely. Instead just set up to use the MLP approach with a place holder value
to indicate that eyetracking is not available.
- log losses for PPO.

Here is a baseline PPO run: https://wandb.ai/csherstan-team/ppo_continuous_action/runs/ywexn8i3

SAC:
- reduce gamma to 0.5
  - Check this one out: https://wandb.ai/csherstan-team/sac_continuous_action/runs/8f7jpgzh
- n-step returns for S
- cascade architecture




2025-05-06
- I *think* the overall algorithm is working, but it doesn't seem very efficient in terms of samples. I can see the agent is sort of able to learn to get some pellets, but
it's not consistent and doesn't seem to differentiate between blue and pink.
- Next:
 - [x] Double-check reward structure
 - [x] Add event based replay buffer
 - Try speeding up the learning using compile, etc.
 - Try multiple envs per step.
 - Add another self-attention block
 - Rework preprocess_bikkle_observation_with_mask

- reward changes: actually, there's no relation to the timer in the reward structure, right now it's just a noise
observation. Corrected
- I keep having issues with cuda. Things often get into a weird state where any use of CUDA fails. I wonder if this
has to do with overheating or something.
  - I have not been able to resolve this. I tried upgrading my nvidia drivers, reinstalling cuda 12.8 and installing torch with cuda support. I think I should try the baseline leanrl files and see if that works.
  - my current baseline on CPU with compile is 62 it/s... actually 7? It briefly spiked, not sure what's going on.
  - It's not obvious that compile is helping at all.
- action changes: just assume we always want to move at a constant rate and just control the direction.
	Undecided:
	Let's change the action space. Instead of controlling the x-y movement, we assume the agent always moves at full speed: a vector of magnitude 1. The action space is only 1 element that varies from 0 to 1 which gets converted between -180 degrees and 180 degrees. So the action describes the direction in which the movement vector faces.



2025-05-05

- Change env to produce variable numbers of each block type. I'm trying to figure out how the data is being handled
 in the replay buffer. When the data comes out of the replay buffer it is already in tensor format. Actually, do I
 really need to support this right now? I think I can just use the same number of each block type for now.
 I can always add this later.
- The system runs but the agent learns to very quickly just push into the wall -> the actions are maxed.
	- Once it gets into this condition it's not seeing alternative experiences in the replay buffer.
	- Change training so that the env gets reset at some interval.
	- Record average action -> we expect this to be zero
	- Add secondary replay buffer just for non-zero reward experiences. Hmm, to do this correctly we would need to keep
	some history as well... not sure I want to do that.

	AI suggests:

	Yes, there are several potential causes for the behavior you're observing, aside from gradient issues. Here are some possibilities to investigate:


Improper Initialization of Weights:
If the weights of your neural network layers are not initialized properly, it can lead to unstable training and outputs that quickly saturate.

**Clamped Log Standard Deviation**:
In your BikklePolicy class, the log_std is clamped between -5 and 2. If this range is too restrictive, it might cause the policy to favor extreme actions. Consider whether this range is appropriate for your problem.

**Action Scaling or Tanh Saturation**:

The torch.tanh function in get_action can saturate near its bounds (-1 and 1), leading to gradients vanishing for extreme values. This could cause the model to favor extreme actions.

**Reward Scaling**:

If the rewards in your environment are not properly scaled, the policy might learn to maximize extreme actions to achieve higher rewards.

**Improper Loss Function**:
If the loss function for training the policy is not correctly implemented or balanced, it might encourage extreme actions.

**Key Padding Mask in Attention**:

In the BaseBikkleModel, the key_padding_mask is passed to the self-attention layer. If this mask is not correctly set, it could lead to incorrect attention weights and unstable outputs.

**Observation Preprocessing**:
The preprocess_bikkle_observation_with_mask function applies padding and masking. If there are errors in how observations are preprocessed, it could lead to incorrect inputs to the model.

**Learning Rate**:
A learning rate that is too high can cause the model to overshoot optimal solutions, leading to extreme outputs.

**Gradient Clipping**:

If gradients are not clipped, they might explode during training, causing the model to produce extreme outputs.

**Action Space Mismatch**:

Ensure that the action space of your environment matches the expected output of the policy. If there is a mismatch, the policy might produce invalid or extreme actions.

**Exploration Noise**:


If you're adding noise for exploration (e.g., in get_action), ensure that the noise level is appropriate and not causing the policy to favor extreme actions.
Overfitting:


If the model is overfitting to a specific set of observations, it might produce extreme actions for those cases.


- At some point today the code just stopped working on the GPU. A hard reboot of the computer resolved the issue.

- Expected before of the action metrics:
```- the ideal action would always be full speed and we would just control the direction. Actually maybe I should just change
 the action space to be an angle of travel. It might be easier to debug.