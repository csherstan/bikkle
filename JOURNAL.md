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