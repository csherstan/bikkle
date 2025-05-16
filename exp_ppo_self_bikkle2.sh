python leanrl_ppo_selfattention.py \
--env-id BikkleFakeEyeTracking-v0 \
--total-timesteps 2000000 \
--compile \
--policy-params.base-params.num-layers 2 \
--policy-params.base-params.token-size 16 \
--policy-params.mlp-hidden-size 64 \
--policy-params.activation gelu \
--value-params.base-params.num-layers 2 \
--value-params.base-params.token-size 16 \
--value-params.mlp-hidden-size 64 \
--value-params.activation gelu \

