python leanrl_ppo_selfattention.py \
--env-id BikkleFakeEyeTracking-v0 \
--total-timesteps 2000000 \
--compile \
--policy-params.base-params.num-layers 1 \
--policy-params.base-params.token-size 8 \
--policy-params.mlp-hidden-size 16 \
--policy-params.activation relu \
--value-params.base-params.num-layers 1 \
--value-params.base-params.token-size 8 \
--value-params.mlp-hidden-size 16 \
--value-params.activation relu \

