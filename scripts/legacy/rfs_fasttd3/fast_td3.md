In this section, we describe the key design choices made in the development of FastTD3 and their
impact on performance. For details of TD3, we refer the reader to Fujimoto et al. (2018).
Parallel environments Similar to observations in Li et al. (2023a), we find that using massively
parallel environments significantly accelerates TD3 training. We hypothesize that combining deterministic policy gradient algorithms (Silver et al., 2014) with parallel simulation is particularly
effective, because the randomness from parallel environments increases diversity in the data distribution. This enables TD3 to leverage its strength – efficient exploitation of value functions – while
mitigating its weakness in exploration.
Large-batch training We find that using an unusually large batch size of 32,768 for training
the FastTD3 agent is highly effective. We hypothesize that, with massively parallel environments,
large-batch updates provide a more stable learning signal for the critic by ensuring high data diversity
in each gradient update. Otherwise, unless with high update-to-data ratio, a large portion of data will
never be seen by the agent. While increasing the batch size incurs a higher per-update wall-clock
time, it often reduces overall training time due to improved training efficiency.
Distributional RL We also find that using the distributional critic (Bellemare et al., 2017) is helpful
in most cases, similar to the observation of Li et al. (2023b). However, we note that this comes at the
cost of additional hyperparameters – vmin and vmax. Although we empirically find that they are not
particularly difficult to tune1
, one may be able to consider incorporating the reward normalization for
the distributional critic proposed in SimbaV2 (Lee et al., 2025) into FastTD3.
Clipped Double Q-learning (CDQ) While Nauman et al. (2024a) report that using the average of
Q-values rather than the minimum employed in CDQ leads to better performance when combined
with layer normalization, our findings indicate a different trend in the absence of layer normalization.
Specifically, without layer normalization, CDQ remains a critical design choice and using minimum
generally performs better across a range of tasks. This suggests that CDQ continues to be an important
hyperparameter that must be tuned per task to achieve optimal reinforcement learning performance.
Architecture We use an MLP with a descending hidden layer configuration of 1024, 512, and 256
units for the critic, and 512, 256, and 128 units for the actor. We find that using smaller models
1We provide tuned hyperparameters in our open-source implementation.
4
tends to degrade both time-efficiency and sample-efficiency in our experiments. We also experiments
with residual paths and layer normalization (Ba et al., 2016) similar to BRO (Nauman et al., 2024b)
or Simba (Lee et al., 2024), but they tend to slow down training without significant gains in our
experiments. We hypothesize that this is because the data diversity afforded by parallel simulation
and large-batch training reduces the effective off-policyness of updates, thereby mitigating instability
often associated with the deadly triad of bootstrapping, function approximation, and off-policy
learning (Sutton & Barto, 2018). As a result, the training process remains stable even without
additional architectural stabilizers like residual connections or layer normalization.
Exploration noise schedules In contrast to PQL (Li et al., 2023b) which found the effectiveness of
mixed noise – using different Gaussian noise scales for each environment sampled from [σmin, σmax],
we find no significant gains from the mixed noise scheme. Nonetheless, we used mixed noise
schedule, as it allows for flexible noise scheduling with only a few lines of additional code. But we
find that using large σmax = 0.4 is helpful for FastTD3 as shown in Li et al. (2023b).
Update-to-data ratio In contrast to prior work showing that increasing the update-to-data (UTD)
ratio – that is, the number of gradient updates per environment step – typically requires additional
techniques (D’Oro et al., 2023; Schwarzer et al., 2023) or architectural changes (Nauman et al.,
2024b; Lee et al., 2024, 2025), we find that FastTD3 does not require such modifications. Using a
standard 3-layer MLP without normalization, FastTD3 scales favorably with higher UTDs in terms
of sample efficiency. In particular, we find sample-efficiency tends to improve with higher UTDs, but
at the cost of increased wall-time for training. We hypothesize that this is because FastTD3 operates
at extremely low UTDs – typically 2, 4, 8 updates per 128 to 4096 (parallel) environment steps –
reducing the risk of early overfitting often associated with high UTDs.
Replay buffer size Instead of defining a global replay buffer size, we set the size as N ×num_envs
(see Section 2.2 for more details on replay buffer design). In practice, we find that using a larger N
improves performance, th