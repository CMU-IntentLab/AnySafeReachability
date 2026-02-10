def get_trajectory(self, policy):
    gt_env = Dubins_Env(
        nominal_policy=self.nominal_policy_type, dist_type=self.config.env_dist_type
    )
    unsuccessful = 0
    timeout = 0
    out_of_bounds = 0

    total = 250
    fig, ax = plt.subplots(figsize=(5, 5))
    avg_t = 0
    for idx in tqdm(range(total)):
        gt_env.reset()
        trajectory = []
        obs, __ = self.reset()
        priv_state = self.privileged_state.squeeze()

        while (
            # Check if in bounding box
            priv_state[0] < -1.0
            or priv_state[0] > 1.0
            or priv_state[1] < -1.0
            or priv_state[1] > 1.0
            # Check if already in constraint
            or np.linalg.norm(priv_state[:2] - self.gt_constraint[:2])
            < self.gt_constraint[2]
            or evaluate_V(obs=obs, policy=policy, critic=policy.critic) < 0.1
        ):
            obs, __ = self.reset()
            priv_state = self.privileged_state.squeeze()

        gt_state = torch.tensor(
            [
                priv_state[0],
                priv_state[1],
                np.sin(priv_state[2]),
                np.cos(priv_state[2]),
            ]
        )
        obs_gt, _ = gt_env.reset(initial_state=gt_state.cpu().numpy())
        # TODO: Set constraints of gt_env to the same as self.constraint
        gt_env.constraint = self.gt_constraint.copy()
        done_gt = False

        t = 0

        while not done_gt:  # Rollout trajectory with safety filtering
            trajectory.append(gt_env.state[:2])
            with torch.no_grad():
                V = evaluate_V(obs=obs, policy=policy, critic=policy.critic)
                V = V.squeeze()
            # if V < self.config.safety_filter_eps:
            #     action = find_a(obs=obs, policy=policy)
            # else:
            #     action = self.nominal_policy()

            action = find_a(obs=obs, policy=policy)

            obs, rew, done, _, info = self.step(action)
            obs_gt, rew_gt, done_gt, _, _ = gt_env.step(action)
            if done_gt:
                out_of_bounds += 1

            if rew_gt < 0:
                unsuccessful += 1
                done_gt = True

            if t > 64:
                done_gt = True
                timeout += 1
            t += 1

        avg_t += t

    gt_env.close()
    print("Average time: ", avg_t / total)
    print(
        f"Out of Bounds: {out_of_bounds / total}, Timeout: {timeout / total}, Unsuccessful: {unsuccessful / total}"
    )
    return 1 - unsuccessful / total
