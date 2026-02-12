from PyHJ.data import Batch


def find_a(obs, policy):
    # assert isinstance(obs, dict), "Observation must be a dictionary."
    if isinstance(obs, dict):
        if obs["state"].ndim == 1:  # Unbatched observation
            tmp_obs = {k: v.reshape((1,) + v.shape) for k, v in obs.items()}
        else:  # Batched observation
            tmp_obs = obs
    else:
        if obs.ndim == 1:  # Unbatched observation
            obs = obs.reshape((1,) + obs.shape)
        tmp_obs = obs
    tmp_batch = Batch(obs=tmp_obs, info=Batch())
    tmp = policy(tmp_batch, model="actor_old").act
    # act = policy.map_action(tmp).cpu().detach().numpy().flatten()
    act = policy.map_action(tmp).flatten()
    return act


def evaluate_V(obs, policy, critic):
    # assert isinstance(obs, dict), "Observation must be a dictionary."
    if isinstance(obs, dict):
        if obs["state"].ndim == 1:  # Unbatched observation
            tmp_obs = {k: v.reshape((1,) + v.shape) for k, v in obs.items()}
        else:  # Batched observation
            tmp_obs = obs
    else:
        if obs.ndim == 1:  # Unbatched observation
            obs = obs.reshape((1,) + obs.shape)
        tmp_obs = obs
    tmp_batch = Batch(obs=tmp_obs, info=Batch())
    tmp = critic(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()
