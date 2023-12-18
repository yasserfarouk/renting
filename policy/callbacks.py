from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy


class InfoCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        opponent_utillity = episode.last_info_for("__common__")["opponent_utility"]
        for opponent, utility in opponent_utillity.items():
            episode.custom_metrics[opponent] = utility


    # def on_train_result(self, *, algorithm, result: dict, **kwargs):
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True

    #     # Normally, RLlib would aggregate any custom metric into a mean, max and min
    #     # of the given metric.
    #     # For the sake of this example, we will instead compute the variance and mean
    #     # of the pole angle over the evaluation episodes.
    #     pole_angle = result["custom_metrics"]["pole_angle"]
    #     var = np.var(pole_angle)
    #     mean = np.mean(pole_angle)
    #     result["custom_metrics"]["pole_angle_var"] = var
    #     result["custom_metrics"]["pole_angle_mean"] = mean
    #     # We are not interested in these original values
    #     del result["custom_metrics"]["pole_angle"]
    #     del result["custom_metrics"]["num_batches"]