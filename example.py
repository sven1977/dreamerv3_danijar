import crafter
import gym
import numpy as np
import warnings

import dreamerv3
from dreamerv3 import embodied
from embodied.envs import from_gym
from embodied.envs.atari import Atari
from train import make_env


class DebugImgEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(0.0, 1.0, (6,), dtype=np.int32)
        self._ts = 0

    def reset(self, *, seed=None, options=None):
        self._ts = 0
        return self._obs()

    def step(self, action):
        self._ts += 1
        return self._obs(), self._ts * 0.1, self._ts >= 10, {}

    def _obs(self):
        obs = np.full(
            shape=self.observation_space.shape,
            fill_value=self._ts,
            dtype=self.observation_space.dtype,
        )
        obs[0][0][0] = self._ts
        return obs


class DiscreteActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = gym.spaces.Box(
            0, 2, (env.action_space.n,), dtype=np.int32
        )

    def step(self, action):
        action = np.argmax(action)
        return super().step(action)


class OneHot(gym.ObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(self.observation_space.n,), dtype=np.float32
        )

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        return self._get_obs(ret)

    def step(self, action):
        ret = self.env.step(action)
        return self._get_obs(ret[0]), ret[1], ret[2], ret[3]

    def _get_obs(self, obs):
        ret = np.zeros(shape=(self.observation_space.shape[0],), dtype=np.float32)
        ret[obs] = 1.0
        return ret


def main():

  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  # Always use the defaults.
  config = embodied.Config(dreamerv3.configs['defaults'])

  # Enable one of these for Atari100k or CartPole-v1
  #config = config.update(dreamerv3.configs['atari100k'])
  config = config.update(dreamerv3.configs['cartpole'])

  config = config.update(dreamerv3.configs['xsmall'])
  config = config.update({
      #'task': 'atari_pong',
      'logdir': '~/logdir/run1',
      #'run.train_ratio': 64,
      #'run.log_every': 30,  # Seconds
      #'batch_size': 16,
      'jax.prealloc': False,
      #'encoder.mlp_keys': '$^',
      #'decoder.mlp_keys': '$^',
      #'encoder.cnn_keys': 'image',
      #'decoder.cnn_keys': 'image',
      #'encoder.mlp_keys': '.*',
      #'decoder.mlp_keys': '.*',
      #'encoder.cnn_keys': '$^',
      #'decoder.cnn_keys': '$^',
      # 'jax.platform': 'cpu',
  })
  # Debug: ON
  config = config.update(dreamerv3.configs['debug'])

  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  #env = crafter.Env()  # Replace this with your Gym env.
  #env = gym.make("CartPole-v1")

  # Debug
  #gym.register(id="MyEnv-v1", entry_point=DebugImgEnv)
  #env = from_gym.FromGym("MyEnv-v1")

  # CartPole: Use with dreamerv3.configs['cartpole']
  env = from_gym.FromGym(DiscreteActionWrapper(gym.make("CartPole-v1")))
  # FrozenLake-v1
  #env = DiscreteActionWrapper(OneHot(
  #    gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
  #))
  #env = from_gym.FromGym(env)

  # Atari100k: Use with dreamerv3.configs['atari100k']
  #env = make_env(config)

  #env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
