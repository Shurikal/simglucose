import gymnasium as gym
import unittest
from simglucose.controller.basal_bolus_ctrller import BBController
from collections import namedtuple

Observation = namedtuple('Observation', ['CGM'])
class TestGym(unittest.TestCase):
    def test_gym_random_agent(self):
        from gymnasium.envs.registration import register
        register(
            id='simglucose-adolescent2-v0',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': 'adolescent#002'}
        )

        env = gym.make('simglucose-adolescent2-v0')
        ctrller = BBController()

        reward = 0
        done = False

        observation, info = env.reset()
        for t in range(200):
            env.render()
            print(observation)
            # action = env.action_space.sample()
            obs = Observation(observation["CGM"])

            ctrl_action = ctrller.policy(obs, reward, done, **info)
            action = {"basal": ctrl_action.basal , "bolus" : ctrl_action.bolus}

            observation, reward, done, _, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == '__main__':
    unittest.main()
