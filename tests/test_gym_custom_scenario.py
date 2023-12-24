import gymnasium as gym
import unittest
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.basal_bolus_ctrller import BBController
import datetime as dt
from collections import namedtuple

Observation = namedtuple('Observation', ['CGM'])

class TestCustomScenario(unittest.TestCase):
    def test_custom_scenario(self):
        start_time = dt.datetime(2018, 1, 1, 0, 0, 0)

        meals = [(1,50),(2,10),(3,20)]
        meals_checked = [False for _ in range(len(meals))]

        current_pos = 0

        custom_meal_scenario = CustomScenario(start_time=start_time, scenario=meals)


        gym.envs.register(
            id='env-v0',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': 'adult#001',
                'custom_scenario': custom_meal_scenario}
        )

        env = gym.make('env-v0')
        ctrller = BBController()

        reward = 0
        done = False

        observation , info = env.reset()

        sample_step = info["sample_time"]

        for t in range(61):
            env.render()
            obs = Observation(observation["CGM"])

            ctrl_action = ctrller.policy(obs, reward, done, **info)
            action = {"basal": ctrl_action.basal , "bolus" : ctrl_action.bolus}

            observation, reward, done, _, info = env.step(action)

            if info["meal"] > 0 and (t+1)* int(sample_step) == (meals[current_pos][0]*60):
                print("Meal at time {} with amount {}".format((t+1)*sample_step, info["meal"]))
                meals_checked[current_pos] = True
                current_pos += 1


            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        assert(all(meals_checked))


if __name__ == '__main__':
    unittest.main()