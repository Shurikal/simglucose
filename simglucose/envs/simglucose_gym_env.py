import gym
from gym import spaces
import numpy as np

from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action

from datetime import datetime

class T1DSimEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

  
    def __init__(self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        self.np_random = np.random.default_rng(seed=seed)

        if patient_name is None:
            patient_name = ['adolescent#001']

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.custom_scenario = custom_scenario

        self.env, _, _, _ = self._create_env()


        self.observation_space = spaces.Dict(
            {
                "CGM": spaces.Box(low=0,high=10000, shape=(1,)),
                "CHO": spaces.Box(low=0,high= 10000, shape=(1,)),
            }
        )

        self.action_space = spaces.Dict(
            {
                "basal": spaces.Box(low=0,high=self.env.pump._params['max_basal'], shape=(1,)),
                "bolus": spaces.Box(low=0,high=self.env.pump._params['max_bolus'], shape=(1,)),
            }
        )

    # todo
    def _get_obs(self):
        CHO = self.env.scenario.get_action(self.env.time).meal
        return {"CGM": np.array([self.env.sensor.measure(self.env.patient)], dtype=np.float32), "CHO": np.array([CHO], dtype=np.float32)}

    # todo
    def _get_info(self):
        return {"time": self.env.time, 
                "meal": self.env.scenario.get_action(self.env.time).meal, 
                "patient_name": self.env.patient.name, 
                "meal": self.env.scenario.get_action(self.env.time).meal,
                "sample_time": self.env.sensor.sample_time}


    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = self.np_random.integers(0, 2**31)
        seed3 = self.np_random.integers(0, 2**31)
        seed4 = self.np_random.integers(0, 2**31)

        hour = self.np_random.integers(0, 24)
        start_time = datetime(2018, 1, 1, hour, 0, 0)    

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)

        if isinstance(self.custom_scenario, list):
           scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = RandomScenario(start_time=start_time, seed=seed3) if self.custom_scenario is None else self.custom_scenario
        
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env, _, _, _ = self._create_env()
        self.env.reset()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    
    def step(self, action):
        act = Action(basal=action["basal"], bolus=action["bolus"])


        if self.reward_fun is None:
            cache = self.env.step(act)
        else:
            cache = self.env.step(act, reward_fun=self.reward_fun)

        return self._get_obs(), cache.reward, cache.done, False, self._get_info()


    def render(self, mode='human', close=False):
        self.env.render(close=close)