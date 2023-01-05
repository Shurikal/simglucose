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

  
    def __init__(self,  patient_name=None, 
                        custom_scenario=None, 
                        reward_fun=None, 
                        seed=None, 
                        history_length=1, 
                        enable_bolus=False,
                        enable_meal=False,
                        enable_insulin_history=True):
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

        self.enable_bolus = enable_bolus

        self.t1dsimenv, _, _, _ = self._create_env()

        self.history_length = history_length

        self.CGM_hist = [0] * history_length
        self.CHO_hist = [0] * history_length
        self.insulin_hist = [0] * history_length

        self.observation_space = spaces.Dict(
            {
                "CGM": spaces.Box(low=0,high=10000, shape=(history_length,)),
            }
        )
        self.enable_meal = enable_meal
        if enable_meal:
            self.observation_space["CHO"] = spaces.Box(low=0,high=10000, shape=(history_length,))

        self.enable_insulin_history = enable_insulin_history
        if enable_insulin_history:
            self.observation_space["insulin"] = spaces.Box(low=0,high=10000, shape=(history_length,))

        self.action_space = spaces.Dict(
            {
                "basal": spaces.Box(low=0,high=self.t1dsimenv.pump._params['max_basal'], shape=()),
            }
        )

        if enable_bolus:
            self.action_space["bolus"] = spaces.Box(low=0,high=self.t1dsimenv.pump._params['max_bolus'], shape=())

    def _get_obs(self):
        cache = {"CGM": np.array(self.CGM_hist, dtype=np.float32)}
        if self.enable_meal:
            cache["CHO"] =  np.array(self.CHO_hist, dtype=np.float32)
        if self.enable_insulin_history:
            cache["insulin"] =  np.array(self.insulin_hist, dtype=np.float32)

        return cache
 
    # todo match time
    def _get_info(self):
        return {"time": self.t1dsimenv.time, 
                "meal": self.t1dsimenv.scenario.get_action(self.t1dsimenv.time).meal, 
                "patient_name": self.t1dsimenv.patient.name, 
                "sample_time": self.t1dsimenv.sensor.sample_time}


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

        self.t1dsimenv, _, _, _ = self._create_env()
        cache = self.t1dsimenv.reset()

        self.CGM_hist = [cache.observation.CGM] * self.history_length
        self.insulin_hist = [0] * self.history_length
        self.CHO_hist = [0] * self.history_length

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    
    def step(self, action):
        insulin = 0
        if self.enable_bolus:
            act = Action(basal=action["basal"], bolus=action["bolus"])
            insulin = action["bolus"] + action["basal"]
        else:
            act = Action(basal=action["basal"], bolus=0)
            insulin = action["basal"]

        if self.reward_fun is None:
            cache = self.t1dsimenv.step(act)
        else:
            cache = self.t1dsimenv.step(act, reward_fun=self.reward_fun)

        self.CGM_hist = self.CGM_hist[1:] + [cache.observation.CGM]
        self.CHO_hist = self.CHO_hist[1:] + [cache.info["meal"]]
        self.insulin_hist = self.insulin_hist[1:] + [insulin]

        return self._get_obs(), cache.reward, cache.done, False, self._get_info()


    def render(self, mode='human', close=False):
        self.t1dsimenv.render(close=close)

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed=seed)
        seed1 = self.np_random.integers(0, 2**31)
        self.t1dsimenv, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def get_t1dsimenv(self):
        return self.t1dsimenv


class T1DSimEnvDiscrete(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

  
    def __init__(self,  patient_name=None, 
                        custom_scenario=None, 
                        reward_fun=None, 
                        seed=None, 
                        history_length=1, 
                        enable_bolus=False,
                        enable_meal=False,
                        enable_insulin_history=True):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        self.np_random = np.random.default_rng(seed=seed)

        self.insulin_rate = 0

        if patient_name is None:
            patient_name = ['adolescent#001']

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.custom_scenario = custom_scenario

        self.enable_bolus = enable_bolus

        self.t1dsimenv, _, _, _ = self._create_env()

        self.history_length = history_length

        self.CGM_hist = [0] * history_length
        self.CHO_hist = [0] * history_length
        self.insulin_hist = [0] * history_length

        self.observation_space = spaces.Dict(
            {
                "CGM": spaces.Box(low=0,high=10000, shape=(history_length,)),
            }
        )
        self.enable_meal = enable_meal
        if enable_meal:
            self.observation_space["CHO"] = spaces.Box(low=0,high=10000, shape=(history_length,))

        self.enable_insulin_history = enable_insulin_history
        if enable_insulin_history:
            self.observation_space["insulin"] = spaces.Box(low=0,high=10000, shape=(history_length,))

        self.action_space = spaces.Discrete(5, start=0)

        # bolus is not implemented

    def _get_obs(self):
        cache = {"CGM": np.array(self.CGM_hist, dtype=np.float32)}
        if self.enable_meal:
            cache["CHO"] =  np.array(self.CHO_hist, dtype=np.float32)
        if self.enable_insulin_history:
            cache["insulin"] =  np.array(self.insulin_hist, dtype=np.float32)

        return cache
 
    # todo match time
    def _get_info(self):
        return {"time": self.t1dsimenv.time, 
                "meal": self.t1dsimenv.scenario.get_action(self.t1dsimenv.time).meal, 
                "patient_name": self.t1dsimenv.patient.name, 
                "sample_time": self.t1dsimenv.sensor.sample_time}


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

        self.t1dsimenv, _, _, _ = self._create_env()
        cache = self.t1dsimenv.reset()

        self.CGM_hist = [cache.observation.CGM] * self.history_length
        self.insulin_hist = [0] * self.history_length
        self.CHO_hist = [0] * self.history_length

        self.basal_rate = self.t1dsimenv.patient._params['u2ss'] * self.t1dsimenv.patient._params['BW'] / 6000

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    
    def step(self, action):
        action *= 0.5 # 0-4 -> 0-2
        insulin_rate = self.basal_rate * action

        act = Action(basal=insulin_rate, bolus=0)

        if self.reward_fun is None:
            cache = self.t1dsimenv.step(act)
        else:
            cache = self.t1dsimenv.step(act, reward_fun=self.reward_fun)

        self.CGM_hist = self.CGM_hist[1:] + [cache.observation.CGM]
        self.CHO_hist = self.CHO_hist[1:] + [cache.info["meal"]]
        self.insulin_hist = self.insulin_hist[1:] + [insulin_rate]

        return self._get_obs(), cache.reward, cache.done, False, self._get_info()


    def render(self, mode='human', close=False):
        self.t1dsimenv.render(close=close)

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed=seed)
        seed1 = self.np_random.integers(0, 2**31)
        self.t1dsimenv, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def get_t1dsimenv(self):
        return self.t1dsimenv