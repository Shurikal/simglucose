import gym
from gym import spaces
import numpy as np

from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action

from simglucose.controller.basal_bolus_ctrller import CONTROL_QUEST

import pandas as pd

from datetime import datetime
import gymnasium


class T1DSimEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    metadata = {"render.modes": ["human"]}

  
    def __init__(self,  patient_name=None, 
                        custom_scenario=None, 
                        reward_fun=None, 
                        seed=None, 
                        history_length=1, 
                        enable_manual_bolus=False,
                        enable_meal=False,
                        enable_insulin_history=True):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        or ['adolescent#001', 'adolescent#002', ...]
        '''
        self.np_random = np.random.default_rng(seed=seed)

        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.custom_scenario = custom_scenario

        self.enable_manual_bolus = enable_manual_bolus

        self.t1dsimenv, _, _, _ = self._create_env()

        self.history_length = history_length

        self.CGM_hist = [0] * history_length
        self.CHO_hist = [0] * history_length
        self.insulin_hist = [0] * history_length

        # Default observation space
        self.observation_space = spaces.Dict(
            {
                "CGM": spaces.Box(low=0,high=10000, shape=(history_length,)),
            }
        )

        # Add meal observation space
        self.enable_meal = enable_meal
        if enable_meal:
            self.observation_space["CHO"] = spaces.Box(low=0,high=10000, shape=(history_length,))

        # Add insulin observation space
        self.enable_insulin_history = enable_insulin_history
        if enable_insulin_history:
            self.observation_space["insulin"] = spaces.Box(low=0,high=10000, shape=(history_length,))

        # Default action space
        self.action_space = spaces.Dict(
            {
                "basal": spaces.Box(low=0,high=self.t1dsimenv.pump._params['max_basal'], shape=()),
            }
        )

        # Add bolus action space
        if enable_manual_bolus:
            self.action_space["bolus"] = spaces.Box(low=0,high=self.t1dsimenv.pump._params['max_bolus'], shape=())

    def _get_obs(self):
        cache = {"CGM": np.array(self.CGM_hist, dtype=np.float32)}
        if self.enable_meal:
            cache["CHO"] =  np.array(self.CHO_hist, dtype=np.float32)
        if self.enable_insulin_history:
            cache["insulin"] =  np.array(self.insulin_hist, dtype=np.float32)

        return cache
 
    def _get_info(self):
        return {"time": self.t1dsimenv.current_time, 
                "meal": self.t1dsimenv.scenario.get_action(self.t1dsimenv.current_time).meal, 
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
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t1dsimenv, _, _, _ = self._create_env()
        cache = self.t1dsimenv.reset()

        # Initialize history buffer
        self.CGM_hist = [cache.observation.CGM] * self.history_length
        self.insulin_hist = [0] * self.history_length
        self.CHO_hist = [0] * self.history_length

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    
    def step(self, action):
        insulin = 0
        if self.enable_manual_bolus:
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
                        enable_manual_bolus=False,
                        enable_auto_bolus=True,
                        enable_meal=False,
                        enable_insulin_history=True,
                        action_space_size=5):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        or ['adolescent#001', 'adolescent#002', ...]
        '''
        self.np_random = np.random.default_rng(seed=seed)

        self.quest = pd.read_csv(CONTROL_QUEST)
        self.action_space_size = action_space_size
        self.enable_auto_bolus = enable_auto_bolus

        self.insulin_rate = 0

        if patient_name is None:
            patient_name = ['adolescent#001']

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.custom_scenario = custom_scenario

        self.enable_manual_bolus = enable_manual_bolus

        self.t1dsimenv, _, _, _ = self._create_env()

        self.history_length = history_length

        self.CGM_hist = [0] * history_length
        self.CHO_hist = [0] * history_length
        self.insulin_hist = [0] * history_length

        # Default observation space
        self.observation_space = spaces.Dict(
            {
                "CGM": spaces.Box(low=0,high=10000, shape=(history_length,)),
            }
        )

        # Add meal observation space
        self.enable_meal = enable_meal
        if enable_meal:
            self.observation_space["CHO"] = spaces.Box(low=0,high=10000, shape=(history_length,))

        # Add insulin observation space
        self.enable_insulin_history = enable_insulin_history
        if enable_insulin_history:
            self.observation_space["insulin"] = spaces.Box(low=0,high=10000, shape=(history_length,))

        # Default action space
        self.action_space = spaces.Discrete(self.action_space_size, start=0)

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
        return {"time": self.t1dsimenv.current_time, 
                "meal": self.t1dsimenv.scenario.get_action(self.t1dsimenv.current_time).meal, 
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

        # Initialize history buffer
        self.CGM_hist = [cache.observation.CGM] * self.history_length
        self.insulin_hist = [0] * self.history_length
        self.CHO_hist = [0] * self.history_length

        # Calculate default basal rate
        self.basal_rate = self.t1dsimenv.patient._params['u2ss'] * self.t1dsimenv.patient._params['BW'] / 6000


        # Set params for bolus injection
        if any(self.quest.Name.str.match(self.t1dsimenv.patient.name)):
            # Import params from patient questionnaire
            quest = self.quest[self.quest.Name.str.match(self.t1dsimenv.patient.name)]
            self.CR = quest.CR.values[0]
            self.CF = quest.CF.values[0]
        else:
            # If no params are found, use default values
            self.CR = 1 / 15
            self.CF = 1 / 50

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def automatic_bolus(self):
        # Automatic bolus injection with uncertainty
        bolus = 0
        food = self.t1dsimenv.CHO_hist
        if len(food) > 0 and food[-1] > 0:
            bolus = (food[-1] * self.t1dsimenv.sample_time) / self.CR \
                    + (self.CGM_hist[-1] > 150) * (self.CGM_hist[-1] - 140) /self.CF  # unit: U
            bolus = bolus *  (0.7 + np.random.random()*0.4) / self.t1dsimenv.sample_time
        return bolus
    
    def step(self, action):
        action *= 0.5 # scale action to 0.5 steps of basal rate
        insulin_rate = self.basal_rate * action

        # add bolus injection
        bolus = 0
        if self.enable_auto_bolus:
            bolus = self.automatic_bolus()

        act = Action(basal=insulin_rate, bolus=bolus)

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
