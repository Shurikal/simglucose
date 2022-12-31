import gym
from gym import spaces
import numpy as np

from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
from simglucose.simulation.rendering import Viewer

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

        # todo, noise seed
        self.sensor = CGMSensor.withName(self.SENSOR_HARDWARE)
        self.pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)

        self.sample_time = self.sensor.sample_time

        if patient_name is None:
            patient_name = ['adolescent#001']

        self.patient_name = patient_name
        self.reward_fun = reward_fun

        if custom_scenario is None:
            self.custom_scenario = RandomScenario(start_time=datetime(2018, 1, 1, 0, 0))
        else:
            self.custom_scenario = custom_scenario

        self.time = None

        self.observation_space = spaces.Dict(
            {
                "GCM": spaces.Box(low=0,high=10000, shape=(1,)),
                "CHO": spaces.Box(low=0,high= 10000, shape=(1,)),
            }
        )

        self.action_space = spaces.Dict(
            {
                "basal": spaces.Box(low=0,high=self.pump._params['max_basal'], shape=(1,)),
                "bolus": spaces.Box(low=0,high=0, shape=(1,)),
            }
        )

    # todo
    def _get_obs(self):
        patient_action = self.scenario.get_action(self.time)
        return {"GCM": np.array([self.sensor.measure(self.patient)], dtype=np.float32), "CHO": np.array([patient_action.meal], dtype=np.float32)}

    # todo
    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            self.patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed)
        else:
            self.patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed)

        if isinstance(self.custom_scenario, list):
            self.scenario = self.np_random.choice(self.custom_scenario)
        else:
            self.scenario = RandomScenario(start_time=datetime(2018,1,1,0,0,0), seed=seed) if self.custom_scenario is None else self.custom_scenario

        self.time = self.scenario.start_time

        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def mini_step(self, action):
        # current action
        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action['basal'])
        bolus = self.pump.bolus(action['bolus'])
        insulin = basal + bolus
        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM
    
    def step(self, action):

        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0

        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action)
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            CGM += tmp_CGM / self.sample_time

        observation = self._get_obs()
        info = self._get_info()
        reward = 0.0
        terminated = False

        if self.reward_fun is None:
            return (observation, reward, terminated, False, info)

        return (observation, reward, terminated, False, info)


    def render(self):
        pass