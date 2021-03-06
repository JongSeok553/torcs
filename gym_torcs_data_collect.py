# import gym
# from gym import spaces
# import numpy as np
# # from os import path
# import snakeoil3_gym as snakeoil3
# import numpy as np
# import copy
# import collections as col
# import os
# import time
# import cv2 as cv
# import math
#
# class TorcsEnv:
#     terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
#     termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
#     default_speed = 50
#
#     initial_reset = True
#
#     def __init__(self, vision=True, throttle=False, gear_change=False):
#         self.vision = vision
#         self.throttle = throttle
#         self.gear_change = gear_change
#         self.straight_speed = 0
#         self.initial_run = True
#
#         ##print("launch torcs")
#         os.system('pkill torcs')
#         time.sleep(0.5)
#         if self.vision is True:
#             os.system('torcs -nofuel -nodamage -nolaptime -vision &')
#         else:
#             os.system('torcs -nofuel -nolaptime &')
#         time.sleep(0.5)
#         os.system('sh autostart.sh')
#         time.sleep(0.5)
#
#         """
#         # Modify here if you use multiple tracks in the environment
#         self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
#         self.client.MAX_STEPS = np.inf
#
#         client = self.client
#         client.get_servers_input()  # Get the initial input from torcs
#
#         obs = client.S.d  # Get the current full-observation from torcs
#         """
#         if throttle is False:
#             self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
#         else:
#             self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
#
#         if vision is False:
#             high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
#             low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
#             self.observation_space = spaces.Box(low=low, high=high)
#         else:
#             high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
#             low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
#             self.observation_space = spaces.Box(low=low, high=high)
#
#     def step(self, u):
#        #print("Step")
#         # convert thisAction to the actual torcs actionstr
#         client = self.client
#
#         this_action = self.agent_to_torcs(u)
#
#         # Apply Action
#         action_torcs = client.R.d
#
#         # Steering
#         action_torcs['steer'] = this_action['steer']  # in [-1, 1]
#
#         #  Simple Autnmatic Throttle Control by Snakeoil
#         if self.throttle is False:
#             target_speed = self.default_speed
#             if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
#                 client.R.d['accel'] += .01
#             else:
#                 client.R.d['accel'] -= .01
#
#             if client.R.d['accel'] > 0.2:
#                 client.R.d['accel'] = 0.2
#
#             if client.S.d['speedX'] < 10:
#                 client.R.d['accel'] += 1/(client.S.d['speedX']+.1)
#
#             # Traction Control System
#             if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
#                (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
#                 action_torcs['accel'] -= .2
#         else:
#             action_torcs['accel'] = this_action['accel']
#             action_torcs['brake'] = this_action['brake']
#
#         #  Automatic Gear Change by Snakeoil
#         if self.gear_change is True:
#             action_torcs['gear'] = this_action['gear']
#         else:
#             #  Automatic Gear Change by Snakeoil is possible
#             action_torcs['gear'] = 1
#             if self.throttle:
#                 if client.S.d['speedX'] > 50:
#                     action_torcs['gear'] = 2
#                 if client.S.d['speedX'] > 80:
#                     action_torcs['gear'] = 3
#                 if client.S.d['speedX'] > 110:
#                     action_torcs['gear'] = 4
#                 if client.S.d['speedX'] > 140:
#                     action_torcs['gear'] = 5
#                 if client.S.d['speedX'] > 170:
#                     action_torcs['gear'] = 6
#         # Save the privious full-obs from torcs for the reward calculation
#         obs_pre = copy.deepcopy(client.S.d)
#
#         # One-Step Dynamics Update #################################
#         # Apply the Agent's action into torcs
#         client.respond_to_server()
#         # Get the response of TORCS
#         client.get_servers_input()
#
#         # Get the current full-observation from torcs
#         obs = client.S.d
#         # print("obs", obs)
#         # Make an obsevation from a raw observation vector from TORCS
#         self.observation = self.make_observaton(obs)
#
#         # Reward setting Here #######################################
#         # direction-dependent positive reward
#         track = np.array(obs['track'])
#         trackPos = np.array(obs['trackPos'])
#         trackPos = trackPos * 1.0
#         sp = np.array(obs['speedX'])
#         sp_y = np.array(obs['speedY'])
#         damage = np.array(obs['damage'])
#         rpm = np.array(obs['rpm'])
#         radian = np.array(obs['angle'])
#
#         curtime = np.array(obs['curLapTime'])
#         lasttime = np.array(obs['lastLapTime'])
#         print("cur ", curtime, "last ", lasttime)
#         # print("angle", obs['angle'] * (180.0 / math.pi), corner)
#         # print("y ", np.abs(sp_y))
#         # print("trackpos ", trackPos)
#     ##############################  Left  #############################
#     #     if corner == 4:     # straight to Left
#     #         if -0.5 < trackPos < 0:
#     #             progress = np.abs(trackPos) * np.cos(radian) * sp
#     #         else:
#     #             progress = -trackPos * np.abs(np.sin(radian)) * sp
#     #     elif corner == 0:   # Left to Left
#     #         if 0.5 > trackPos > 0:
#     #             progress = np.abs(trackPos) * np.cos(radian) * sp
#     #         else:
#     #             progress = -np.abs(trackPos) * np.abs(np.sin(radian)) * sp
#     #     elif corner == 1:   # Left to straight
#     #         if -0.5 < trackPos < 0:
#     #             progress = np.abs(trackPos) * np.cos(radian) * sp
#     #         else:
#     #             progress = -trackPos * np.abs(np.sin(radian)) * sp
#     #
#     # ##############################  Right  #############################
#     #
#     #     elif corner == 5:   # straight to Right
#     #         if 0.5 > trackPos > 0:
#     #             progress = trackPos * np.cos(radian) * sp
#     #         else:
#     #             progress = -np.abs(trackPos) * np.abs(np.sin(radian)) * sp
#     #     elif corner == 2:   # Right to Right
#     #         if -0.5 < trackPos < 0:
#     #             progress = np.abs(trackPos) * np.cos(radian) * sp
#     #         else:
#     #             progress = -trackPos * np.abs(np.sin(radian)) * sp
#     #     elif corner == 3:   # Right to straight
#     #         if 0.5 > trackPos > 0:
#     #             progress = trackPos * np.cos(radian) * sp
#     #         else:
#     #             progress = -np.abs(trackPos) * np.abs(np.sin(radian)) * sp
#
#         ##############################  Straight  #############################
#         # else:   #corner == 6:  # Right to straight
#         # if corner == -1:     # Left
#         #     if 0 < trackPos < 0.9:
#         #         # print("left + trackpos")
#         #         progress = sp * np.cos(radian) - np.abs(sp * np.sin(radian)) + sp * np.abs(obs['trackPos'])
#         #     else:
#         #         # print("left - trackpos")
#         #         progress = sp * np.cos(radian) - np.abs(sp * np.sin(radian)) - sp * np.abs(obs['trackPos'])
#         # elif corner == 1:   # Right
#         #     if 0 > trackPos > -0.9:
#         #         # print("right + trackpos")
#         #         progress = sp * np.cos(radian) - np.abs(sp * np.sin(radian)) + sp * np.abs(obs['trackPos'])
#         #     else:
#         #         # print("right - trackpos")
#         #         progress = sp * np.cos(radian) - np.abs(sp * np.sin(radian)) - sp * np.abs(obs['trackPos'])
#         # else:
#         #     # print("straight")
#         #     progress = sp * np.cos(radian) - np.abs(sp * np.sin(radian)) - sp * np.abs(obs['trackPos'])
#         #
#         # reward = progress
#         progress = sp * np.cos(radian) - np.abs(sp * np.sin(radian)) - sp * np.abs(obs['trackPos'])
#         reward = progress
#         # collision detection
#
#         episode_terminate = False
#         if obs['damage'] - obs_pre['damage'] > 0:
#             reward = -1
#             episode_terminate = True
#             client.R.d['meta'] = True
#
#         # Termination judgement #########################
#         # episode_terminate = False
#         # if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
#         #     reward = -200
#         #     episode_terminate = True
#         #     client.R.d['meta'] = True
#         #
#         # if self.terminal_judge_start < self.time_step:  # Episode terminates if the progress of agent is small
#         #     if progress < self.termination_limit_progress:
#         #         print("No progress")
#         #         episode_terminate = True
#         #         client.R.d['meta'] = True
#         #
#         # if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
#         #     episode_terminate = True
#         #     client.R.d['meta'] = True
#         #
#         # if client.R.d['meta'] is True: # Send a reset signal
#         #     self.initial_run = False
#         #     client.respond_to_server()
#
#         self.time_step += 1
#
#         return self.get_obs(), reward, client.R.d['meta'], {}
#
#     def reset(self, relaunch=False):
#         #print("Reset")
#
#         self.time_step = 0
#
#         if self.initial_reset is not True:
#             self.client.R.d['meta'] = True
#             self.client.respond_to_server()
#
#             ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
#             if relaunch is True:
#                 self.reset_torcs()
#                 print("### TORCS is RELAUNCHED ###")
#
#         # Modify here if you use multiple tracks in the environment
#         self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
#         self.client.MAX_STEPS = np.inf
#
#         client = self.client
#         client.get_servers_input()  # Get the initial input from torcs
#
#         obs = client.S.d  # Get the current full-observation from torcs
#         self.observation = self.make_observaton(obs)
#
#         self.last_u = None
#
#         self.initial_reset = False
#         return self.get_obs()
#
#     def end(self):
#         os.system('pkill torcs')
#
#     def get_obs(self):
#         return self.observation
#
#     def reset_torcs(self):
#        #print("relaunch torcs")
#         os.system('pkill torcs')
#         time.sleep(0.5)
#         if self.vision is True:
#             os.system('torcs -nofuel -nodamage -nolaptime -vision &')
#         else:
#             os.system('torcs -nofuel -nolaptime &')
#         time.sleep(0.5)
#         os.system('sh autostart.sh')
#         time.sleep(0.5)
#
#     def agent_to_torcs(self, u):
#         torcs_action = {'steer': u[0]}
#
#         if self.throttle is True:  # throttle action is enabled
#             torcs_action.update({'accel': u[1]})
#             torcs_action.update({'brake': u[2]})
#
#         if self.gear_change is True: # gear change action is enabled
#             torcs_action.update({'gear': int(u[3])})
#
#         return torcs_action
#
#
#     def obs_vision_to_image_rgb(self, obs_image_vec):
#         image_vec =  obs_image_vec
#         r = image_vec[0:len(image_vec):3]
#         g = image_vec[1:len(image_vec):3]
#         b = image_vec[2:len(image_vec):3]
#
#         sz = (64, 64)
#         r = np.array(r).reshape(sz)
#         g = np.array(g).reshape(sz)
#         b = np.array(b).reshape(sz)
#         # r = r * 0.2989
#         # g = g * 0.5870
#         # b = b * 0.1140
#         rgb = cv.merge([b, g, r])
#         rgb = cv.flip(rgb, 0)
#         rgb[48][53] = rgb[47][51]
#         rgb[49][53] = rgb[47][51]
#         rgb[50][53] = rgb[47][51]
#         rgb[51][53] = rgb[47][51]
#         rgb[52][53] = rgb[47][51]
#         rgb[53][53] = rgb[47][51]
#         rgb[54][53] = rgb[47][51]
#         rgb[55][53] = rgb[47][51]
#         rgb[56][53] = rgb[47][51]
#         rgb[57][53] = rgb[47][51]
#         rgb[58][53] = rgb[47][51]
#         rgb[59][53] = rgb[47][51]
#
#         rgb[48][60] = rgb[47][61]
#         rgb[49][60] = rgb[47][61]
#         rgb[50][60] = rgb[47][61]
#         rgb[51][60] = rgb[47][61]
#         rgb[52][60] = rgb[47][61]
#         rgb[53][60] = rgb[47][61]
#         rgb[54][60] = rgb[47][61]
#         rgb[55][60] = rgb[47][61]
#         rgb[56][60] = rgb[47][61]
#         rgb[57][60] = rgb[47][61]
#         rgb[58][60] = rgb[47][61]
#         rgb[59][60] = rgb[47][61]
#         rgb[53][48:59] = rgb[52][47]
#
#         return rgb
#
#     def make_observaton(self, raw_obs):
#         if self.vision is False:
#             names = ['focus',
#                      'speedX', 'speedY', 'speedZ', 'angle', 'damage',
#                      'opponents',
#                      'rpm',
#                      'track',
#                      'trackPos',
#                      'wheelSpinVel']
#             Observation = col.namedtuple('Observaion', names)
#             return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
#                                speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
#                                speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
#                                speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
#                                angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
#                                damage=np.array(raw_obs['damage'], dtype=np.float32),
#                                opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
#                                rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
#                                track=np.array(raw_obs['track'], dtype=np.float32)/200.,
#                                trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
#                                wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
#         else:
#             names = ['focus',
#                      'speedX', 'speedY', 'speedZ', 'angle',
#                      'opponents',
#                      'rpm',
#                      'track',
#                      'trackPos',
#                      'wheelSpinVel',
#                      'img',
#                      'curLapTime',
#                      'lastLapTime']
#             Observation = col.namedtuple('Observaion', names)
#
#             # Get RGB from observation
#             image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[10]])
#
#             return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
#                                speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
#                                speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
#                                speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
#                                angle=np.array(raw_obs['angle'], dtype=np.float32) / 3.1416,
#                                opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
#                                rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
#                                track=np.array(raw_obs['track'], dtype=np.float32)/200.,
#                                trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
#                                wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
#                                img=image_rgb,
#                                curLapTime=np.array(raw_obs['curLapTime'], dtype=np.float32),
#                                lastLapTime=np.array(raw_obs['lastLapTime'], dtype=np.float32))
#
import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time


class TorcsEnv:
    terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.n = 0
        self.pre_lasttime = 0
        self.file = open('speed_/' + 'speed_file' + str(self.n) + '.txt', 'w')
        self.initial_run = True

        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf
        client = self.client
        client.get_servers_input()  # Get the initial input from torcs
        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        reward = progress

        curtime = np.array(obs['curLapTime'])
        lasttime = np.array(obs['lastLapTime'])
        dist = np.array(obs['distFromStart'])
        # if self.n < 10:
        #     if self.pre_lasttime == lasttime:
        #         self.file.write(str(sp) + '\n')
        #     else:
        #         self.file.close()c
        #         self.n += 1
        #         self.file = open('speed_/' + 'speed_file' + str(self.n) + '.txt', 'w')
        #         self.pre_lasttime = lasttime
        # else:
        #     self.file.close()

        print("cur ", curtime, "last ", lasttime, "dist ", dist)
        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement #########################
        episode_terminate = False
        #if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
        #    reward = -200
        #    episode_terminate = True
        #    client.R.d['meta'] = True

        #if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
        #    if progress < self.termination_limit_progress:
        #        print("No progress")
        #        episode_terminate = True
        #        client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img',
                     'curLapTime',
                     'lastLapTime'
                     'distFromStart']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[10]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32) / 3.1416,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb,
                               curLapTime=np.array(raw_obs['curLapTime'], dtype=np.float32),
                               lastLapTime=np.array(raw_obs['lastLapTime'], dtype=np.float32),
                               distFromStart=np.array(raw_obs['distFromStart'], dtype=np.float32))