# from gym_torcs_data_collect import TorcsEnv
# import numpy as np
# import random
# import tensorflow as tf
# import cv2 as cv
# from keras.models import load_model
# from ReplayBuffer import ReplayBuffer
# from ActorNetwork import ActorNetwork
# from CriticNetwork import CriticNetwork
# from OU import OU
# import timeit
#
# OU = OU()       #Ornstein-Uhlenbeck Process
#
# def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
#     BUFFER_SIZE = 100000
#     BATCH_SIZE = 32
#     GAMMA = 0.99
#     TAU = 0.001     #Target Network HyperParameters
#     LRA = 0.0001    #Learning rate for Actor
#     LRC = 0.001     #Lerning rate for Critic
#
#     action_dim = 3  #Steering/Acceleration/Brake
#     state_dim = 29  #of sensors input
#
#     np.random.seed(1337)
#
#     vision = True
#
#     EXPLORE = 100000.
#     episode_count = 2000
#     max_steps = 150000
#     reward = 0
#     done = False
#     step = 0
#     epsilon = 1
#     indicator = 0
#
#
#     #Tensorflow GPU optimization
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     from keras import backend as K
#     K.set_session(sess)
#
#     actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
#     critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
#     buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
#     corner_detector = 0
#     # Generate a Torcs environment
#     env = TorcsEnv(vision=vision, throttle=True,gear_change=False)
#
#     #Now load the weight
#     print("Now we load the weight")
#     try:
#         actor.model.load_weights("actormodel.h5")
#         critic.model.load_weights("criticmodel.h5")
#         actor.target_model.load_weights("actormodel.h5")
#         critic.target_model.load_weights("criticmodel.h5")
#         corner_detector = load_model("corner_detector/model_name.h5")
#         print("Weight load successfully")
#     except:
#         print("Cannot find the weight")
#
#     try:
#         # f = open('save_state/' + 'state.txt', mode='w')
#         # a_f = open('save_action/' + 'action.txt', mode='w')
#         print("file open Successfully !")
#     except:
#         print("file open Failed !")
#
#     print("TORCS Experiment Start.")
#     for i in range(max_steps):
#
#         print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
#
#         if np.mod(i, 3) == 0:
#             ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
#         else:
#             ob = env.reset()
#
#         s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
#
#         total_reward = 0.
#         for j in range(max_steps):
#             track = []
#             # crop = ob.img
#             # crop = crop[20:64, 0:64]
#             # crop = np.reshape(crop, (-1, 44, 64, 3)) / 255.0
#             a_t = np.zeros([1, action_dim])
#
#             # is_corner = corner_detector.predict([crop])
#             # cv.imwrite("dwdwdw/" + str(step) + str(is_corner[0][0]) + ".jpg", crop)
#             a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
#             # print("corner?", is_corner[0][0])
#             # noise_t = np.zeros([1, action_dim])
#             # noise_t[0][0] = 1 * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
#             # noise_t[0][1] = 1 * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
#             # noise_t[0][2] = 1 * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)
#
#             # The following code do the stochastic brake
#             # if random.random() <= 0.1:
#             #    print("********Now we apply the brake***********")
#             #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)
#             # rnd = (0.4 * random.random())
#             # print(rnd)
#
#             a_t[0][0] = a_t_original[0][0] #+ noise_t[0][0] * rnd
#             a_t[0][1] = a_t_original[0][1] #+ noise_t[0][1] * rnd
#             a_t[0][2] = a_t_original[0][2] #+ noise_t[0][2] * rnd
#             ob, r_t, done, info = env.step(a_t[0])
#             # print("@@@@@@@@@@@@@",ob.img)
#
#             #ss = np.reshape(ob.track, (1, 19))
#
#             # for i in range(19):
#             #     track.append(ob.track[i])
#             #print("@#@#@#@", ob.wheelSpinVel)
#             # act = str(a_t[0][0]) + "\t" + str(a_t[0][1]) + "\t" + str(a_t[0][2]) + "\n"
#             # state = str(ob.angle) + "\t" + str(ob.track[0]) + "\t" + str(ob.track[1]) + "\t" + str(ob.track[2]) + "\t" + str(ob.track[3]) + "\t" + str(ob.track[4]) + "\t" + str(ob.track[5]) + "\t" + str(ob.track[6]) + "\t" + str(ob.track[7]) + "\t" + str(ob.track[8]) + "\t" + str(ob.track[9]) + "\t" + str(ob.track[10]) + "\t" + str(ob.track[11]) + "\t" + str(ob.track[12]) + "\t" + str(ob.track[13]) + "\t" + str(ob.track[14]) + "\t" + str(ob.track[15]) + "\t" + str(ob.track[16]) + "\t" + str(ob.track[17]) + "\t" + str(ob.track[18]) + "\t" +  str(ob.trackPos) + "\t" +str(ob.speedX) + "\t" + str(ob.speedY) + "\t" + str(ob.speedZ) + "\t" + str(ob.wheelSpinVel[0]/100.0) + "\t" + str(ob.wheelSpinVel[1]/100.0) + "\t" + str(ob.wheelSpinVel[2]/100.0) + "\t" + str(ob.wheelSpinVel[3]/100.0) + "\t" + str(ob.rpm) + "\n"
#             # f.write(state)
#             # a_f.write(act)
#             #print("##########", state)
#             s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
#
#             total_reward += r_t
#             s_t = s_t1
#
#             # print("Step", step)
#
#             step += 1
#             if indicator:
#                 if done:
#                     break
#                 if step > 150000:
#                    # f.close()
#                    # a_f.close()
#                    env.end()  # This is for shutting down TORCS
#                    print("150,000 Data save and Program Finish !!")
#                    return 0
#             if done:
#                 break
#         print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
#         print("Total Step: " + str(step))
#         print("")
#
#     env.end()  # This is for shutting down TORCS
#     print("Finish.")
#     # f.close()
#     # a_f.close()
#
# if __name__ == "__main__":
#     playGame(0)
from gym_torcs_data_collect import TorcsEnv
import numpy as np
import tensorflow as tf
import json
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=0):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    action_dim = 3  # Steering/Acceleration/Brake
    state_dim = 29  # of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    train_indicator = 0
    n = 0
    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    # Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel_test.h5")
        critic.model.load_weights("criticmodel_test.h5")
        actor.target_model.load_weights("actormodel_test.h5")
        critic.target_model.load_weights("criticmodel_test.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack(
            (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

            # The following code do the stochastic brake
            # if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            s_t1 = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            # print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel_test.h5", overwrite=True)
                with open("actormodel_test.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel_test.h5", overwrite=True)
                with open("criticmodel_test.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    playGame()