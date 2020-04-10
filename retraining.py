from gym_torcs import TorcsEnv
import numpy as np
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Flatten, add, concatenate, Input
from ReplayBuffer import ReplayBuffer
import tensorflow as tf
import cv2 as cv
from OU import OU
global base_model
OU = OU()

print("Now we load the weight")
try:
    base_model = load_model("sl_save_model/sl_model.h5")
    print("Weight load successfully")
except:
    print("Cannot find the weight")


def create_model():
    global base_model

    pretrained_model = Model(base_model.input, base_model.layers[-5].output)
    x = pretrained_model.output
    a = Input((3, ))
    aa=base_model.layers[1]
    value_prediction = Dense(3, activation='linear', name='action_value')(x)
    critic = Model(base_model.input, value_prediction, name='critic_model')
    target_critic = Model(base_model.input, value_prediction, name='target_critic_model')



    actor = Model(base_model.input, base_model.output)
    target_actor = Model(base_model.input, base_model.output)

    critic.summary()
    target_critic.summary()

    actor.summary()
    target_actor.summary()

    return critic, target_critic, actor, target_actor

def get_gradient():


def playGame(train_indicator=0, actor=None):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    np.random.seed(1337)
    vision = True
    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    model = 0
    step = 0
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    critic, target_critic, actor, target_actor = create_model()
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer
    print("TORCS Experiment Start.")

    for i in range(episode_count):
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        img = ob.img
        total_reward = 0.0
        # for j in range(max_steps):
        #     a_t = np.zeros([1, 3])
        #     a_t_original = actor.predict([np.reshape(img, (-1, 64, 64, 1)) / 255.0, s_t.reshape(1, s_t.shape[0])])
        #
        #     a_t[0][0] = a_t_original[0][0]
        #     a_t[0][1] = a_t_original[0][1]
        #     a_t[0][2] = a_t_original[0][2]
        #
        #     ob, r_t, done, info = env.step(a_t[0])
        #     new_img = ob.img
        #     s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        #
        #     buff.add(img, new_img, s_t, a_t[0], r_t, s_t1, done)
        #
        #     batch = buff.getBatch(BATCH_SIZE)
        #     old_image = np.asarray([e[0] for e in batch])
        #     new_image = np.asarray([e[1] for e in batch])
        #     state = np.asarray([e[2] for e in batch])
        #     actions = np.asarray([e[3] for e in batch])
        #     rewards = np.asarray([e[4] for e in batch])
        #     new_state = np.asarray([e[5] for e in batch])
        #     dones = np.asarray([e[6] for e in batch])
        #     y_t = np.asarray([e[3] for e in batch])
        #
        #     old_image = np.reshape(old_image, (-1, 64, 64, 1))
        #     new_image = np.reshape(new_image, (-1, 64, 64, 1))
        #
        #     # a = target_actor.predict([new_image / 255.0, new_state.reshape(1, new_state.shape[0])])
        #     target_q_values = target_critic.predict([new_image / 255.0, new_state.reshape(1, new_state.shape[0])])
        #
        #     for k in range(len(batch)):
        #         if dones[k]:
        #             y_t[k] = rewards[k]
        #         else:
        #             y_t[k] = rewards[k] + GAMMA * target_q_values[k]
        #
        #     if train_indicator:
        #         loss = critic.train_on_batch([old_image / 255.0, state.reshape(1, state.shape[0])], y_t)
        #         a_for_grad = actor.predict([old_image / 255.0, state.reshape(1, state.shape[0])])
        #         grads = get_gradient(critic, input_1, input_2)
        #         grads = critic.gradients(old_image, a_for_grad)
        #         tf.gradients(, a_for_grad)
        #         actor.train(old_image, grads)
        #         actor.target_train()
        #         critic.target_train()
        #         K.gradients(loss=loss, critic)
        #
        #     total_reward += r_t
        #     img = new_img
        #     s_t = s_t1
        #     print("Step", step, r_t)
        #     step += 1
        #     if done:
        #         break
        #     if step > 100000:
        #        env.end()  # This is for shutting down TORCS

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    playGame(0)
