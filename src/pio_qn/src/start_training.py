#!/usr/bin/env python

import numpy
import time
import os
import sys
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import csv
from datetime import datetime
import subprocess

def encode_state(observation):
    return ''.join(map(str, observation))


def shutdown_simulation():
    """Close Gazebo and its GUI window.

    openai_ros starts the simulation from a roslaunch of its own, so nothing
    tears it down when this node exits. "rosnode kill" only sends a shutdown
    request over XML-RPC, which gzclient routinely ignores, so the remaining
    processes are terminated directly afterwards.
    """

    rospy.logwarn("Closing Gazebo simulation")

    subprocess.call([
        'rosnode',
        'kill',
        '/gazebo',
        '/gazebo_gui',
        '/robot_state_publisher',
    ])

    # Give the nodes a chance to exit cleanly before forcing it.
    time.sleep(2.0)

    # gzclient first, so the GUI does not hang waiting on a dead server.
    for process_name in ('gzclient', 'gzserver'):
        subprocess.call(['pkill', '-f', process_name])


if __name__ == '__main__':

    rospy.init_node('pio_depth_qlearn',
                    anonymous=True, log_level=rospy.WARN)

    # Rooms to cycle through during training (one gazebo.launch world name
    # each), switching every EPISODES_PER_ROOM_BLOCK episodes. Leave
    # TRAINING_ROOMS empty to train in a single room for the whole run, as
    # before.
    TRAINING_ROOMS = ['room_train_1', 'room_train_2', 'room_train_3', 'room_train_4', 'room_train_5']
    EPISODES_PER_ROOM_BLOCK = int(
        rospy.get_param('/pio/episodes_per_room_block')
    )
    
    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/pio/task_and_robot_environment_name')

    if TRAINING_ROOMS:
        os.environ['PIO_WORLD'] = TRAINING_ROOMS[0]

    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.on_shutdown(env.close)
    rospy.loginfo("Starting Learning")

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('pio_qn')

    # roslaunch runs the installed wrapper from devel/lib, so add this package's
    # source directory before importing the local qlearn.py helper.
    sys.path.insert(0, os.path.join(pkg_path, 'src'))
    import qlearn


    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    outdir = os.path.join(pkg_path, 'training_results', run_name)
    os.makedirs(outdir, exist_ok=True)

    steps_path = os.path.join(outdir, 'steps.csv')
    episodes_path = os.path.join(outdir, 'episodes.csv')
    model_path = os.path.join(outdir, 'q_table.pkl')

    steps_file = open(steps_path, 'w', newline='')
    episodes_file = open(episodes_path, 'w', newline='')

    steps_writer = csv.DictWriter(
        steps_file,
        fieldnames=[
            'episode',
            'step',
            'state',
            'action',
            'reward',
            'next_state',
            'done',
            'closest_depth',
            'cumulated_reward',
            'epsilon',
            'elapsed_seconds',
        ]
    )

    episodes_writer = csv.DictWriter(
        episodes_file,
        fieldnames=[
            'episode',
            'steps',
            'reward',
            'collision',
            'forward_actions',
            'left_actions',
            'right_actions',
            'minimum_depth',
            'unique_states',
            'epsilon',
            'duration_seconds',
        ]
    )

    steps_writer.writeheader()
    episodes_writer.writeheader()
    steps_file.flush()
    episodes_file.flush()


    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/pio/alpha")
    alpha_discount = rospy.get_param("/pio/alpha_discount")
    alpha_min = rospy.get_param("/pio/alpha_min")
    Epsilon = rospy.get_param("/pio/epsilon")
    Gamma = rospy.get_param("/pio/gamma")
    epsilon_discount = rospy.get_param("/pio/epsilon_discount")
    epsilon_min = rospy.get_param("/pio/epsilon_min", 0.05)
    nepisodes = rospy.get_param("/pio/nepisodes")
    nsteps = rospy.get_param("/pio/nsteps")

    checkpoint_interval = int(
        rospy.get_param(
            "/pio/checkpoint_interval",
            25
        )
    )

    if checkpoint_interval <= 0:
        raise ValueError(
            "checkpoint_interval must be greater than zero"
        )


    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)

    start_time = time.time()

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        if rospy.is_shutdown():
            break

        # Switch rooms every EPISODES_PER_ROOM_BLOCK episodes. openai_ros
        # only loads a world once, at environment construction, so changing
        # rooms means tearing down Gazebo and building a fresh environment.
        if TRAINING_ROOMS and x > 0 and x % EPISODES_PER_ROOM_BLOCK == 0:
            next_room = TRAINING_ROOMS[(x // EPISODES_PER_ROOM_BLOCK) % len(TRAINING_ROOMS)]
            env.unwrapped.cleanup_subscribers()
            shutdown_simulation()
            os.environ['PIO_WORLD'] = next_room
            env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

        cumulated_reward = 0
        done = False

        episode_start_time = time.time()
        action_counts = {0: 0, 1: 0, 2: 0}
        states_seen = set()
        minimum_depth = float('inf')
        completed_steps = 0

        qlearn.alpha = max(alpha_min, qlearn.alpha * alpha_discount)
        qlearn.epsilon = max(
            epsilon_min,
            qlearn.epsilon * epsilon_discount
        )

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = encode_state(observation)
        states_seen.add(state)

        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            if rospy.is_shutdown():
                break

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            completed_steps = i + 1
            action_counts[action] = action_counts.get(action, 0) + 1

            task_env = env.unwrapped
            closest_depth = getattr(task_env, 'closest_depth', float('nan'))

            if numpy.isfinite(closest_depth):
                minimum_depth = min(minimum_depth, closest_depth)

            cumulated_reward += reward

            nextState = encode_state(observation)
            states_seen.add(nextState)

            steps_writer.writerow({
                'episode': x + 1,
                'step': i + 1,
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': nextState,
                'done': done,
                'closest_depth': closest_depth,
                'cumulated_reward': cumulated_reward,
                'epsilon': qlearn.epsilon,
                'elapsed_seconds': round(time.time() - start_time, 3),
            })            


            qlearn.learn(state, action, reward, nextState, done=done)

            if not (done):
                state = nextState
            else:
                break
        
        steps_file.flush()

        episode_duration = time.time() - episode_start_time

        if minimum_depth == float('inf'):
            minimum_depth_value = ''
        else:
            minimum_depth_value = minimum_depth

        episodes_writer.writerow({
            'episode': x + 1,
            'steps': completed_steps,
            'reward': cumulated_reward,
            'collision': bool(done),
            'forward_actions': action_counts.get(0, 0),
            'left_actions': action_counts.get(1, 0),
            'right_actions': action_counts.get(2, 0),
            'minimum_depth': minimum_depth_value,
            'unique_states': len(states_seen),
            'epsilon': qlearn.epsilon,
            'duration_seconds': round(episode_duration, 3),
        })

        episodes_file.flush()

        # Checkpoint after every episode, so interrupted training is preserved.
        qlearn.save_model(model_path)

        # Preserve a permanent checkpoint at the configured interval.
        if (x + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                outdir,
                "q_table_episode_{:04d}.pkl".format(x + 1)
            )

            qlearn.save_model(checkpoint_path)

            rospy.logwarn(
                "Saved checkpoint for episode %d to: %s",
                x + 1,
                checkpoint_path
            )
        m, s = divmod(int(time.time() - start_time), 60)
        


        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))


    qlearn.save_model(model_path)

    steps_file.close()
    episodes_file.close()

    rospy.logwarn("Saved step history to: " + steps_path)
    rospy.logwarn("Saved episode summary to: " + episodes_path)
    rospy.logwarn("Saved trained Q-table to: " + model_path)

    env.close()

    shutdown_simulation()
