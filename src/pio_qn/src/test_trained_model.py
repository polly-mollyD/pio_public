#!/usr/bin/env python

import glob
import os
import pickle
import sys
import time
import csv
import json
import math
import subprocess

import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

def get_robot_position(env):

    odometry = env.unwrapped.get_odom()

    if odometry is None:
        return None

    return (
        odometry.pose.pose.position.x,
        odometry.pose.pose.position.y,
    )


def encode_state(observation):
    return ''.join(map(str, observation))


def shutdown_simulation():

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


def find_latest_model(results_directory):
    pattern = os.path.join(
        results_directory,
        "run_*",
        "q_table.pkl"
    )

    models = glob.glob(pattern)

    if not models:
        raise RuntimeError(
            "No q_table.pkl model found inside: "
            + results_directory
        )

    return max(models, key=os.path.getmtime)


def get_next_test_name(results_root, world_name):
    """Return test_roomN_trailM using the next trail number for the room."""

    world_prefix = "room_trial_"
    if not world_name.startswith(world_prefix):
        raise RuntimeError(
            "PIO_WORLD must be room_trial_01 through room_trial_10; got: "
            + world_name
        )

    try:
        room_number = int(world_name[len(world_prefix):])
    except ValueError:
        raise RuntimeError("Invalid PIO_WORLD: " + world_name)

    if room_number < 1 or room_number > 10:
        raise RuntimeError("Invalid PIO_WORLD: " + world_name)

    test_prefix = "test_room{}_trail".format(room_number)
    existing_trails = []

    for path in glob.glob(os.path.join(results_root, test_prefix + "*")):
        folder_name = os.path.basename(path)
        suffix = folder_name[len(test_prefix):]

        if os.path.isdir(path) and suffix.isdigit():
            existing_trails.append(int(suffix))

    trail_number = max(existing_trails, default=0) + 1
    return test_prefix + str(trail_number), trail_number


def load_model(model_path, qlearn_module):
    with open(model_path, "rb") as model_file:
        saved_model = pickle.load(model_file)

    learner = qlearn_module.QLearn(
        actions=saved_model["actions"],
        alpha=saved_model["alpha"],
        gamma=saved_model["gamma"],

        # No random exploration during testing.
        epsilon=0.0,
    )

    learner.q = saved_model["q_table"]

    return learner



def run_single_test(
    env,
    learner,
    run_number,
    duration_seconds,
    trajectory_csv_path=None
):

    run_reward = 0.0
    run_steps = 0
    run_collisions = 0
    distance_travelled = 0.0
    time_to_first_collision = None
    collision_free_intervals = []
    trajectory_points = []


    action_counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    # Start every run from the environment's reset pose.
    observation = env.reset()
    previous_position = get_robot_position(env)

    # Start measuring only after the initial reset is finished.
    run_start_time = time.time()
    collision_free_start = run_start_time

    if previous_position is not None:
        trajectory_points.append((
            0,
            0.0,
            previous_position[0],
            previous_position[1],
            False
        ))


    state = encode_state(observation)

    while not rospy.is_shutdown():
        elapsed_time = time.time() - run_start_time

        if elapsed_time >= duration_seconds:
            break

        # Because epsilon is zero, this uses the learned policy.
        action = learner.chooseAction(state)

        observation, reward, done, info = env.step(action)

        current_position = get_robot_position(env)
        if (
            previous_position is not None
            and current_position is not None
        ):
            delta_x = (
                current_position[0]
                - previous_position[0]
            )

            delta_y = (
                current_position[1]
                - previous_position[1]
            )

            step_distance = math.sqrt(
                delta_x ** 2 + delta_y ** 2
            )

            # Ignore unrealistic odometry jumps caused by Gazebo resets.
            if step_distance < 1.0:
                distance_travelled += step_distance

        previous_position = current_position


        run_steps += 1
        run_reward += reward

        action_counts[action] = (
            action_counts.get(action, 0) + 1
        )

        if current_position is not None:
            trajectory_points.append((
                run_steps,
                elapsed_time,
                current_position[0],
                current_position[1],
                done
            ))

        if done:
            run_collisions += 1

            collision_time = time.time()

            collision_free_intervals.append(
                collision_time - collision_free_start
            )

            time_to_first_collision = (
                collision_time - run_start_time
            )

            break

        state = encode_state(observation)


    run_end_time = time.time()

    if run_collisions == 0:
        collision_free_intervals.append(
            run_end_time - collision_free_start
        )

    longest_collision_free_time = max(
        collision_free_intervals
    ) if collision_free_intervals else 0.0

    mean_collision_free_time = (
        sum(collision_free_intervals)
        / float(len(collision_free_intervals))
        if collision_free_intervals
        else 0.0
    )
    elapsed_time = time.time() - run_start_time

    if trajectory_csv_path is not None:
        with open(trajectory_csv_path, "w", newline="") as trajectory_file:
            trajectory_writer = csv.writer(trajectory_file)
            trajectory_writer.writerow(
                ["step", "elapsed_seconds", "x", "y", "collision"]
            )
            trajectory_writer.writerows(trajectory_points)

    result = {
        "run": run_number,
        "seconds": elapsed_time,
        "steps": run_steps,
        "collisions": run_collisions,
        "collision_free_run": run_collisions == 0,
        "time_to_first_collision_seconds": (
            time_to_first_collision
        ),
        "longest_collision_free_seconds": (
            longest_collision_free_time
        ),
        "mean_collision_free_seconds": (
            mean_collision_free_time
        ),
        "distance_travelled_m": distance_travelled,
        "reward": run_reward,
        "forward": action_counts.get(0, 0),
        "left": action_counts.get(1, 0),
        "right": action_counts.get(2, 0),
    }

    return result



if __name__ == "__main__":
    rospy.init_node(
        "pio_test_trained_model",
        anonymous=True
    )

    rospack = rospkg.RosPack()
    package_path = rospack.get_path("pio_qn")

    # Allows this program to import src/qlearn.py.
    sys.path.insert(
        0,
        os.path.join(package_path, "src")
    )

    import qlearn

    requested_model_path = rospy.get_param(
        "/pio_test/model_path",
        ""
    )

    # Length of each individual run.
    # Default: 300 seconds = 5 minutes.
    duration_seconds = float(
        rospy.get_param(
            "/pio_test/duration_seconds",
            300.0
        )
    )

    # Default: 3 separate test runs.
    number_of_runs = int(
        rospy.get_param(
            "/pio_test/number_of_runs",
            3
        )
    )

    if duration_seconds <= 0:
        raise ValueError(
            "duration_seconds must be greater than zero"
        )

    if number_of_runs <= 0:
        raise ValueError(
            "number_of_runs must be greater than zero"
        )

    if requested_model_path:
        model_path = os.path.abspath(
            os.path.expanduser(requested_model_path)
        )
    else:
        results_directory = os.path.join(
            package_path,
            "training_results"
        )

        model_path = find_latest_model(
            results_directory
        )

    if not os.path.isfile(model_path):
        raise RuntimeError(
            "Model file does not exist: " + model_path
        )

    learner = load_model(
        model_path,
        qlearn
    )

    environment_name = rospy.get_param(
        "/pio/task_and_robot_environment_name"
    )

    env = StartOpenAI_ROS_Environment(
        environment_name
    )

    rospy.on_shutdown(env.close)


    test_results_root = os.path.join(
        package_path,
        "test_results"
    )

    os.makedirs(test_results_root, exist_ok=True)

    world_name = os.environ.get("PIO_WORLD", "room_trial_01")
    test_name, trail_number = get_next_test_name(
        test_results_root,
        world_name
    )

    results_directory = os.path.join(test_results_root, test_name)
    os.makedirs(results_directory)

    runs_csv_path = os.path.join(
        results_directory,
        "runs.csv"
    )

    metadata_path = os.path.join(
        results_directory,
        "metadata.json"
    )

    summary_path = os.path.join(
        results_directory,
        "summary.json"
    )

    rospy.logwarn(
        "Test results will be saved to: %s",
        results_directory
    )

    metadata = {
        "test_name": test_name,
        "world": world_name,
        "trail_number": trail_number,
        "model_path": model_path,
        "environment": environment_name,
        "number_of_runs": number_of_runs,
        "duration_seconds_per_run": duration_seconds,
        "learning_enabled": False,
        "evaluation_epsilon": learner.epsilon,
        "pio_parameters": rospy.get_param(
            "/pio",
            {}
        ),
    }

    with open(metadata_path, "w") as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
            sort_keys=True
        )

    all_results = []
    runs_csv_file = None

    try:
        csv_fieldnames = [
            "run",
            "seconds",
            "steps",
            "collisions",
            "collision_free_run",
            "time_to_first_collision_seconds",
            "longest_collision_free_seconds",
            "mean_collision_free_seconds",
            "distance_travelled_m",
            "reward",
            "forward",
            "left",
            "right",
        ]

        runs_csv_file = open(
            runs_csv_path,
            "w",
            newline=""
        )

        runs_writer = csv.DictWriter(
            runs_csv_file,
            fieldnames=csv_fieldnames
        )

        runs_writer.writeheader()
        runs_csv_file.flush()
        for run_number in range(
            1,
            number_of_runs + 1
        ):
            if rospy.is_shutdown():
                break

            trajectory_csv_path = os.path.join(
                results_directory,
                "trajectory_run_{}.csv".format(run_number)
            )

            result = run_single_test(
                env=env,
                learner=learner,
                run_number=run_number,
                duration_seconds=duration_seconds,
                trajectory_csv_path=trajectory_csv_path
            )

            all_results.append(result)
            runs_writer.writerow(result)
            runs_csv_file.flush()

    finally:
        if runs_csv_file is not None:
            runs_csv_file.close()

        env.close()

    if all_results:
        completed_runs = len(all_results)

        total_steps = sum(
            result["steps"]
            for result in all_results
        )

        total_collisions = sum(
            result["collisions"]
            for result in all_results
        )

        total_reward = sum(
            result["reward"]
            for result in all_results
        )

        total_forward = sum(
            result["forward"]
            for result in all_results
        )

        total_left = sum(
            result["left"]
            for result in all_results
        )

        total_right = sum(
            result["right"]
            for result in all_results
        )

        average_steps = (
            total_steps / float(completed_runs)
        )

        average_collisions = (
            total_collisions / float(completed_runs)
        )

        average_reward = (
            total_reward / float(completed_runs)
        )

        collision_free_runs = sum(
            1
            for result in all_results
            if result["collision_free_run"]
        )

        average_distance = sum(
            result["distance_travelled_m"]
            for result in all_results
        ) / float(completed_runs)

        average_longest_collision_free = sum(
            result["longest_collision_free_seconds"]
            for result in all_results
        ) / float(completed_runs)

        summary = {
            "completed_runs": completed_runs,
            "requested_runs": number_of_runs,
            "total_steps": total_steps,
            "total_collisions": total_collisions,
            "average_collisions_per_run": average_collisions,
            "collision_free_runs": collision_free_runs,
            "collision_free_run_rate": (
                collision_free_runs / float(completed_runs)
            ),
            "total_reward": total_reward,
            "average_reward": average_reward,
            "average_steps": average_steps,
            "average_distance_travelled_m": average_distance,
            "average_longest_collision_free_seconds": (
                average_longest_collision_free
            ),
            "total_forward_actions": total_forward,
            "total_left_actions": total_left,
            "total_right_actions": total_right,
        }

        with open(summary_path, "w") as summary_file:
            json.dump(
                summary,
                summary_file,
                indent=2,
                sort_keys=True
            )

    shutdown_simulation()
