# Pio robot: obstacle avoidance with Q-learning

This is the public version of the project; files that cannot be disclosed and the original Git history have been omitted.

## 1. Aim and software versions

This bachelor project trains a mobile robot to avoid obstacles using depth-camera observations and tabular Q-learning, and evaluates the learned policy in different simulated rooms.

Development environment: **Ubuntu 20.04.6 LTS, ROS Noetic (ROS 1), Gazebo Classic 11.11.0, and Python 3.8.10**. A working `openai_ros` installation and its dependencies are assumed. RViz and `teleop_twist_keyboard` are used for visualization and manual control.

## 2. Linux folder structure and custom openai_ros files

Place the main workspace at `~/pio_public` (`~` means `/home/<username>`):

```text
~/pio_public/
  README.md
  src/
    CMakeLists.txt
    pio_description/Asem_Reduced.SLDASM/
    pio_qn/
```

The examples assume that the existing `openai_ros` source installation is in `~/catkin_ws/src/openai_ros/openai_ros/`. The submission contains only the custom files, not the full library. Copy them into the matching folders of that installation:

```text
~/catkin_ws/src/openai_ros/openai_ros/src/openai_ros/
  robot_envs/pio_env.py
  task_envs/pio/__init__.py
  task_envs/pio/pio_depth_obstacle.py
  task_envs/pio/config/pio_depth_obstacle.yaml
  task_envs/task_envs_list.py
```

Add or update the first four files. **Replace the existing `task_envs_list.py` with the supplied version** to register `PioDepthObstacle-v0`. Keep the rest of the installed library. 

Adjust `/pio/ros_ws_abspath` in `pio_qn/config/pio_qlearn.yaml` to your absolute workspace path. In `pio_description/Asem_Reduced.SLDASM/launch/gazebo.launch`, replace `/home/pollymolly/.gazebo/models` with your own absolute model directory.

Rebuild the existing library workspace after copying the custom files:

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## 3. Required obstacle models

The obstacle models are assumed to be installed, with their model files, meshes, and materials, at:

```text
~/.gazebo/models/beer/
~/.gazebo/models/table_marble/
```

The optional `turtlebot3_house.world` additionally requires `cafe_table/`, `first_2015_trash_can/`, and `mailbox/` in the same model directory.

## 4. Build and launch pio_description

In **Terminal 1**, build the Pio workspace and launch Gazebo:

```bash
source /opt/ros/noetic/setup.bash
cd ~/pio_public
catkin_make
source devel/setup.bash
roslaunch pio_description gazebo.launch world:=room_trial_01
```

To change worlds, stop the launch with `Ctrl+C` and run it again with another filename stem, such as `world:=room_train_1` or `world:=turtlebot3_house`. Available experiment worlds are `room_train_1`-`room_train_5` and `room_trial_01`-`room_trial_10`. 


## 5. RViz and keyboard control

Run these setup commands in each new terminal:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/pio_public/devel/setup.bash
```

**Standalone robot preview:** stop Gazebo first, then run:

```bash
roslaunch pio_description display.launch
```

This opens RViz with `config/pio.rviz` and starts the joint-state GUI and robot-state publisher. Stop this launch with `Ctrl+C` before returning to simulation, training, or testing.

**Keyboard control:** with Gazebo running in Terminal 1, start teleoperation in a separate **Terminal 2**:

```bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```

Keep keyboard focus in Terminal 2: `i` moves forward, `,` backward, `j`/`l` turn left/right, and `k` stops. Use `q`/`z` to increase/decrease speeds and `Ctrl+C` to exit.

## 6. Build pio_qn and train

`catkin_make` builds both Pio packages. To rebuild after changes, use:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/pio_public/devel/setup.bash --extend
cd ~/pio_public
catkin_make
```

Start training with the GUI:

```bash
PIO_GUI=true roslaunch pio_qn start_training.launch
```

Or disable the GUI while keeping the training running:

```bash
PIO_GUI=false roslaunch pio_qn start_training.launch
```

Settings are in `src/pio_qn/config/pio_qlearn.yaml`. Training starts from scratch, runs 1,000 episodes by default, and changes training rooms every ten episodes. Logs and checkpoints are saved under `src/pio_qn/training_results/`.

## 7. Test different rooms and models

In a terminal, run:

```bash
PIO_WORLD=room_trial_01 PIO_GUI=true \
roslaunch pio_qn test_trained_model.launch \
  model_path:="$HOME/pio_public/src/pio_qn/training_results/run_20260824_211441/q_table_episode_0900.pkl" \
  duration_seconds:=300 number_of_runs:=3
```

- Change `PIO_WORLD` to another room, for example `room_trial_04`.
- Change `model_path` to another saved checkpoint, such as `q_table_episode_1000.pkl`, or a model from another run.
- Set `PIO_GUI=false` for testing without the Gazebo window.

Evaluation disables exploration and learning. Each trial lasts up to 300 seconds and ends early on collision. Results are saved under `src/pio_qn/test_results/`.

**The submitted ten-room evaluation results use checkpoint 900.** If `model_path` is omitted, the script instead loads `q_table.pkl` from the newest training run.
