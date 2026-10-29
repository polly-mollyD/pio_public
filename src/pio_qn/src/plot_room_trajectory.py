#!/usr/bin/env python
"""Plot a robot's test-run trajectory on top of its Gazebo room layout.

Standalone (no ROS/rospy import needed) -- just reads a room .world file
and a trajectory_run_*.csv produced by test_trained_model.py, and draws
the room walls, obstacles, and the path the robot took.

Usage:
    # Single room
    python3 plot_room_trajectory.py \
        --world path/to/room_trial_06.world \
        --trajectory path/to/test_room_6/trajectory_run_1.csv \
        --out room_6.png

    # Grid of multiple rooms (e.g. to recreate the experiment 3 figure)
    python3 plot_room_trajectory.py \
        --world path/to/room_trial_03.world path/to/room_trial_04.world ... \
        --trajectory path/to/test_room_3/trajectory_run_1.csv ... \
        --title "Room 3" "Room 4" ... \
        --out experiment3_room_paths.png
"""

import argparse
import csv
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# table_marble is a textured mesh, so its exact footprint isn't in the SDF.
# These are an approximation of its real-world footprint, good enough for
# illustrating where it sits and how it's rotated in the room.
TABLE_LENGTH = 1.4
TABLE_WIDTH = 0.7

# beer (used as the small "can" obstacles) is a simple cylinder; radius is
# read from the model, but it renders as a tiny dot rather than a scaled
# circle so it doesn't dominate a 5x5 m room plot.
CAN_MARKER_SIZE = 4

COLORS = {
    "box": "#c0c0c0",
    "cylinder": "#d2691e",
    "table": "#d2b48c",
    "can": "#8b0000",
    "path": "#1f77b4",
    "start": "#2ca02c",
    "collision": "#d62728",
    "end": "#000000",
}


def parse_pose(pose_text):
    values = [float(v) for v in pose_text.split()]
    x, y, _z, _roll, _pitch, yaw = values
    return x, y, yaw


def parse_world(world_path):
    """Return the room bounds and obstacle lists from a .world SDF file."""

    tree = ET.parse(world_path)
    world = tree.getroot().find("world")

    room = {"boxes": [], "cylinders": [], "tables": [], "cans": []}
    bounds = {"min_x": None, "max_x": None, "min_y": None, "max_y": None}

    for model in world.findall("model"):
        name = model.get("name")
        pose_el = model.find("pose")
        if pose_el is None:
            continue
        x, y, yaw = parse_pose(pose_el.text)

        box_size_el = model.find(".//geometry/box/size")
        cylinder_radius_el = model.find(".//geometry/cylinder/radius")

        if name.startswith("wall_"):
            sx, sy, _sz = (float(v) for v in box_size_el.text.split())
            half_x, half_y = sx / 2.0, sy / 2.0
            for corner_x in (x - half_x, x + half_x):
                if bounds["min_x"] is None or corner_x < bounds["min_x"]:
                    bounds["min_x"] = corner_x
                if bounds["max_x"] is None or corner_x > bounds["max_x"]:
                    bounds["max_x"] = corner_x
            for corner_y in (y - half_y, y + half_y):
                if bounds["min_y"] is None or corner_y < bounds["min_y"]:
                    bounds["min_y"] = corner_y
                if bounds["max_y"] is None or corner_y > bounds["max_y"]:
                    bounds["max_y"] = corner_y
            continue

        if box_size_el is not None:
            sx, sy, _sz = (float(v) for v in box_size_el.text.split())
            room["boxes"].append((x, y, yaw, sx, sy))
        elif cylinder_radius_el is not None:
            radius = float(cylinder_radius_el.text)
            room["cylinders"].append((x, y, radius))

    for include in world.findall("include"):
        uri = include.find("uri").text
        pose_el = include.find("pose")
        if pose_el is None:
            continue
        x, y, yaw = parse_pose(pose_el.text)

        if uri == "model://table_marble":
            room["tables"].append((x, y, yaw))
        elif uri == "model://beer":
            room["cans"].append((x, y))

    return bounds, room


def load_trajectory(trajectory_csv_path):
    with open(trajectory_csv_path, newline="") as trajectory_file:
        rows = list(csv.DictReader(trajectory_file))

    xs = [float(row["x"]) for row in rows]
    ys = [float(row["y"]) for row in rows]
    collisions = [row["collision"] == "True" for row in rows]

    return xs, ys, collisions


def draw_room(ax, bounds, room):
    ax.add_patch(Rectangle(
        (bounds["min_x"], bounds["min_y"]),
        bounds["max_x"] - bounds["min_x"],
        bounds["max_y"] - bounds["min_y"],
        fill=False, edgecolor="black", linewidth=1.5, zorder=1,
    ))

    for x, y, yaw, sx, sy in room["boxes"]:
        rect = Rectangle(
            (-sx / 2.0, -sy / 2.0), sx, sy,
            facecolor=COLORS["box"], edgecolor="black", zorder=2,
        )
        transform = (
            plt.matplotlib.transforms.Affine2D().rotate(yaw).translate(x, y)
            + ax.transData
        )
        rect.set_transform(transform)
        ax.add_patch(rect)

    for x, y, radius in room["cylinders"]:
        ax.add_patch(Circle(
            (x, y), radius,
            facecolor=COLORS["cylinder"], edgecolor="black", zorder=2,
        ))

    for x, y, yaw in room["tables"]:
        rect = Rectangle(
            (-TABLE_LENGTH / 2.0, -TABLE_WIDTH / 2.0), TABLE_LENGTH, TABLE_WIDTH,
            facecolor=COLORS["table"], edgecolor="black", zorder=2,
        )
        transform = (
            plt.matplotlib.transforms.Affine2D().rotate(yaw).translate(x, y)
            + ax.transData
        )
        rect.set_transform(transform)
        ax.add_patch(rect)

    for x, y in room["cans"]:
        ax.plot(x, y, "o", color=COLORS["can"], markersize=CAN_MARKER_SIZE, zorder=2)


def draw_trajectory(ax, xs, ys, collisions):
    ax.plot(xs, ys, "-", color=COLORS["path"], linewidth=1.2, zorder=3)
    ax.plot(xs[0], ys[0], "o", color=COLORS["start"], markersize=8, zorder=4)
    ax.plot(xs[-1], ys[-1], "s", color=COLORS["end"], markersize=7, zorder=4)

    for x, y, collided in zip(xs, ys, collisions):
        if collided:
            ax.plot(x, y, "x", color=COLORS["collision"], markersize=10,
                     markeredgewidth=2, zorder=5)


def build_legend_handles():
    return [
        plt.Line2D([0], [0], color=COLORS["path"], label="Robot path"),
        plt.Line2D([0], [0], marker="o", color=COLORS["start"], linestyle="",
                    markersize=8, label="Start"),
        plt.Line2D([0], [0], marker="s", color=COLORS["end"], linestyle="",
                    markersize=7, label="End"),
        plt.Line2D([0], [0], marker="x", color=COLORS["collision"], linestyle="",
                    markersize=10, markeredgewidth=2, label="Collision"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["box"], edgecolor="black",
                       label="Box obstacle"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["cylinder"], edgecolor="black",
                       label="Cylinder obstacle"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["table"], edgecolor="black",
                       label="Table"),
        plt.Line2D([0], [0], marker="o", color=COLORS["can"], linestyle="",
                    markersize=CAN_MARKER_SIZE, label="Can obstacle"),
    ]


def plot_room_trajectory(world_path, trajectory_csv_path, ax, title=None):
    bounds, room = parse_world(world_path)
    xs, ys, collisions = load_trajectory(trajectory_csv_path)

    draw_room(ax, bounds, room)
    draw_trajectory(ax, xs, ys, collisions)

    margin = 0.1
    ax.set_xlim(bounds["min_x"] - margin, bounds["max_x"] + margin)
    ax.set_ylim(bounds["min_y"] - margin, bounds["max_y"] + margin)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", nargs="+", required=True,
                         help="One or more room .world file paths.")
    parser.add_argument("--trajectory", nargs="+", required=True,
                         help="One trajectory_run_*.csv per --world, same order.")
    parser.add_argument("--title", nargs="+", default=None,
                         help="One subplot title per room. Defaults to 'Room N'.")
    parser.add_argument("--cols", type=int, default=5,
                         help="Subplot grid columns when plotting multiple rooms.")
    parser.add_argument("--out", default="room_paths.png",
                         help="Output image path.")
    args = parser.parse_args()

    if len(args.world) != len(args.trajectory):
        parser.error("--world and --trajectory must have the same count")

    titles = args.title or [
        "Room {}".format(i + 1) for i in range(len(args.world))
    ]
    if len(titles) != len(args.world):
        parser.error("--title must have the same count as --world")

    n = len(args.world)
    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for i, (world_path, trajectory_path, title) in enumerate(
        zip(args.world, args.trajectory, titles)
    ):
        plot_room_trajectory(world_path, trajectory_path, flat_axes[i], title=title)

    unused_axes = flat_axes[n:]
    for ax in unused_axes:
        ax.axis("off")

    if len(unused_axes):
        unused_axes[0].legend(handles=build_legend_handles(), loc="center",
                              frameon=False)
        fig.tight_layout()
    else:
        fig.legend(handles=build_legend_handles(), loc="lower center", ncol=4)
        fig.tight_layout(rect=[0, 0.18 / rows, 1, 1])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("Saved: " + args.out)


if __name__ == "__main__":
    main()
