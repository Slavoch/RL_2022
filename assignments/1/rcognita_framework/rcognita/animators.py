#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains an interface class `animator` along with concrete realizations, each of which is associated with a corresponding system.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
import numpy.linalg as la
from .utilities import upd_line
from .utilities import reset_line
from .utilities import upd_text
from .utilities import rc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# !pip install mpldatacursor <-- to install this
from mpldatacursor import datacursor

# !pip install svgpath2mpl matplotlib <-- to install this
from svgpath2mpl import parse_path

from collections import namedtuple


class Animator:
    """
    Interface class of visualization machinery for simulation of system-controller loops.
    To design a concrete animator: inherit this class, override:
        | :func:`~animators.Animator.__init__` :
        | define necessary visual elements (required)
        | :func:`~animators.Animator.init_anim` :
        | initialize necessary visual elements (required)
        | :func:`~animators.Animator.animate` :
        | animate visual elements (required)

    Attributes
    ----------
    objects : : tuple
        Objects to be updated within animation cycle
    pars : : tuple
        Fixed parameters of objects and visual elements

    """

    def __init__(self, objects=[], pars=[]):
        pass

    def init_anim(self):
        pass

    def animate(self, k):
        pass

    def get_anm(self, anm):
        """
        ``anm`` should be a ``FuncAnimation`` object.
        This method is needed to hand the animator access to the currently running animation, say, via ``anm.event_source.stop()``.

        """
        self.anm = anm

    def stop_anm(self):
        """
        Stops animation, provided that ``self.anm`` was defined via ``get_anm``.

        """
        self.anm.event_source.stop()
        # plt.close('all')
        raise Exception("exit")


class AnimatorGridWorld(Animator):
    def __init__(
        self,
        actor,
        critic,
        reward_cell_xy,
        starting_cell_xy,
        punishment_cells,
        scenario,
        n_iters=50,
    ):
        length = 3
        self.actions_map = {
            0: np.array([0.01 * length, 0]),
            1: np.array([-0.01 * length, 0]),
            2: np.array([0, 0.01 * length]),
            3: np.array([0, -0.01 * length]),
            4: np.array([0, 0]),
        }
        self.actor = actor
        self.critic = critic
        self.starting_cell_xy = starting_cell_xy
        self.reward_cell_xy = reward_cell_xy
        self.punishment_cells = punishment_cells
        self.scenario = scenario
        self.n_iters = n_iters

        self.colormap = plt.get_cmap("RdYlGn_r")

        self.fig_sim = plt.figure(figsize=(10, 10))

        self.ax = self.fig_sim.add_subplot(
            211,
            xlabel="red-green gradient corresponds to value (except for target and black cells).\nStarting cell is yellow",
            ylabel="",
            xlim=(0, 1.015),
            ylim=(-0.015, 1),
            facecolor="grey",
        )
        self.ax.set_title(label="Pause - space, q - quit", pad="20.0")
        self.ax.set_aspect("equal")
        self.arrows_patch_pack, self.rect_patch_pack, self.text_pack = self.create_grid(
            self.ax
        )
        normalize = mcolors.Normalize(vmin=70, vmax=100)
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=self.colormap)
        scalarmappaple.set_array(self.critic.model.weights)
        self.colorbar = plt.colorbar(scalarmappaple)

        self.ax_value_plot = self.fig_sim.add_subplot(
            212,
            autoscale_on=False,
            xlabel="Iteration",
            ylabel="Value",
            xlim=(0, self.n_iters),
            ylim=(-100, 100),
            title="Plot of the value at starting cell",
        )
        (self.line_value,) = self.ax_value_plot.plot(
            0,
            self.critic.model.weights[
                self.starting_cell_xy[0], self.starting_cell_xy[1]
            ],
            "g-",
            lw=1.5,
            label="Value",
        )
        plt.axhline(76, c="r", linestyle="--", label="Value optimal")
        plt.legend()

    def update_grid(self, iter):
        table = self.critic.model.weights
        shape = table.shape
        lenght = shape[0]
        width = shape[1]
        for i in range(lenght):
            for j in range(width):
                val = table[i, j]
                action = self.actor.model.weights[i, j]
                table_range = np.ptp(np.fmax(table, 70))
                color = self.colormap((val - np.max([np.min(table), 70])) / table_range)
                rectangle = self.rect_patch_pack[i * width + j]
                arr_x, arr_y = self.map_action2arrow(action, rectangle)

                arrow = self.arrows_patch_pack[i * width + j]
                arrow.set_positions(
                    (arr_x, arr_y),
                    (
                        arr_x + self.actions_map[action][0],
                        arr_y + self.actions_map[action][1],
                    ),
                )
                text = self.text_pack[i * width + j]
                if (
                    self.reward_cell_xy != [i, j]
                    and self.starting_cell_xy != [i, j]
                    and [i, j] not in self.punishment_cells
                ):
                    rectangle.set_facecolor(color)
                text.set_text(str(int(val)))

                if self.starting_cell_xy == [i, j]:
                    upd_line(self.line_value, iter, val)

    def create_grid(self, ax, space=0.01):
        table = self.critic.model.weights
        shape = table.shape
        lenght = shape[0]
        width = shape[1]
        rect_patch_pack = []
        arrows_patch_pack = []
        text_pack = []

        for i in range(lenght):
            for j in range(width):
                val = table[i, j]
                table_range = 200
                color = self.colormap(
                    ((val - np.min(table)) / table_range)
                    if np.abs(table_range) > 1e-3
                    else 1
                )
                if self.reward_cell_xy == [i, j]:
                    rectangle = patches.Rectangle(
                        (j / width + space, 1 - (i + 1) / lenght),
                        1 / lenght - space,
                        1 / width - space,
                        linewidth=2,
                        edgecolor="g",
                        facecolor="r",
                    )
                elif self.starting_cell_xy == [i, j]:
                    rectangle = patches.Rectangle(
                        (j / width + space, 1 - (i + 1) / lenght),
                        1 / lenght - space,
                        1 / width - space,
                        linewidth=2,
                        edgecolor="b",
                        facecolor="yellow",
                    )
                elif [i, j] in self.punishment_cells:
                    rectangle = patches.Rectangle(
                        (j / width + space, 1 - (i + 1) / lenght),
                        1 / lenght - space,
                        1 / width - space,
                        linewidth=2,
                        edgecolor="black",
                        facecolor="grey",
                    )
                else:
                    rectangle = patches.Rectangle(
                        (j / width + space, 1 - (i + 1) / lenght),
                        1 / lenght - space,
                        1 / width - space,
                        linewidth=2,
                        edgecolor="g",
                        facecolor=color,
                    )
                rx, ry = rectangle.get_xy()
                cx = rx + rectangle.get_width() / 2.0
                cy = ry + rectangle.get_height() / 2.0
                text = ax.text(
                    cx,
                    cy,
                    str(np.floor(val)),
                    color="black",
                    weight="bold",
                    fontsize=7,
                    ha="center",
                    va="center",
                )
                text.set_path_effects(
                    [PathEffects.withStroke(linewidth=2, foreground="w")]
                )
                ax.add_patch(rectangle)
                rect_patch_pack.append(rectangle)
                text_pack.append(text)
                ax.set(xticks=[], yticks=[])
                action = self.actor.model.weights[i, j]
                arr_x, arr_y = self.map_action2arrow(action, rectangle)

                pos_head = self.actions_map[action]
                arrowstyle = patches.ArrowStyle.Fancy(
                    head_length=0.4, head_width=1, tail_width=4
                )
                if self.reward_cell_xy == [i, j]:
                    arrowstyle = patches.ArrowStyle.Fancy(
                        head_length=0, head_width=1, tail_width=1
                    )
                arrow = patches.FancyArrowPatch(
                    (arr_x, arr_y),
                    (arr_x + pos_head[0], arr_y + pos_head[1]),
                    arrowstyle=arrowstyle,
                )
                # arrow.set_arrowstyle("fancy", head_length=0.05)
                ax.add_patch(arrow)
                arrows_patch_pack.append(arrow)
                if i == 0:
                    text = ax.text(cx, 1.03, f"{j}", ha="center", va="center")
                if j == 0:
                    text = ax.text(-0.03, cy, f"{i}", ha="center", va="center")

        return arrows_patch_pack, rect_patch_pack, text_pack

    def map_action2arrow(self, action, rectangle):
        rx, ry = rectangle.get_xy()
        if action == 0:
            arr_x = rx + rectangle.get_width() * (7 / 10)
            arr_y = ry + rectangle.get_height() / 2
        elif action == 1:
            arr_x = rx + rectangle.get_width() * (3 / 10)
            arr_y = ry + rectangle.get_height() / 2
        elif action == 2:
            arr_x = rx + rectangle.get_width() / 2
            arr_y = ry + rectangle.get_height() * (7 / 10)
        elif action == 3:
            arr_x = rx + rectangle.get_width() / 2
            arr_y = ry + rectangle.get_height() * (3 / 10)
        elif action == 4:
            arr_x = rx + rectangle.get_width() / 2
            arr_y = ry + rectangle.get_height() / 2

        return arr_x, arr_y

    def animate(self, k):
        self.scenario.iter()
        self.update_grid(k)


class RobotMarker:
    """
    Robot marker for visualization.

    """

    def __init__(self, angle=None, path_string=None):
        self.angle = angle or []
        self.path_string = (
            path_string
            or """m 66.893258,227.10128 h 5.37899 v 0.91881 h 1.65571 l 1e-5,-3.8513 3.68556,-1e-5 v -1.43933
        l -2.23863,10e-6 v -2.73937 l 5.379,-1e-5 v 2.73938 h -2.23862 v 1.43933 h 3.68556 v 8.60486 l -3.68556,1e-5 v 1.43158
        h 2.23862 v 2.73989 h -5.37899 l -1e-5,-2.73989 h 2.23863 v -1.43159 h -3.68556 v -3.8513 h -1.65573 l 1e-5,0.91881 h -5.379 z"""
        )
        self.path = parse_path(self.path_string)
        self.path.vertices -= self.path.vertices.mean(axis=0)
        self.marker = mpl.markers.MarkerStyle(marker=self.path)
        self.marker._transform = self.marker.get_transform().rotate_deg(angle)

    def rotate(self, angle=0):
        self.marker._transform = self.marker.get_transform().rotate_deg(
            angle - self.angle
        )
        self.angle = angle


from pipelines import pipeline_3wrobot, pipeline_3wrobot_NI, pipeline_2tank


class Animator3WRobot(Animator, pipeline_3wrobot.Pipeline3WRobot):
    """
    Animator class for a 3-wheel robot with dynamic actuators.

    """

    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars

        # Unpack entities
        (
            self.simulator,
            self.sys,
            self.nominal_controller,
            self.controller,
            self.datafiles,
            self.logger,
            self.actor_optimizer,
            self.critic_optimizer,
            self.running_objective,
        ) = self.objects

        (
            state_init,
            action_init,
            time_start,
            time_final,
            state_full_init,
            xMin,
            xMax,
            yMin,
            yMax,
            control_mode,
            action_manual,
            Fmin,
            Mmin,
            Fmax,
            Mmax,
            Nruns,
            no_print,
            is_log,
            is_playback,
            running_obj_init,
        ) = self.pars

        # Store some parameters for later use
        self.t_old = 0
        self.accum_obj_val = 0
        self.time_start = time_start
        self.state_full_init = state_full_init
        self.time_final = time_final
        self.control_mode = control_mode
        self.action_manual = action_manual
        self.Nruns = Nruns
        self.no_print = no_print
        self.is_log = is_log
        self.is_playback = is_playback

        xCoord0 = state_init[0]
        yCoord0 = state_init[1]
        alpha0 = state_init[2]
        alpha_deg0 = alpha0 / 2 / np.pi

        plt.close("all")

        self.fig_sim = plt.figure(figsize=(10, 10))

        # xy plane
        self.axs_xy_plane = self.fig_sim.add_subplot(
            221,
            autoscale_on=False,
            xlim=(xMin, xMax),
            ylim=(yMin, yMax),
            xlabel="x [m]",
            ylabel="y [m]",
            title="Pause - space, q - quit, click - data cursor",
        )
        self.axs_xy_plane.set_aspect("equal", adjustable="box")
        self.axs_xy_plane.plot([xMin, xMax], [0, 0], "k--", lw=0.75)  # Help line
        self.axs_xy_plane.plot([0, 0], [yMin, yMax], "k--", lw=0.75)  # Help line
        (self.line_traj,) = self.axs_xy_plane.plot(xCoord0, yCoord0, "b--", lw=0.5)
        self.robot_marker = RobotMarker(angle=alpha_deg0)
        text_time = "Time = {time:2.3f}".format(time=time_start)
        self.text_time_handle = self.axs_xy_plane.text(
            0.05,
            0.95,
            text_time,
            horizontalalignment="left",
            verticalalignment="center",
            transform=self.axs_xy_plane.transAxes,
        )
        self.axs_xy_plane.format_coord = lambda state, observation: "%2.2f, %2.2f" % (
            state,
            observation,
        )

        # Solution
        self.axs_sol = self.fig_sim.add_subplot(
            222,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(2 * np.min([xMin, yMin]), 2 * np.max([xMax, yMax])),
            xlabel="Time [s]",
        )
        self.axs_sol.plot([time_start, time_final], [0, 0], "k--", lw=0.75)  # Help line
        (self.line_norm,) = self.axs_sol.plot(
            time_start,
            la.norm([xCoord0, yCoord0]),
            "b-",
            lw=0.5,
            label=r"$\Vert(x,y)\Vert$ [m]",
        )
        (self.line_alpha,) = self.axs_sol.plot(
            time_start, alpha0, "r-", lw=0.5, label=r"$\alpha$ [rad]"
        )
        self.axs_sol.legend(fancybox=True, loc="upper right")
        self.axs_sol.format_coord = lambda state, observation: "%2.2f, %2.2f" % (
            state,
            observation,
        )

        # Cost
        if is_playback:
            running_objective = running_obj_init
        else:
            observation_init = self.sys.out(state_init)
            running_objective = self.running_objective(observation_init, action_init)

        self.axs_cost = self.fig_sim.add_subplot(
            223,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(0, 1e4 * running_objective),
            yscale="symlog",
            xlabel="Time [s]",
        )

        text_accum_obj = (
            r"$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.3f}".format(
                accum_obj=0
            )
        )
        self.text_accum_obj_handle = self.fig_sim.text(
            0.05,
            0.5,
            text_accum_obj,
            horizontalalignment="left",
            verticalalignment="center",
        )
        (self.line_running_obj,) = self.axs_cost.plot(
            time_start, running_objective, "r-", lw=0.5, label="Stage obj."
        )
        (self.line_accum_obj,) = self.axs_cost.plot(
            time_start,
            0,
            "g-",
            lw=0.5,
            label=r"$\int \mathrm{Stage\,obj.} \,\mathrm{d}t$",
        )
        self.axs_cost.legend(fancybox=True, loc="upper right")

        # Control
        self.axs_action = self.fig_sim.add_subplot(
            224,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(1.1 * np.min([Fmin, Mmin]), 1.1 * np.max([Fmax, Mmax])),
            xlabel="Time [s]",
        )
        self.axs_action.plot(
            [time_start, time_final], [0, 0], "k--", lw=0.75
        )  # Help line
        self.lines_action = self.axs_action.plot(
            time_start, rc.to_col(action_init).T, lw=0.5
        )
        self.axs_action.legend(
            iter(self.lines_action),
            ("F [N]", "M [Nm]"),
            fancybox=True,
            loc="upper right",
        )

        # Pack all lines together
        cLines = namedtuple(
            "lines",
            [
                "line_traj",
                "line_norm",
                "line_alpha",
                "line_running_obj",
                "line_accum_obj",
                "lines_action",
            ],
        )
        self.lines = cLines(
            line_traj=self.line_traj,
            line_norm=self.line_norm,
            line_alpha=self.line_alpha,
            line_running_obj=self.line_running_obj,
            line_accum_obj=self.line_accum_obj,
            lines_action=self.lines_action,
        )

        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)

    def set_sim_data(
        self, ts, xCoords, yCoords, alphas, vs, omegas, rs, accum_objs, Fs, Ms
    ):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """
        self.ts, self.xCoords, self.yCoords, self.alphas, self.vs, self.omegas = (
            ts,
            xCoords,
            yCoords,
            alphas,
            vs,
            omegas,
        )
        self.rs, self.accum_objs, self.Fs, self.Ms = rs, accum_objs, Fs, Ms
        self.curr_step = 0

    def upd_sim_data_row(self):
        self.time = self.ts[self.curr_step]
        self.state_full = np.array(
            [
                self.xCoords[self.curr_step],
                self.yCoords[self.curr_step],
                self.alphas[self.curr_step],
                self.vs[self.curr_step],
                self.omegas[self.curr_step],
            ]
        )
        self.running_objective = self.rs[self.curr_step]
        self.accum_obj = self.accum_objs[self.curr_step]
        self.action = np.array([self.Fs[self.curr_step], self.Ms[self.curr_step]])

        self.curr_step = self.curr_step + 1

    def init_anim(self):
        state_init, *_ = self.pars

        xCoord0 = state_init[0]
        yCoord0 = state_init[1]

        self.scatter_sol = self.axs_xy_plane.scatter(
            xCoord0, yCoord0, marker=self.robot_marker.marker, s=400, c="b"
        )
        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]

    def animate(self, k):

        if self.is_playback:
            self.upd_sim_data_row()
            time = self.time
            state_full = self.state_full
            action = self.action
            running_objective = self.running_objective
            accum_obj = self.accum_obj

        else:
            self.simulator.sim_step()

            time, state, observation, state_full = self.simulator.get_sim_step_data()

            delta_t = time - self.t_old

            self.t_old = time

            if self.control_mode == "nominal":
                action = self.nominal_controller.compute_action_sampled(
                    time, observation
                )
            else:
                action = self.controller.compute_action_sampled(
                    time,
                    observation,
                )

            self.sys.receive_action(action)

        xCoord = state_full[0]
        yCoord = state_full[1]
        alpha = state_full[2]
        alpha_deg = alpha / np.pi * 180
        v = state_full[3]
        omega = state_full[4]

        running_objective = self.running_objective(observation, action)
        self.upd_accum_obj(observation, action, delta_t)
        accum_obj = self.accum_obj_val

        if not self.no_print:
            self.logger.print_sim_step(
                time, state_full, action, running_objective, accum_obj
            )

        if self.is_log:
            self.logger.log_data_row(
                self.datafile_curr,
                time,
                state_full,
                action,
                running_objective,
                accum_obj,
            )

        # xy plane
        text_time = "Time = {time:2.3f}".format(time=time)
        upd_text(self.text_time_handle, text_time)
        upd_line(self.line_traj, xCoord, yCoord)  # Update the robot's track on the plot

        self.robot_marker.rotate(1e-3)  # Rotate the robot on the plot
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(
            5, 5, marker=self.robot_marker.marker, s=400, c="b"
        )

        self.robot_marker.rotate(alpha_deg)  # Rotate the robot on the plot
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(
            xCoord, yCoord, marker=self.robot_marker.marker, s=400, c="b"
        )

        # # Solution
        upd_line(self.line_norm, time, la.norm([xCoord, yCoord]))
        upd_line(self.line_alpha, time, alpha)

        # Cost
        upd_line(self.line_running_obj, time, running_objective)
        upd_line(self.line_accum_obj, time, accum_obj)
        text_accum_obj = (
            r"$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.1f}".format(
                accum_obj=np.squeeze(np.array(accum_obj))
            )
        )
        upd_text(self.text_accum_obj_handle, text_accum_obj)

        # Control
        for (line, action_single) in zip(self.lines_action, np.array(action)):
            upd_line(line, time, action_single)

        # Run done
        if time >= self.time_final:
            if not self.no_print:
                print(
                    ".....................................Run {run:2d} done.....................................".format(
                        run=self.run_curr
                    )
                )

            self.run_curr += 1

            if self.run_curr > self.Nruns:
                print("Animation done...")
                self.stop_anm()
                return

            if self.is_log:
                self.datafile_curr = self.datafiles[self.run_curr - 1]

            # Reset simulator
            self.simulator.reset()

            # Reset controller
            if self.control_mode > 0:
                self.controller.reset(self.time_start)
            else:
                self.nominal_controller.reset(self.time_start)

            accum_obj = 0

            reset_line(self.line_norm)
            reset_line(self.line_alpha)
            reset_line(self.line_running_obj)
            reset_line(self.line_accum_obj)
            reset_line(self.lines_action[0])
            reset_line(self.lines_action[1])

            # for item in self.lines:
            #     if item != self.line_traj:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)

            upd_line(self.line_traj, np.nan, np.nan)


class Animator3WRobotNI(Animator, pipeline_3wrobot_NI.Pipeline3WRobotNI):
    """
    Animator class for a 3-wheel robot with static actuators.

    """

    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars

        # Unpack entities
        (
            self.simulator,
            self.sys,
            self.nominal_controller,
            self.controller,
            self.datafiles,
            self.logger,
            self.actor_optimizer,
            self.optimizer,
            self.running_objective,
        ) = self.objects

        (
            state_init,
            action_init,
            time_start,
            time_final,
            state_full_init,
            xMin,
            xMax,
            yMin,
            yMax,
            control_mode,
            action_manual,
            v_min,
            omega_min,
            v_max,
            omega_max,
            Nruns,
            no_print,
            is_log,
            is_playback,
            running_obj_init,
        ) = self.pars

        # Store some parameters for later use
        self.t_old = 0
        self.accum_obj_val = 0
        self.time_start = time_start
        self.state_full_init = state_full_init
        self.time_final = time_final
        self.control_mode = control_mode
        self.action_manual = action_manual
        self.Nruns = Nruns
        self.no_print = no_print
        self.is_log = is_log
        self.is_playback = is_playback

        xCoord0 = state_init[0]
        yCoord0 = state_init[1]
        alpha0 = state_init[2]
        alpha_deg0 = alpha0 / 2 / np.pi

        plt.close("all")

        self.fig_sim = plt.figure(figsize=(10, 10))

        # xy plane
        self.axs_xy_plane = self.fig_sim.add_subplot(
            221,
            autoscale_on=False,
            xlim=(xMin, xMax),
            ylim=(yMin, yMax),
            xlabel="x [m]",
            ylabel="y [m]",
            title="Pause - space, q - quit, click - data cursor",
        )
        self.axs_xy_plane.set_aspect("equal", adjustable="box")
        self.axs_xy_plane.plot([xMin, xMax], [0, 0], "k--", lw=0.75)  # Help line
        self.axs_xy_plane.plot([0, 0], [yMin, yMax], "k--", lw=0.75)  # Help line
        (self.line_traj,) = self.axs_xy_plane.plot(xCoord0, yCoord0, "b--", lw=0.5)
        self.robot_marker = RobotMarker(angle=alpha_deg0)
        text_time = "Time = {time:2.3f}".format(time=time_start)
        self.text_time_handle = self.axs_xy_plane.text(
            0.05,
            0.95,
            text_time,
            horizontalalignment="left",
            verticalalignment="center",
            transform=self.axs_xy_plane.transAxes,
        )
        self.axs_xy_plane.format_coord = lambda state, observation: "%2.2f, %2.2f" % (
            state,
            observation,
        )

        # Solution
        self.axs_sol = self.fig_sim.add_subplot(
            222,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(2 * np.min([xMin, yMin]), 2 * np.max([xMax, yMax])),
            xlabel="Time [s]",
        )
        self.axs_sol.plot([time_start, time_final], [0, 0], "k--", lw=0.75)  # Help line
        (self.line_norm,) = self.axs_sol.plot(
            time_start,
            la.norm([xCoord0, yCoord0]),
            "b-",
            lw=0.5,
            label=r"$\Vert(x,y)\Vert$ [m]",
        )
        (self.line_alpha,) = self.axs_sol.plot(
            time_start, alpha0, "r-", lw=0.5, label=r"$\alpha$ [rad]"
        )
        self.axs_sol.legend(fancybox=True, loc="upper right")
        self.axs_sol.format_coord = lambda state, observation: "%2.2f, %2.2f" % (
            state,
            observation,
        )

        # Cost
        if is_playback:
            running_objective = running_obj_init
        else:
            observation_init = self.sys.out(state_init)
            running_objective = self.running_objective(observation_init, action_init)

        self.axs_cost = self.fig_sim.add_subplot(
            223,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(0, 1e4 * running_objective),
            yscale="symlog",
            xlabel="Time [s]",
        )

        text_accum_obj = (
            r"$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.3f}".format(
                accum_obj=0
            )
        )
        self.text_accum_obj_handle = self.fig_sim.text(
            0.05,
            0.5,
            text_accum_obj,
            horizontalalignment="left",
            verticalalignment="center",
        )
        (self.line_running_obj,) = self.axs_cost.plot(
            time_start, running_objective, "r-", lw=0.5, label="Stage obj."
        )
        (self.line_accum_obj,) = self.axs_cost.plot(
            time_start,
            0,
            "g-",
            lw=0.5,
            label=r"$\int \mathrm{Stage\,obj.} \,\mathrm{d}t$",
        )
        self.axs_cost.legend(fancybox=True, loc="upper right")

        # Control
        self.axs_action = self.fig_sim.add_subplot(
            224,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(1.1 * np.min([v_min, omega_min]), 1.1 * np.max([v_max, omega_max])),
            xlabel="Time [s]",
        )
        self.axs_action.plot(
            [time_start, time_final], [0, 0], "k--", lw=0.75
        )  # Help line
        self.lines_action = self.axs_action.plot(
            time_start, rc.to_col(action_init).T, lw=0.5
        )
        self.axs_action.legend(
            iter(self.lines_action),
            ("v [m/s]", r"$\omega$ [rad/s]"),
            fancybox=True,
            loc="upper right",
        )

        # Pack all lines together
        cLines = namedtuple(
            "lines",
            [
                "line_traj",
                "line_norm",
                "line_alpha",
                "line_running_obj",
                "line_accum_obj",
                "lines_action",
            ],
        )
        self.lines = cLines(
            line_traj=self.line_traj,
            line_norm=self.line_norm,
            line_alpha=self.line_alpha,
            line_running_obj=self.line_running_obj,
            line_accum_obj=self.line_accum_obj,
            lines_action=self.lines_action,
        )

        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)

    def set_sim_data(self, ts, xCoords, yCoords, alphas, rs, accum_objs, vs, omegas):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """
        self.ts, self.xCoords, self.yCoords, self.alphas = ts, xCoords, yCoords, alphas
        self.rs, self.accum_objs, self.vs, self.omegas = rs, accum_objs, vs, omegas
        self.curr_step = 0

    def upd_sim_data_row(self):
        self.time = self.ts[self.curr_step]
        self.state_full = np.array(
            [
                self.xCoords[self.curr_step],
                self.yCoords[self.curr_step],
                self.alphas[self.curr_step],
            ]
        )
        self.running_objective = self.rs[self.curr_step]
        self.accum_obj = self.accum_objs[self.curr_step]
        self.action = np.array([self.vs[self.curr_step], self.omegas[self.curr_step]])

        self.curr_step = self.curr_step + 1

    def init_anim(self):
        state_init, *_ = self.pars

        xCoord0 = state_init[0]
        yCoord0 = state_init[1]

        self.scatter_sol = self.axs_xy_plane.scatter(
            xCoord0, yCoord0, marker=self.robot_marker.marker, s=400, c="b"
        )
        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]

    def animate(self, k):

        if self.is_playback:
            self.upd_sim_data_row()
            time = self.time
            state_full = self.state_full
            action = self.action
            running_objective = self.running_objective
            accum_obj = self.accum_obj

        else:
            self.simulator.sim_step()

            time, state, observation, state_full = self.simulator.get_sim_step_data()

            delta_time = time - self.t_old

            self.time_old = time

            if self.control_mode == "nominal":
                action = self.nominal_controller.compute_action_sampled(
                    time, observation
                )
            else:
                action = self.controller.compute_action_sampled(time, observation)
            # if rc_type:
            #     action = np.array(action).reshape(-1)
            self.sys.receive_action(action)

        xCoord = state_full[0]
        yCoord = state_full[1]
        alpha = state_full[2]
        alpha_deg = alpha / np.pi * 180

        running_objective = self.running_objective(observation, action)
        self.upd_accum_obj(observation, action, delta_time)
        accum_obj = self.accum_obj_val

        if not self.no_print:
            self.logger.print_sim_step(
                time,
                state_full,
                action,
                running_objective,
                accum_obj,
            )

        if self.is_log:
            self.logger.log_data_row(
                self.datafile_curr,
                time,
                state_full,
                action,
                running_objective,
                accum_obj,
            )

        # xy plane
        text_time = "t = {time:2.3f}".format(time=time)
        upd_text(self.text_time_handle, text_time)
        upd_line(self.line_traj, xCoord, yCoord)  # Update the robot's track on the plot

        self.robot_marker.rotate(1e-3)  # Rotate the robot on the plot
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(
            5, 5, marker=self.robot_marker.marker, s=400, c="b"
        )

        self.robot_marker.rotate(alpha_deg)  # Rotate the robot on the plot
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(
            xCoord, yCoord, marker=self.robot_marker.marker, s=400, c="b"
        )

        # # Solution
        upd_line(self.line_norm, time, la.norm([xCoord, yCoord]))
        upd_line(self.line_alpha, time, alpha)

        # Cost
        upd_line(self.line_running_obj, time, running_objective)
        upd_line(self.line_accum_obj, time, accum_obj)
        text_accum_obj = (
            r"$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.1f}".format(
                accum_obj=np.squeeze(np.array(accum_obj))
            )
        )
        upd_text(self.text_accum_obj_handle, text_accum_obj)

        # Control
        for (line, action_single) in zip(self.lines_action, np.array(action)):
            upd_line(line, time, action_single)

        # Run done
        if time >= self.time_final:
            if not self.no_print:
                print(
                    ".....................................Run {run:2d} done.....................................".format(
                        run=self.run_curr
                    )
                )

            self.run_curr += 1

            if self.run_curr > self.Nruns:
                print("Animation done...")
                self.stop_anm()
                return

            if self.is_log:
                self.datafile_curr = self.datafiles[self.run_curr - 1]

            # Reset simulator
            self.simulator.reset()

            # Reset controller
            if self.control_mode > 0:
                self.controller.reset(self.time_start)
            else:
                self.nominal_controller.reset(self.time_start)

            accum_obj = 0

            reset_line(self.line_norm)
            reset_line(self.line_alpha)
            reset_line(self.line_running_obj)
            reset_line(self.line_accum_obj)
            reset_line(self.lines_action[0])
            reset_line(self.lines_action[1])

            # for item in self.lines:
            #     if item != self.line_traj:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)

            upd_line(self.line_traj, np.nan, np.nan)


class Animator2Tank(Animator, pipeline_2tank.Pipeline2Tank):
    """
    Animator class for a 2-tank system.

    """

    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars

        # Unpack entities
        (
            self.simulator,
            self.sys,
            self.nominal_controller,
            self.controller,
            self.datafiles,
            self.logger,
        ) = self.objects

        (
            state_init,
            action_init,
            time_start,
            time_final,
            state_full_init,
            control_mode,
            action_manual,
            action_min,
            action_max,
            Nruns,
            no_print,
            is_log,
            is_playback,
            running_obj_init,
            level_target,
        ) = self.pars

        # Store some parameters for later use
        self.time_start = time_start
        self.state_full_init = state_full_init
        self.time_final = time_final
        self.control_mode = control_mode
        self.action_manual = action_manual
        self.Nruns = Nruns
        self.no_print = no_print
        self.is_log = is_log
        self.is_playback = is_playback

        self.level_target = level_target

        h1_0 = state_init[0]
        h2_0 = state_init[1]
        p0 = action_init

        plt.close("all")

        self.fig_sim = plt.figure(figsize=(10, 10))

        # h1, h2 plot
        self.axs_sol = self.fig_sim.add_subplot(
            221,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(-2, 2),
            xlabel="Time [s]",
            title="Pause - space, q - quit, click - data cursor",
        )
        self.axs_sol.plot([time_start, time_final], [0, 0], "k--", lw=0.75)  # Help line
        self.axs_sol.plot(
            [time_start, time_final], [level_target[0], level_target[0]], "b--", lw=0.75
        )  # Help line (target)
        self.axs_sol.plot(
            [time_start, time_final], [level_target[1], level_target[1]], "r--", lw=0.75
        )  # Help line (target)
        (self.line_h1,) = self.axs_sol.plot(
            time_start, h1_0, "b-", lw=0.5, label=r"$h_1$"
        )
        (self.line_h2,) = self.axs_sol.plot(
            time_start, h2_0, "r-", lw=0.5, label=r"$h_2$"
        )
        self.axs_sol.legend(fancybox=True, loc="upper right")
        self.axs_sol.format_coord = lambda state, observation: "%2.2f, %2.2f" % (
            state,
            observation,
        )

        # Cost
        if is_playback:
            running_objective = running_obj_init
        else:
            observation_init = self.sys.out(state_init)
            running_objective = self.controller.running_objective(
                observation_init, action_init
            )

        self.axs_cost = self.fig_sim.add_subplot(
            223,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(0, 1e4 * running_objective),
            yscale="symlog",
            xlabel="Time [s]",
        )

        text_accum_obj = (
            r"$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.3f}".format(
                accum_obj=0
            )
        )
        self.text_accum_obj_handle = self.fig_sim.text(
            0.05,
            0.5,
            text_accum_obj,
            horizontalalignment="left",
            verticalalignment="center",
        )
        (self.line_running_obj,) = self.axs_cost.plot(
            time_start, running_objective, "r-", lw=0.5, label="Stage obj."
        )
        (self.line_accum_obj,) = self.axs_cost.plot(
            time_start,
            0,
            "g-",
            lw=0.5,
            label=r"$\int \mathrm{Stage\,obj.} \,\mathrm{d}t$",
        )
        self.axs_cost.legend(fancybox=True, loc="upper right")

        # Control
        self.axs_action = self.fig_sim.add_subplot(
            222,
            autoscale_on=False,
            xlim=(time_start, time_final),
            ylim=(action_min - 0.1, action_max + 0.1),
            xlabel="Time [s]",
        )
        self.axs_action.plot(
            [time_start, time_final], [0, 0], "k--", lw=0.75
        )  # Help line
        (self.line_action,) = self.axs_action.plot(time_start, p0, lw=0.5, label="p")
        self.axs_action.legend(fancybox=True, loc="upper right")

        # Pack all lines together
        cLines = namedtuple(
            "lines",
            ["line_h1", "line_h2", "line_running_obj", "line_accum_obj", "line_action"],
        )
        self.lines = cLines(
            line_h1=self.line_h1,
            line_h2=self.line_h2,
            line_running_obj=self.line_running_obj,
            line_accum_obj=self.line_accum_obj,
            line_action=self.line_action,
        )

        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)

    def set_sim_data(self, ts, h1s, h2s, ps, rs, accum_objs):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """
        self.ts, self.h1s, self.h2s, self.ps = ts, h1s, h2s, ps
        self.rs, self.accum_objs = rs, accum_objs
        self.curr_step = 0

    def upd_sim_data_row(self):
        self.time = self.ts[self.curr_step]
        self.state_full = np.array([self.h1s[self.curr_step], self.h2s[self.curr_step]])
        self.running_objective = self.rs[self.curr_step]
        self.accum_obj = self.accum_objs[self.curr_step]
        self.action = np.array([self.ps[self.curr_step]])

        self.curr_step = self.curr_step + 1

    def init_anim(self):
        state_init, *_ = self.pars

        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]

    def animate(self, k):

        if self.is_playback:
            self.upd_sim_data_row()
            time = self.time
            state_full = self.state_full
            action = self.action
            running_objective = self.running_objective
            accum_obj = self.accum_obj

        else:
            self.simulator.sim_step()

            time, state, observation, state_full = self.simulator.get_sim_step_data()

            action = self.controller.compute_action(time, observation)

            self.sys.receive_action(action)
            self.controller.upd_accum_obj(observation, action)

            running_objective = self.controller.running_objective(observation, action)
            accum_obj = self.controller.accum_obj_val

        h1 = state_full[0]
        h2 = state_full[1]
        p = action

        if not self.no_print:
            self.logger.print_sim_step(time, h1, h2, p, running_objective, accum_obj)

        if self.is_log:
            self.logger.log_data_row(
                self.datafile_curr, time, h1, h2, p, running_objective, accum_obj
            )

        # # Solution
        upd_line(self.line_h1, time, h1)
        upd_line(self.line_h2, time, h2)

        # Cost
        upd_line(self.line_running_obj, time, running_objective)
        upd_line(self.line_accum_obj, time, accum_obj)
        text_accum_obj = (
            r"$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.1f}".format(
                accum_obj=np.squeeze(np.array(accum_obj))
            )
        )
        upd_text(self.text_accum_obj_handle, text_accum_obj)

        # Control
        upd_line(self.line_action, time, p)

        # Run done
        if time >= self.time_final:
            if not self.no_print:
                print(
                    ".....................................Run {run:2d} done.....................................".format(
                        run=self.run_curr
                    )
                )

            self.run_curr += 1

            if self.run_curr > self.Nruns:
                print("Animation done...")
                self.stop_anm()
                return

            if self.is_log:
                self.datafile_curr = self.datafiles[self.run_curr - 1]

            # Reset simulator
            self.simulator.reset()

            # Reset controller
            if self.control_mode > 0:
                self.controller.reset(self.time_start)
            else:
                self.nominal_controller.reset(self.time_start)

            accum_obj = 0

            reset_line(self.line_h1)
            reset_line(self.line_h1)
            reset_line(self.line_action)
            reset_line(self.line_running_obj)
            reset_line(self.line_accum_obj)

            # for item in self.lines:
            #     if item != self.line_traj:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)

            # upd_line(self.line_h1, np.nan)
