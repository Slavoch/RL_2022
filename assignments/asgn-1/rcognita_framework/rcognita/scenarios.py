"""
This module contains various simulation scenarios.
For instance, an online scenario is when the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

"""

from rcognita.utilities import rc
from abc import ABC, abstractmethod


class TabularScenarioBase:
    """
    A tabular scenario blueprint.

    """

    def __init__(self, actor, critic, n_iters):
        self.actor = actor
        self.critic = critic
        self.n_iters = n_iters

    def run(self):
        for i in range(self.n_iters):
            self.iter()

    @abstractmethod
    def iter(self):
        pass


class TabularScenarioVI(TabularScenarioBase):
    """
    Tabular scenario for the use with tabular agents.
    Each iteration entails processing of the whole observation (or state) and action spaces, correponds to a signle update of the agent.
    Implements a scenario for value iteration (VI) update.

    """

    def iter(self):
        self.actor.update()
        self.critic.update()


class TabularScenarioPI(TabularScenarioBase):
    """
    Tabular scenario for the use with tabular agents.
    Each iteration entails processing of the whole observation (or state) and action spaces, correponds to a signle update of the agent.
    Implements a scenario for policy iteration (PI) update.
    """

    def iter(self):
        self.critic.update()
        self.actor.update()


class OnlineScenario:
    """
    Online scenario: the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

    """

    def __init__(
        self, simulator, controller, actor, critic, logger, datafiles, time_final
    ):
        self.simulator = simulator
        self.controller = controller
        self.actor = actor
        self.critic = critic
        self.logger = logger
        self.time_final = time_final
        self.accum_obj_val = 0
        self.datafile = datafiles[0]
        time = time_old = 0

        while True:

            self.my_simulator.sim_step()

            time_old = time

            (time, _, observation, state_full,) = self.simulator.get_sim_step_data()

            delta_time = time - time_old

            if self.save_trajectory:
                self.trajectory.append(rc.concatenate((state_full, time), axis=None))
            if self.control_mode == "nominal":
                action = self.controller.compute_action_sampled(time, observation)
            else:
                action = self.controller.compute_action_sampled(time, observation)

            self.my_system.receive_action(action)

            running_objective = self.running_objective(observation, action)
            self.upd_accum_obj(observation, action, delta_time)
            accum_obj = self.accum_obj_val

            if not self.no_print:
                self.logger.print_sim_step(
                    time, state_full, action, running_objective, accum_obj
                )

            if self.is_log:
                self.logger.log_data_row(
                    self.datafile,
                    time,
                    state_full,
                    action,
                    running_objective,
                    accum_obj,
                )
