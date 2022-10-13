"""
This module contains various simulation scenarios.
For instance, an online scenario is when the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

"""

from re import S
from rcognita.utilities import rc
from abc import ABC, abstractmethod
from copy import deepcopy
import matplotlib.pyplot as plt
import sys


class TabularScenarioBase:
    """
    A tabular scenario blueprint.

    """

    def __init__(self, actor, critic, N_iterations):
        self.actor = actor
        self.critic = critic
        self.N_iterations = N_iterations

    def run(self):
        for i in range(self.N_iterations):
            self.iterate()

    @abstractmethod
    def iterate(self):
        pass


class TabularScenarioVI(TabularScenarioBase):
    """
    Tabular scenario for the use with tabular agents.
    Each iteration entails processing of the whole observation (or state) and action spaces, correponds to a signle update of the agent.
    Implements a scenario for value iteration (VI) update.

    """

    def iterate(self):
        self.actor.update()
        self.critic.update()


class TabularScenarioPI(TabularScenarioBase):
    """
    Tabular scenario for the use with tabular agents.
    Each iteration entails processing of the whole observation (or state) and action spaces, correponds to a signle update of the agent.
    Implements a scenario for policy iteration (PI) update.
    """

    def iterate(self):
        self.critic.update()
        self.actor.update()


class OnlineScenario:
    """
    Online scenario: the controller and system interact with each other live via exchange of observations and actions, successively in time steps.

    """

    def __init__(
        self,
        system,
        simulator,
        controller,
        actor,
        critic,
        logger,
        datafiles,
        time_final,
        running_objective,
        no_print=False,
        is_log=False,
        is_playback=False,
        state_init=None,
        action_init=None,
    ):
        self.system = system
        self.simulator = simulator
        self.controller = controller
        self.actor = actor
        self.critic = critic
        self.logger = logger
        self.time_final = time_final
        self.outcome = 0
        self.datafile = datafiles[0]
        self.running_objective = running_objective
        self.trajectory = []
        self.no_print = no_print
        self.is_log = is_log
        self.time_old = 0
        self.is_playback = is_playback
        self.state_init = state_init
        self.action_init = action_init
        if self.is_playback:
            self.table = []

    def run(self):
        while True:
            self.step()

    def step(self):
        sim_status = self.simulator.do_sim_step()
        if sim_status == -1:
            return -1

        (
            self.time,
            _,
            self.observation,
            self.state_full,
        ) = self.simulator.get_sim_step_data()
        self.trajectory.append(rc.concatenate((self.state_full, self.time), axis=None))
        delta_time = self.time - self.time_old
        self.time_old = self.time

        self.action = self.controller.compute_action_sampled(
            self.time, self.observation
        )
        self.system.receive_action(self.action)

        self.running_objective_value = self.running_objective(
            self.observation, self.action
        )
        self.update_outcome(self.observation, self.action, delta_time)

        if not self.no_print:
            self.logger.print_sim_step(
                self.time,
                self.state_full,
                self.action,
                self.running_objective_value,
                self.outcome,
            )
        if self.is_log:
            self.logger.log_data_row(
                self.datafile,
                self.time,
                self.state_full,
                self.action,
                self.running_objective_value,
                self.outcome,
            )

        if self.is_playback:
            self.table.append(
                [
                    self.time,
                    *self.state_full,
                    *self.action,
                    self.running_objective_value,
                    self.outcome,
                ]
            )

    def update_outcome(self, observation, action, delta):

        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``outcome`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize objective instead).

        """

        self.outcome += self.running_objective(observation, action) * delta


class EpisodicScenario(OnlineScenario):
    def __init__(self, N_episodes, N_iterations, *args, learning_rate=0.001, **kwargs):
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.learning_rate = learning_rate
        self.episode_REINFORCE_objective_gradients = []
        self.weights_historical = []
        super().__init__(*args, **kwargs)
        self.weights_historical.append(self.actor.model.weights[0])
        self.outcomes_of_episodes = []
        self.outcome_episodic_means = []
        self.sim_status = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.weights_history = [self.actor.model.weights]
        if self.is_playback:
            self.episode_tables = []

    def reset_episode(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.outcome = 0
        self.action = self.action_init
        self.system.reset()
        self.actor.reset()
        self.critic.reset()
        self.controller.reset(time_start=0)
        self.simulator.reset()
        self.observation = self.system.out(self.state_init, time=0)
        self.sim_status = 0

    def run(self):

        for _ in range(self.N_iterations):
            for _ in range(self.N_episodes):
                while self.sim_status not in [
                    "episode_ended",
                    "simulation_ended",
                    "iteration_ended",
                ]:
                    self.sim_status = self.step()
                self.reset_episode()
        if self.is_playback:
            if len(self.episode_tables) > 1:
                self.episode_tables = rc.vstack(self.episode_tables)
            else:
                self.episode_tables = rc.array(self.episode_tables[0])

        # return self.episode_tables

        # DEBUG =================================================
        # plt.subplot(121)
        # plt.plot([i + 1 for i in range(self.N_iterations)], self.outcome_episodic_mean)
        # plt.subplot(122)
        # plt.plot([i for i in range(self.N_iterations + 1)], self.weights_history)
        # plt.show()
        # /DEBUG =================================================

    def store_REINFORCE_objective_gradient(self):
        self.outcomes_of_episodes.append(self.critic.outcome)
        episode_REINFORCE_objective_gradient = self.critic.outcome * sum(
            self.actor.gradients
        )
        self.episode_REINFORCE_objective_gradients.append(
            episode_REINFORCE_objective_gradient
        )

    def step(self):
        sim_status = super().step()
        # print(f"Model weights: {self.actor.model.weights}")
        if sim_status == -1:
            if self.is_playback:
                new_table = rc.array(
                    [
                        rc.array(
                            [
                                self.iteration_counter,
                                self.episode_counter,
                                *x,
                                *self.actor.model.weights,
                            ]
                        )
                        for x in self.table
                    ]
                )
                self.episode_tables.append(new_table)
                self.table = []
            self.store_REINFORCE_objective_gradient()
            self.episode_counter += 1

            if self.episode_counter >= self.N_episodes:

                self.outcome_episodic_means.append(rc.mean(self.outcomes_of_episodes))
                self.outcomes_of_episodes = []
                mean_grad_value = sum(self.episode_REINFORCE_objective_gradients) / len(
                    self.episode_REINFORCE_objective_gradients
                )
                self.actor.update_weights_by_gradient(
                    mean_grad_value, self.learning_rate
                )
                self.episode_REINFORCE_objective_gradients = []

                self.episode_counter = 0
                self.iteration_counter += 1
                self.weights_history.append(self.actor.model.weights)
                if self.iteration_counter >= self.N_iterations:
                    return "simulation_ended"
                else:
                    return "iteration_ended"
            else:
                return "episode_ended"
