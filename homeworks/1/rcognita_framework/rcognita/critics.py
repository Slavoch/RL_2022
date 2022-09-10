"""
Module containing critics, which are integrated in controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
import numpy as np
from .utilities import rc, NUMPY, CASADI, TORCH
from abc import ABC, abstractmethod
import scipy as sp
from functools import partial
import torch
from copy import deepcopy
from multiprocessing import Pool


class Critic(ABC):
    """
    Blueprint of a critic.
    """

    def __init__(
        self,
        dim_input,
        dim_output,
        data_buffer_size,
        optimizer=None,
        model=None,
        running_objective=[],
        discount_factor=1,
        observation_target=[],
    ):
        if optimizer.engine == "Torch":
            data_type = TORCH
        elif optimizer.engine == "CasADi":
            data_type = CASADI
        else:
            data_type = NUMPY

        self.data_buffer_size = data_buffer_size

        self.action_buffer = rc.zeros([data_buffer_size, dim_input], rc_type=data_type)
        self.observation_buffer = rc.zeros(
            [data_buffer_size, dim_output], rc_type=data_type
        )
        self.observation_target = observation_target

        self.discount_factor = discount_factor
        self.running_objective = running_objective
        self.optimizer = optimizer

        # DEBUG------------------------------------------------------------------------------------
        self.g_critic_values = []
        # /DEBUG-----------------------------------------------------------------------------------

        self.model = model

    def __call__(self, *args, use_stored_weights=False):
        if len(args) == 2:
            chi = rc.concatenate(tuple(args))
        else:
            chi = args
        return self.model(chi, use_stored_weights=use_stored_weights)

    def update(self, constraint_functions=(), time=None):

        """
        Update of critic weights.

        """

        if self.optimizer.engine == "CasADi":
            self._CasADi_update(constraint_functions)

        elif self.optimizer.engine == "SciPy":
            self._SciPy_update(constraint_functions)

        elif self.optimizer.engine == "Torch":
            self._Torch_update()

    def update_buffers(self, observation, action):
        self.action_buffer = rc.push_vec(
            self.action_buffer, rc.array(action, prototype=self.action_buffer)
        )
        self.observation_buffer = rc.push_vec(
            self.observation_buffer,
            rc.array(observation, prototype=self.observation_buffer),
        )

    # def grad_observation(self, observation):

    #     observation_symbolic = rc.array_symb(rc.shape(observation), literal="x")
    #     weights_symbolic = rc.array_symb(rc.shape(self.weights), literal="w")

    #     critic_func = self.forward(weights_symbolic, observation_symbolic)

    #     f = Function("f", [observation_symbolic, weights_symbolic], [critic_func])

    #     gradient = rc.autograd(f, observation_symbolic, weights_symbolic)

    #     gradient_evaluated = gradient(observation, weights_symbolic)

    #     # Lie_derivative = rc.dot(v, gradient(observation_symbolic))
    #     return gradient_evaluated, weights_symbolic

    def _SciPy_update(self, constraint_functions=()):

        weights_init = self.model.weights

        constraints = ()
        weight_bounds = [self.model.weight_min, self.model.weight_max]
        data_buffer = {
            "observation_buffer": self.observation_buffer,
            "action_buffer": self.action_buffer,
        }

        cost_function = lambda weights: self.objective(data_buffer, weights=weights)

        if constraint_functions:
            constraints = sp.optimize.NonlinearConstraint(
                partial(
                    self.create_constraints, constraint_functions=constraint_functions,
                ),
                -np.inf,
                0,
            )

        optimized_weights = self.optimizer.optimize(
            cost_function, weights_init, weight_bounds, constraints=constraints,
        )

        self.model.update_and_cache_weights(optimized_weights)

    def _CasADi_update(self, constraint_functions=()):

        weights_init = rc.DM(self.model.cache.weights)
        symbolic_var = rc.array_symb(tup=rc.shape(weights_init), prototype=weights_init)

        constraints = ()
        weight_bounds = [self.model.weight_min, self.model.weight_max]
        data_buffer = {
            "observation_buffer": self.observation_buffer,
            "action_buffer": self.action_buffer,
        }

        cost_function = lambda weights: self.objective(data_buffer, weights=weights)

        cost_function = rc.lambda2symb(cost_function, symbolic_var)

        if constraint_functions:
            constraints = self.create_constraints(constraint_functions, symbolic_var)

        optimized_weights = self.optimizer.optimize(
            cost_function,
            weights_init,
            weight_bounds,
            constraints=constraints,
            decision_variable_symbolic=symbolic_var,
        )

        self.model.update_and_cache_weights(optimized_weights)

    def _Torch_update(self):

        data_buffer = {
            "observation_buffer": torch.tensor(self.observation_buffer),
            "action_buffer": torch.tensor(self.action_buffer),
        }

        self.optimizer.optimize(
            objective=self.objective, model=self.model, model_input=data_buffer,
        )

        self.model.update_and_cache_weights()
        print(
            rc.norm_2(self.model.state_dict()["fc1.weight"]),
            self.objective(data_buffer),
        )

    def update_target(self, new_target):
        self.target = new_target

    @abstractmethod
    def objective(self):
        pass


class CriticValue(Critic):
    def objective(self, data_buffer, weights=None):
        """
        Objective of the critic, say, a squared temporal difference.

        """
        observation_buffer = data_buffer.observation_buffer
        action_buffer = data_buffer.action_buffer

        Jc = 0

        for k in range(self.data_buffer_size - 1, -1, -1):
            observation_old = observation_buffer[k - 1, :]
            observation_next = observation_buffer[k, :]
            action_old = action_buffer[k - 1, :]

            # Temporal difference

            critic_old = self.model(observation_old, weights=weights)
            critic_next = self.model(observation_next, use_stored_weights=True)

            e = (
                critic_old
                - self.discount_factor * critic_next
                - self.running_objective(observation_old, action_old)
            )

            Jc += 1 / 2 * e ** 2

        return Jc


class CriticActionValue(Critic):
    def objective(self, data_buffer, weights=None):
        """
        Objective of the critic, say, a squared temporal difference.

        """

        observation_buffer = data_buffer["observation_buffer"]
        action_buffer = data_buffer["action_buffer"]

        critic_objective = 0

        for k in range(self.data_buffer_size - 1, -1, -1):
            observation_old = observation_buffer[k - 1, :]
            observation_next = observation_buffer[k, :]
            action_old = action_buffer[k - 1, :]
            action_next = action_buffer[k, :]

            # Temporal difference

            critic_old = self.model(observation_old, action_old, weights=weights)
            critic_next = self.model(
                observation_next, action_next, use_stored_weights=True
            )

            temporal_difference = (
                critic_old
                - self.discount_factor * critic_next
                - self.running_objective(observation_old, action_old)
            )

            critic_objective += 1 / 2 * temporal_difference ** 2

        return critic_objective


class CriticSTAG(CriticValue):
    """
    Critic of a stabilizing agent.
    contains special stabilizability constraints.

    """

    def __init__(
        self,
        safe_decay_rate=1e-4,
        safe_controller=[],
        predictor=[],
        *args,
        eps=0.01,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.safe_decay_rate = safe_decay_rate
        self.safe_controller = safe_controller
        self.predictor = predictor
        self.eps = eps

    def get_optimized_weights(self, constraint_functions=(), time=None):

        weights_init = self.model_old.weights

        constraints = ()
        weight_bounds = [self.weight_min, self.weight_max]

        observation = self.observation_buffer[-1, :]

        def stailizing_constraint(weights, observation):

            action_safe = self.safe_controller.compute_action(observation)

            observation_next = self.predictor.predict_state(observation, action_safe)

            critic_curr = self.model(observation, self.model_old.weights)
            critic_next = self.model(observation_next, weights)

            return (
                critic_next
                - critic_curr
                + self.predictor.pred_step_size * self.safe_decay_rate
            )

        if self.optimizer.engine == "CasADi":
            cost_function, symbolic_var = rc.func_to_lambda_with_params(
                self.objective, var_prototype=weights_init
            )

            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, symbolic_var
                )

            lambda_constr = (
                lambda weights: stailizing_constraint(weights, observation) - self.eps
            )

            constraints += (rc.lambda2symb(lambda_constr, symbolic_var),)

            optimized_weights = self.optimizer.optimize(
                cost_function,
                weights_init,
                weight_bounds,
                constraints=constraints,
                decision_variable_symbolic=symbolic_var,
            )

        elif self.optimizer.engine == "SciPy":
            cost_function = rc.func_to_lambda_with_params(self.objective)

            if constraint_functions:
                constraints = sp.optimize.NonlinearConstraint(
                    partial(
                        self.create_constraints,
                        constraint_functions=constraint_functions,
                    ),
                    -np.inf,
                    0,
                )

            resulting_constraints = sp.optimize.NonlinearConstraint(
                lambda weights: stailizing_constraint(weights, observation),
                -np.inf,
                self.eps,
            )

            # DEBUG ==============================
            # resulting_constraints = ()
            # /DEBUG =============================

            optimized_weights = self.optimizer.optimize(
                cost_function,
                weights_init,
                weight_bounds,
                constraints=resulting_constraints,
            )

        return optimized_weights


class CriticMPC(Critic):
    """
    This is just a dummy

    """

    def __init__(self):
        pass

    def objective(self, weights):
        pass

    def get_optimized_weights(self, constraint_functions=(), time=None):
        pass

    def update_buffers(self, observation, action):
        pass

    def update(self, constraint_functions=(), time=None):
        pass


class CriticTabularVI(Critic):
    """
    Critic for tabular agents.

    """

    def __init__(
        self,
        dim_state_space,
        running_objective,
        predictor,
        model,
        actor_model,
        discount_factor=1,
        N_parallel_processes=5,
        terminal_state=None,
    ):

        self.objective_table = rc.zeros(dim_state_space)
        self.action_table = rc.zeros(dim_state_space)
        self.running_objective = running_objective
        self.predictor = predictor
        self.model = model
        self.actor_model = actor_model
        self.discount_factor = discount_factor
        self.N_parallel_processes = N_parallel_processes
        self.terminal_state = terminal_state

    def update_single_cell(self, observation):
        action = self.actor_model.weights[observation]
        if tuple(self.terminal_state) == observation:
            return self.running_objective(observation, action)

        return self.objective(observation, action)

    def update(self):
        observation_table_indices = tuple(
            [
                (i, j)
                for i in range(self.model.weights.shape[0])
                for j in range(self.model.weights.shape[1])
            ]
        )
        new_table = deepcopy(self.model.weights)
        for observation in observation_table_indices:
            new_table[observation] = self.update_single_cell(observation)
        self.model.weights = new_table

    def objective(self, observation, action):
        return (
            self.running_objective(observation, action)
            + self.discount_factor
            * self.model.weights[self.predictor.predict(observation, action)]
        )


class CriticTabularPI(CriticTabularVI):
    def __init__(self, *args, tolerance=1e-3, N_update_iters_max=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.tolerance = tolerance
        self.N_update_iters_max = N_update_iters_max

    def update(self):
        observation_table_indices = tuple(
            [
                (i, j)
                for i in range(self.model.weights.shape[0])
                for j in range(self.model.weights.shape[1])
            ]
        )
        for observation in observation_table_indices:
            difference = rc.abs(
                self.model.weights[observation] - self.update_single_cell(observation)
            )
            for _ in range(self.N_update_iters_max):
                new_value = self.update_single_cell(observation)
                difference = self.model.weights[observation] - new_value
                if difference < self.tolerance:
                    self.model.weights[observation] = self.update_single_cell(
                        observation
                    )
                else:
                    break
