#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains one single class that simulates controller-system (agent-environment) loops.
The system can be of three types:
    
- discrete-time deterministic
- continuous-time deterministic or stochastic
- discrete-time stochastic (to model Markov decision processes)

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
import scipy as sp

from .utilities import rej_sampling_rvs


class Simulator:
    """
    Class for simulating closed loops (system-controllers).

    Attributes
    ----------
    sys_type : : string
        Type of system by description:

        | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, u, q)`
        | ``discr_fnc`` : difference equation :math:`state^+ = f(state, u, q)`
        | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, u, q)`

    where:

        | :math:`state` : state
        | :math:`u` : input
        | :math:`q` : disturbance

    compute_closed_loop_rhs : : function
        Right-hand side description of the closed-loop system.
        Say, if you instantiated a concrete system (i.e., as an instance of a subclass of ``system`` class with concrete ``compute_closed_loop_rhs`` method) as ``my_system``,
        this could be just ``my_system.compute_closed_loop_rhs``.

    sys_out : : function
        System output function.
        Same as above, this could be, say, ``my_system.out``.

    is_dynamic_controller : : 0 or 1
        If 1, the controller (a.k.a. agent) is considered as a part of the full state vector.

    state_init, disturb_init, action_init : : vectors
        Initial values of the (open-loop) system state, disturbance and input.

    time_start, time_final, sampling_time : : numbers
        Initial, final times and time step size

    max_step, first_step, atol, rtol : : numbers
        Parameters for an ODE solver (used if ``sys_type`` is ``diff_eqn``).

    See also
    --------

    ``systems`` module

    """

    def __init__(
        self,
        sys_type,
        compute_closed_loop_rhs,
        sys_out,
        state_init,
        disturb_init=[],
        action_init=[],
        time_start=0,
        time_final=1,
        sampling_time=1e-2,
        max_step=0.5e-2,
        first_step=1e-6,
        atol=1e-5,
        rtol=1e-3,
        is_disturb=0,
        is_dynamic_controller=0,
    ):

        """
        Parameters
        ----------
        sys_type : : string
            Type of system by description:

            | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, u, q)`
            | ``discr_fnc`` : difference equation :math:`state^+ = f(state, u, q)`
            | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, u, q)`

        where:

            | :math:`state` : state
            | :math:`u` : input
            | :math:`q` : disturbance

        compute_closed_loop_rhs : : function
            Right-hand side description of the closed-loop system.
            Say, if you instantiated a concrete system (i.e., as an instance of a subclass of ``System`` class with concrete ``compute_closed_loop_rhs`` method) as ``my_system``,
            this could be just ``my_system.compute_closed_loop_rhs``.

        sys_out : : function
            System output function.
            Same as above, this could be, say, ``my_system.out``.

        is_dynamic_controller : : 0 or 1
            If 1, the controller (a.k.a. agent) is considered as a part of the full state vector.

        state_init, disturb_init, action_init : : vectors
            Initial values of the (open-loop) system state, disturbance and input.

        time_start, time_final, sampling_time : : numbers
            Initial, final times and time step size

        max_step, first_step, atol, rtol : : numbers
            Parameters for an ODE solver (used if ``sys_type`` is ``diff_eqn``).
        """

        self.sys_type = sys_type
        self.compute_closed_loop_rhs = compute_closed_loop_rhs
        self.sys_out = sys_out
        self.sampling_time = sampling_time

        # Build full state of the closed-loop
        if is_dynamic_controller:
            if is_disturb:
                state_full_init = np.concatenate(
                    [state_init, disturb_init, action_init]
                )
            else:
                state_full_init = np.concatenate([state_init, action_init])
        else:
            if is_disturb:
                state_full_init = np.concatenate([state_init, disturb_init])
            else:
                state_full_init = state_init

        self.state_full = state_full_init

        self.time = time_start
        self.state = state_init
        self.dim_state = state_init.shape[0]
        self.observation = self.sys_out(state_init)

        if sys_type == "diff_eqn":
            self.ODE_solver = sp.integrate.RK45(
                compute_closed_loop_rhs,
                time_start,
                state_full_init,
                time_final,
                max_step=max_step,
                first_step=first_step,
                atol=atol,
                rtol=rtol,
            )

        # Store these for reset purposes
        self.state_full_init = state_full_init
        self.time_start = time_start

    def sim_step(self):
        """
        Do one simulation step and update current simulation data (time, system state and output).

        """
        if self.sys_type == "diff_eqn":
            self.ODE_solver.step()

            self.time = self.ODE_solver.t
            self.state_full = self.ODE_solver.y

            self.state = self.state_full[0 : self.dim_state]
            self.observation = self.sys_out(self.state)

        elif self.sys_type == "discr_fnc":
            self.time = self.time + self.sampling_time
            self.state_full = self.compute_closed_loop_rhs(self.time, self.state_full)

            self.state = self.state_full[0 : self.dim_state]
            self.observation = self.sys_out(self.state)

        elif self.sys_type == "discr_prob":
            self.state_full = rej_sampling_rvs(
                self.dim_state, self.compute_closed_loop_rhs, 10
            )

            self.time = self.time + self.sampling_time

            self.state = self.state_full[0 : self.dim_state]
            self.observation = self.sys_out(self.state)
        else:
            raise ValueError("Invalid system description")

    def get_sim_step_data(self):
        """
        Collect current simulation data: time, system state and output, and, for completeness, full closed-loop state.

        """

        time, state, observation, state_full = (
            self.time,
            self.state,
            self.observation,
            self.state_full,
        )

        return time, state, observation, state_full

    def reset(self):
        if self.sys_type == "diff_eqn":
            self.ODE_solver.status = "running"
            self.ODE_solver.time = self.time_start
            self.ODE_solver.observation = self.state_full_init
        else:
            self.time = self.time_start
            self.state_full = self.state_full_init
