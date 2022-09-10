"""
This module contains the logger interface along with concrete realizations for each separate system.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from tabulate import tabulate

import csv
import numpy as np


class Logger:
    """
    Interface class for data loggers.
    Concrete loggers, associated with concrete system-controller setups, are should be built upon this class.
    To design a concrete logger: inherit this class, override:
        | :func:`~loggers.Logger.print_sim_step` :
        | print a row of data of a single simulation step, typically into the console (required).
        | :func:`~loggers.Logger.log_data_row` :
        | same as above, but write to a file (required).

    """

    def __init__(
        self, state_components_strings, action_components_strings, row_format_list=None
    ):
        self.state_components_strings = state_components_strings
        self.action_components_strings = action_components_strings
        self.row_header = [
            "t [s]",
            *self.state_components_strings,
            *self.action_components_strings,
            "running_objective",
            "accum_obj",
        ]
        if row_format_list is None:
            self.row_format = tuple(["8.3f" for _ in self.row_header])
        else:
            self.row_format = row_format_list

    def print_sim_step(self, time, state_full, action, running_objective, accum_obj):
        row_data = [
            time,
            *np.array(state_full),
            *np.array(action),
            running_objective,
            accum_obj,
        ]

        table = tabulate(
            [self.row_header, row_data],
            floatfmt=self.row_format,
            headers="firstrow",
            tablefmt="grid",
        )

        print(table)

    def log_data_row(
        self, datafile, time, state_full, action, running_objective, accum_obj
    ):
        with open(datafile, "a", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                [
                    time,
                    *state_full,
                    *action,
                    running_objective,
                    accum_obj,
                ]
            )


logger3WRobot = Logger(
    ["x [m]", "y [m]", "alpha [rad]", "v [m/s]", "omega [rad/s]"], ["F [N]", "M [N m]"]
)

logger3WRobotNI = Logger(
    ["x [m]", "y [m]", "alpha [rad]"],
    ["v [m/s]", "omega [rad/s]"],
    ["8.3f", "8.3f", "8.3f", "8.3f", "8.1f", "8.1f", "8.3f", "8.3f"],
)

logger2Tank = Logger(
    ["h1", "h2"], ["p"], ["8.1f", "8.4f", "8.4f", "8.4f", "8.4f", "8.2f"]
)
