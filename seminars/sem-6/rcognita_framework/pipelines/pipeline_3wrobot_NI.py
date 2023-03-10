import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import pathlib
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
import rcognita

from config_blueprints import Config3WRobotNI
from pipeline_blueprints import PipelineWithDefaults

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = (
        f"this script is being run using "
        f"rcognita ({rcognita.__version__}) "
        f"located in cloned repository at '{PARENT_DIR}'. "
        f"If you are willing to use your locally installed rcognita, "
        f"run this script ('{os.path.basename(__file__)}') outside "
        f"'rcognita/presets'."
    )
else:
    info = (
        f"this script is being run using "
        f"locally installed rcognita ({rcognita.__version__}). "
        f"Make sure the versions match."
    )
print("INFO:", info)

from rcognita import (
    controllers,
    animators,
    systems,
)
from rcognita.loggers import logger3WRobotNI
from datetime import datetime

from rcognita.utilities import rc


class Pipeline3WRobotNI(PipelineWithDefaults):
    config = Config3WRobotNI

    def initialize_system(self):
        self.system = systems.Sys3WRobotNI(
            sys_type="diff_eqn",
            dim_state=self.dim_state,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            dim_disturb=self.dim_disturb,
            pars=[],
            is_dynamic_controller=self.is_dynamic_controller,
            is_disturb=self.is_disturb,
            pars_disturb=rc.array(
                [
                    [200 * self.sampling_time, 200 * self.sampling_time],
                    [0, 0],
                    [0.3, 0.3],
                ]
            ),
        )

    def initialize_safe_controller(self):
        self.nominal_controller = controllers.NominalController3WRobotNI(
            controller_gain=0.5,
            action_bounds=self.action_bounds,
            time_start=self.time_start,
            sampling_time=self.sampling_time,
        )

    def initialize_logger(self):
        if (
            os.path.basename(os.path.normpath(os.path.abspath(os.getcwd())))
            == "presets"
        ):
            self.data_folder = "../simdata"
        else:
            self.data_folder = "simdata"

        pathlib.Path(self.data_folder).mkdir(parents=True, exist_ok=True)

        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%Hh%Mm%Ss")
        self.datafiles = [None] * self.Nruns

        for k in range(0, self.Nruns):
            self.datafiles[k] = (
                self.data_folder
                + "/"
                + self.system.name
                + "__"
                + self.control_mode
                + "__"
                + date
                + "__"
                + time
                + "__run{run:02d}.csv".format(run=k + 1)
            )

            if self.is_log:
                print("Logging data to:    " + self.datafiles[k])

                with open(self.datafiles[k], "w", newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["System", self.system.name])
                    writer.writerow(["Controller", self.control_mode])
                    writer.writerow(["sampling_time", str(self.sampling_time)])
                    writer.writerow(["state_init", str(self.state_init)])
                    writer.writerow(["is_est_model", str(self.is_est_model)])
                    writer.writerow(["model_est_stage", str(self.model_est_stage)])
                    writer.writerow(
                        [
                            "model_est_period_multiplier",
                            str(self.model_est_period_multiplier),
                        ]
                    )
                    writer.writerow(["model_order", str(self.model_order)])
                    writer.writerow(["prob_noise_pow", str(self.prob_noise_pow)])
                    writer.writerow(
                        ["prediction_horizon", str(self.prediction_horizon)]
                    )
                    writer.writerow(
                        [
                            "pred_step_size_multiplier",
                            str(self.pred_step_size_multiplier),
                        ]
                    )
                    writer.writerow(["data_buffer_size", str(self.data_buffer_size)])
                    writer.writerow(
                        ["running_obj_struct", str(self.running_obj_struct)]
                    )
                    writer.writerow(["R1_diag", str(self.R1_diag)])
                    writer.writerow(["R2_diag", str(self.R2_diag)])
                    writer.writerow(["discount_factor", str(self.discount_factor)])
                    writer.writerow(
                        ["critic_period_multiplier", str(self.critic_period_multiplier)]
                    )
                    writer.writerow(["critic_struct", str(self.critic_struct)])
                    writer.writerow(["actor_struct", str(self.actor_struct)])
                    writer.writerow(
                        [
                            "t [s]",
                            "x [m]",
                            "y [m]",
                            "angle [rad]",
                            "running_objective",
                            "outcome",
                            "v [m/s]",
                            "omega [rad/s]",
                        ]
                    )

        # Do not display annoying warnings when print is on
        if not self.no_print:
            warnings.filterwarnings("ignore")

        self.logger = logger3WRobotNI

    def initialize_visualizer(self):

        self.animator = animators.Animator3WRobotNI(
            objects=(
                self.simulator,
                self.system,
                self.nominal_controller,
                self.controller,
                self.datafiles,
                self.logger,
                self.actor_optimizer,
                self.critic_optimizer,
                self.running_objective,
                self.scenario,
            ),
            pars=(
                self.state_init,
                self.action_init,
                self.time_start,
                self.time_final,
                self.state_init,
                self.xMin,
                self.xMax,
                self.yMin,
                self.yMax,
                self.control_mode,
                self.action_manual,
                self.v_min,
                self.omega_min,
                self.v_max,
                self.omega_max,
                self.Nruns,
                self.no_print,
                self.is_log,
                0,
                [],
            ),
        )


if __name__ == "__main__":

    Pipeline3WRobotNI().execute_pipeline()
