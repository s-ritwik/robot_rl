import math
import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

from robot_rl.tasks.manager_based.robot_rl import mdp
from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

##
# Terrain configuration
##

SLIGHTLY_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.0025,  # Reduced from 0.005 in rough terrain
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.01, 0.06), noise_step=0.01  # Reduced from (0.02, 0.10)
        ),
        "random_flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.0, 0.0), noise_step=0.01  # Half flat terrain
        )
    },
)
"""Configuration for slightly rough terrains with reduced difficulty."""


##
# Environment configuration
##
@configclass
class G1SlightRoughLipEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 Slightly Rough environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        # change terrain from flat to slightly rough generated terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = SLIGHTLY_ROUGH_TERRAINS_CFG
        
        # The parent G1RoughLipEnvCfg and the original G1FlatLipEnvCfg disable the height scanner.
        # For a rough terrain, a height scanner is typically needed.
        # If you need to enable it, you should ensure the following lines are NOT present
        # in the parent class or uncomment them here.
        # from isaaclab.managers import SceneEntityCfg
        # self.scene.height_scanner = SceneEntityCfg("height_scanner", ...)
        # self.observations.policy.height_scan = ObsTerm(func=mdp.height_scan)
        
        # no terrain curriculum for now
        self.curriculum.terrain_levels = None


class G1SlightRoughRefTrackingEnvCfg(G1SlightRoughLipEnvCfg):
    """Configuration for the G1 Slightly Rough environment with reference tracking."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # self.rewards.clf_reward = None
        self.rewards.clf_decreasing_condition = None
        self.curriculum.clf_curriculum = None


class G1SlightRoughLipVdotEnvCfg(G1SlightRoughLipEnvCfg):
    """Configuration for the G1 Slightly Rough environment with V-dot reward."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.clf_decreasing_condition = None
        self.curriculum.clf_curriculum = None
        self.rewards.clf_vdot_tanh = RewTerm(
            func=mdp.clf_decreasing_condition,
            weight=2.0,
            params={
                "command_name": "hlip_ref",
            },
        )


class G1SlightRoughLipEnvCfg_PLAY(G1SlightRoughLipEnvCfg):
    """Play-time configuration for the G1 Slightly Rough environment."""
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
        self.events.push_robot.interval_range_s = (5.0, 5.0)  # push robot after 5 seconds
        self.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0, 0)}

        # reduce the number of terrain sections for play mode
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2
