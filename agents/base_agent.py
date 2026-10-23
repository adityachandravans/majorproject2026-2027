"""
Base Agent Interface for Multi-Agent Disaster Response Coordination.
Provides abstract methods for observation generation, action space creation, and reward computation.
"""

from abc import ABC, abstractmethod
import numpy as np
from gymnasium import spaces
from typing import Optional, Any

class BaseAgent(ABC):
    def __init__(self, agent_id: str, role: str, config: dict):
        self.agent_id = agent_id
        self.role = role
        self.config = config
        
    @abstractmethod
    def get_action_space(self, num_zones: int, num_depots: int) -> spaces.Space:
        """Returns the Gymnasium Discrete action space for this agent role."""
        pass

    @abstractmethod
    def get_observation(self, env_state: dict, comm_channel: Optional[Any] = None) -> np.ndarray:
        """
        Constructs a 1D float vector observation from the environment state and comms.
        
        Args:
            env_state: Complete dictionary representation of the environment.
            comm_channel: Optional CommunicationChannel instance.
        Returns:
            np.ndarray: Normalized 1D float vector observation.
        """
        pass

    @abstractmethod
    def compute_reward(self, env_state: dict, action: int, next_state: dict, step_info: dict = None) -> float:
        """
        Computes role-specific reward based on state transition and action taken.
        
        Args:
            env_state: State before action.
            action: Action index executed.
            next_state: State after action.
            step_info: Optional extra metadata.
        Returns:
            float: Scalar reward for this agent.
        """
        pass
