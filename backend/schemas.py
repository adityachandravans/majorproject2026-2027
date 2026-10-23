"""
Pydantic Schemas for FastAPI Backend API (Section 6, FR5, FR6, FR8).
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class CreateSimRequest(BaseModel):
    config: Optional[Dict[str, Any]] = None
    policy: str = Field(default="marl", description="Policy to use: marl, baseline, or random")
    checkpoint: Optional[str] = None
    seed: int = 42
    speed: float = 1.0

class SwitchPolicyRequest(BaseModel):
    policy: str = Field(..., description="Policy to switch to: marl, baseline, or random")
    checkpoint: Optional[str] = None

class CompareRequest(BaseModel):
    config: Optional[Dict[str, Any]] = None
    seed: int = 42
    max_steps: Optional[int] = None

class SimStateResponse(BaseModel):
    sim_id: str
    state: Dict[str, Any]

class CheckpointInfo(BaseModel):
    filename: str
    path: str
    size_bytes: int
    modified_time: str
