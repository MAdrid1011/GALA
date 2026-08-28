"""Discrete-event timing model and hardware module contracts."""

from .config import (
    AsyncMemoryBackend, ComputePathProfile, ComputeStage, ComputeTemplateProfile, CycleConfig,
    ModuleTiming, MemoryBackend,
)
from .engine import (
    CycleEngine, CycleProgress, CycleResult, CycleConfigurationError,
    CycleReplaySession, BufferedVirtualCycleConsumer, OraclePortfolio,
    OraclePortfolioMember,
)
from .resources import ResourceEnvelope, ResourceUsage
from .packets import (
    PhysicalPacketStage, RelationPacketPlan, RelationPacketPlanError,
)
from .memory import RecordedMemoryBackend
from .bounds import (
    CycleBoundComponent, CycleLowerBoundReport, ScenarioCycleBound,
    TargetReachability, analyze_cycle_lower_bounds,
)

__all__ = ["AsyncMemoryBackend", "ComputePathProfile", "ComputeStage", "ComputeTemplateProfile", "CycleConfig", "ModuleTiming", "MemoryBackend", "CycleEngine", "CycleProgress", "CycleResult",
           "CycleConfigurationError", "CycleReplaySession", "BufferedVirtualCycleConsumer", "OraclePortfolio", "OraclePortfolioMember", "ResourceEnvelope", "ResourceUsage", "RecordedMemoryBackend", "CycleBoundComponent", "CycleLowerBoundReport", "ScenarioCycleBound", "TargetReachability", "analyze_cycle_lower_bounds"]
__all__ += ["PhysicalPacketStage", "RelationPacketPlan", "RelationPacketPlanError"]
