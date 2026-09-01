# GALA Simulator Documentation

These documents define the simulator's architecture, input contracts, event
semantics, reproducibility rules, and extension interfaces. They describe the
software and its output formats; generated runs and experiment histories are
kept outside version control.

1. [Project Constraints and Mainline](00_CONSTRAINTS_AND_MAINLINE.md)
2. [Reproduction Targets and Evidence](01_REPRODUCTION_TARGETS.md)
3. [Simulator Software Architecture](02_SIMULATOR_ARCHITECTURE.md)
4. [GALA Hardware Contract](03_HARDWARE_CONTRACT.md)
5. [Parameter Registry](04_PARAMETER_REGISTRY.md)
6. [Model Adapters and Datasets](05_MODELS_AND_DATASETS.md)
7. [Cycle Model and Event Semantics](06_CYCLE_MODEL.md)
8. [Ablation and Output Contract](07_ABLATION_AND_OUTPUTS.md)
9. [Performance Engineering](08_PERFORMANCE_ENGINEERING.md)
10. [Quality Validation](09_VALIDATION_AND_ACCEPTANCE.md)
11. [Implementation Workflow](10_IMPLEMENTATION_WORKFLOW.md)
12. [Workspace and Assets](WORKSPACE_AND_ASSETS.md)

When documents conflict, apply this precedence: project constraints, hardware
contract, parameter registry, cycle model, model and dataset contract, then
workflow guidance. Implementation convenience does not override a higher-level
contract.
