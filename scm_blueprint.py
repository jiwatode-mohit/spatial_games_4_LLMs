
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class SCMBlueprint:
    nodes: Dict[str, str]
    edges: List[Sequence[str]]
    scm_notes: List[str] = field(default_factory=list)
    allow_extra: bool = True

    def to_dict(self) -> Dict:
        return {
            "nodes": self.nodes,
            "edges": [list(edge) for edge in self.edges],
            "scm_notes": self.scm_notes,
        }

    def to_text(self) -> str:
        parts = ["Nodes:"] + [f"- {k}: {v}" for k, v in self.nodes.items()]
        parts += ["\nEdges:"] + [f"- {s}->{t}" for s, t in self.edges]
        if self.scm_notes: parts += ["\nSCM Authoring Rules:"] + [f"- {n}" for n in self.scm_notes]
        if self.allow_extra: parts.append("\nExtra context-dependent nodes allowed.")
        return "\n".join(parts)



bp = SCMBlueprint(
    nodes={
        # Design Layer
        "EntityTypes": "Static sprite types, roles (avatar, wall), & attrs (speed, hp). Not time-varying.",
        "ActionSpace": "Agent interventions/actions available at each step (move, shoot).",
        "GlobalMechanics": "Global state transitions independent of collisions (gravity, timers).",
        "InteractionMechanics": "Collision rules mapping (Entities, State_t, Action_t) -> State_{t+1} events (spawn, destroy).",
        "RewardMechanics": "Maps State_t/Interactions to Reward_t or Score_{t+1}.",
        "TerminationMechanics": "Win/loss conditions based on StateVariables (e.g., timeout, goals_met).",
        # Dynamics Layer
        "StateVariables": "Dynamic vars evolving over t (pos, hp). Eq: X_{t+1}=f(PA_X_t, Action_t, U_X).",
        "InitialState": "StateVariables at t=0 derived from EntityTypes + LevelEncoding.",
        # Observation Layer
        "LevelEncoding": "ASCII grid mapping chars to entity instances/positions; induces InitialState.",
    },
    edges=[
        ("EntityTypes", "InitialState"), ("LevelEncoding", "InitialState"), ("EntityTypes", "StateVariables"),
        ("ActionSpace", "InteractionMechanics"), ("ActionSpace", "StateVariables"),
        ("GlobalMechanics", "StateVariables"), ("InteractionMechanics", "StateVariables"),
        ("StateVariables", "InteractionMechanics"), ("StateVariables", "RewardMechanics"),
        ("InteractionMechanics", "RewardMechanics"), ("RewardMechanics", "StateVariables"),
        ("StateVariables", "TerminationMechanics"), ("InteractionMechanics", "TerminationMechanics"),
    ],
    scm_notes=[
        "Define DYNAMIC SCM with discrete time steps t=0,1...",
        "Separate STATIC design vars from DYNAMIC state vars.",
        "For dynamic X, define X_{t+1} = f(Parents_t, Action_t, Noise).",
        "Define Reward_t & Termination explicitly as equations.",
        "Output machine-readable format (JSON) with: static_nodes, dynamic_variables, edges, equations.",
        "Link LevelEncoding chars to InitialState.",
    ],)