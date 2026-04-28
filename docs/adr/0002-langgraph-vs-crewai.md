# ADR-0002: LangGraph vs CrewAI for Agent Orchestration

**Status:** Accepted

## Context

Mosaic needs an agent orchestration framework to coordinate SCOUT → ATLAS → SCRIBE with a human-in-the-loop checkpoint.

## Decision

Use LangGraph.

## Reasons

- Explicit state machine with typed state — every transition is auditable and drawable in a whiteboard interview
- Native human-in-the-loop support via interrupt_before / interrupt_after
- Persistent checkpointing built in — resume after human review without re-running prior agents
- CrewAI abstracts away the graph, making it harder to explain architectural decisions precisely

## Consequences

LangGraph requires more boilerplate than CrewAI. Acceptable trade-off given the interview-defensibility requirement.
