# ADR-0002: LangGraph vs CrewAI for Agent Orchestration

**Status:** Accepted

---

## Context

Mosaic requires an agent orchestration framework to coordinate three agents (Scout, Atlas, Scribe) with:
1. **Shared typed state** — every agent reads from and writes to a common data structure.
2. **Human-in-the-loop interrupt** — the pipeline must suspend after Atlas, present low-confidence mappings to a human, and resume without re-running prior agents.
3. **Auditability** — every state transition must be traceable and explainable in an interview or to a CDO.
4. **Graph topology** — the flow has a conditional branch (skip human review if all mappings are high-confidence), which should be explicit in the code rather than implicit in an agent's prompt.

Options evaluated:

| Option | Model | Human-in-the-loop | State control | Interview defensibility |
|---|---|---|---|---|
| **LangGraph** | Explicit StateGraph | Native `interrupt()` + `MemorySaver` | Full — TypedDict, typed nodes | High — literally drawable as a whiteboard |
| **CrewAI** | Role-based agents, implicit graph | Workaround via callbacks | Limited — state is managed internally | Medium — high-level, hard to dissect |
| **AutoGen** | Conversational multi-agent | Via termination conditions | Limited — message-passing model | Medium — well-known but different paradigm |
| **Hand-rolled state machine** | Custom Python | Custom | Full | High — but more code, more maintenance |

---

## Decision

Use **LangGraph**.

---

## Rationale

LangGraph's `StateGraph` is the only framework in this list that makes the human-in-the-loop checkpoint a first-class primitive. The `interrupt()` call in `human_review_node` suspends the graph, serializes all state to the `MemorySaver` checkpointer, and returns control to the caller (CLI or Streamlit UI). The caller collects human decisions, then resumes with `graph.invoke(Command(resume=decisions), config={"thread_id": ...})`. No agent reruns. No state is lost.

The explicit `StateGraph` topology maps directly to the architecture diagram in Section 1 of `docs/architecture.md`. Every edge is a line of code. Every conditional branch is a named function. This is not incidental — it means the architecture can be explained precisely in a 30-minute interview without hand-waving about "how the framework handles it."

The `TypedDict`-based `MosaicState` enforces a contract between nodes. An agent that writes a wrong type to state fails loudly at the type-checker, not silently at runtime. This matters when agents are added or modified.

CrewAI's abstraction makes it easier to write a simple multi-agent chat. It makes it harder to implement a deterministic pipeline with a typed state machine and a resumable interrupt. Those two requirements ruled it out.

---

## Alternatives Considered

**CrewAI:** Better choice for conversational multi-agent systems where agents negotiate over a shared task. Its "Crew" abstraction does not expose enough control over the interrupt/resume lifecycle. The role-based model also makes it harder to reason about what data each agent consumes and produces.

**AutoGen:** Microsoft's framework is optimized for conversational, back-and-forth agent collaboration (e.g., a coder agent and a critic agent). Mosaic's pipeline is a directed acyclic flow with one conditional branch and one interrupt — a poor match for AutoGen's message-passing paradigm.

**Hand-rolled state machine:** Would give full control but adds significant boilerplate and maintenance surface. The LangGraph primitives (`StateGraph`, `interrupt()`, `MemorySaver`) cover exactly the features Mosaic needs without the overhead of building them from scratch.

---

## Consequences

- LangGraph requires more boilerplate than CrewAI for a simple pipeline. The `StateGraph` builder, node functions, edge declarations, and graph compilation add ~100 lines that CrewAI would hide. This is an accepted trade-off for auditability.
- `MemorySaver` persists state in-process only. If the process restarts between the interrupt and the resume (e.g., a Streamlit server restart), the thread is lost. v2 should use `SqliteSaver` or a persistent checkpointer for production deployments.
- The graph is literally drawable as a whiteboard diagram (see `docs/architecture.md` Section 1 mermaid). This was a deliberate design goal.
- Adding a new agent (e.g., v2 Alchemist between `finalize_node` and `scribe_node`) requires adding one node function and one `builder.add_edge()` call. The typed state contract enforces what the new agent must read and write.

**Revisit trigger:** v2, if the orchestration needs to support parallel agent execution (e.g., profiling multiple markets concurrently). LangGraph supports this via `Send` — no framework change needed.
