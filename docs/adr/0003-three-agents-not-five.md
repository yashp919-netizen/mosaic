# ADR-0003: Three Agents, Not Five

**Status:** Accepted

## Context

The full Mosaic vision includes five agents: SCOUT, ATLAS, SCRIBE, ALCHEMIST (transformation generator), SENTINEL (output validator).

## Decision

Ship v1 with three agents (SCOUT, ATLAS, SCRIBE). Defer ALCHEMIST and SENTINEL to v2.

## Reasons

- Three agents cover the highest-value phase: profiling → mapping → audit
- ALCHEMIST requires code execution sandboxing — meaningful extra scope
- SENTINEL requires a ground-truth oracle — only meaningful after the eval harness exists
- A complete, benchmarked three-agent system is more defensible than five half-built agents

## Consequences

v1 does not generate transformation code or validate outputs against target schema constraints. Documented as known limitation in README.
