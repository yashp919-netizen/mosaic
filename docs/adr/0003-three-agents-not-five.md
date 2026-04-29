# ADR-0003: Three Agents, Not Five

**Status:** Accepted

---

## Context

The original Mosaic design (see `PROJECT_BRIEF.md` Section 4 and Section 11) specified five agents:

1. **Scout** — statistical profiler + LLM column characterization
2. **Atlas** — semantic column mapper (embeddings + heuristics + LLM reasoning)
3. **Scribe** — per-SKU lineage records + executive summary
4. **Alchemist** — transformation code generator (given a confirmed mapping, generate Python/SQL to convert source values to target format)
5. **Sentinel** — adversarial output validator (check harmonized master for constraint violations, referential integrity, schema conformance)

The sprint is 14 days. The team is one person.

There were two ways to approach this:
- **Option A:** Attempt all five agents, ship none of them well.
- **Option B:** Ship three agents with full eval, benchmarks, and documentation; defer two with honest rationale.

---

## Decision

**Ship v1 with Scout, Atlas, and Scribe. Defer Alchemist and Sentinel to v2.**

---

## Rationale

This decision was made on Day 1 and held through the sprint. The reasoning did not change as the week progressed — in fact the Day 7 calibration check confirmed it was correct.

### Why Scout, Atlas, Scribe are the right three

These three agents cover the highest-value phase of a CPG migration: **you cannot do anything until you know what you have (Scout) and where it goes (Atlas)**. Scribe closes the loop by creating the audit trail a CDO needs before signing off on go-live. Together they compress the analysis-and-mapping phase — typically weeks of spreadsheet work — to minutes, with a quantified confidence score on every decision.

A migration team that had only these three agents would have a working tool. A migration team that had five half-built agents would have nothing.

### Why Alchemist was deferred

Transformation code generation requires more than generating code — it requires verifying that the generated code is correct. A wrong transformation silently corrupts the harmonized master. This means Alchemist needs a self-testing loop: generate → run on sample → validate output schema → fix if wrong. That loop adds significant complexity:

- A sandboxed Python execution environment
- A set of test fixtures per transformation type (date format conversion, unit conversion, code normalization, etc.)
- Error handling for code that generates but doesn't run

None of this is beyond v2 scope. But it requires the mapping layer (Atlas) to be stable first, because Alchemist's inputs are Atlas's outputs. v1 establishes that stability with a benchmarked eval. v2 Alchemist has a solid foundation to build on.

In the meantime, every `LineageRecord.transformations_applied` field is `[]` — an explicit, honest placeholder documented in the code and in `docs/v2-roadmap.md`. The field is in the schema so v2 Alchemist can fill it without a schema migration.

### Why Sentinel was deferred

Sentinel's value proposition — validating the harmonized master before go-live — requires a ground-truth oracle to validate against. That oracle is the target schema plus a set of domain-specific constraints (allergen codes, category hierarchies, unit ranges, referential integrity rules). In v1, the target schema has 12 fields and no constraint rules beyond Pydantic type validation.

Sentinel without meaningful validation rules is a schema linter, not an adversarial validator. Building the full Sentinel on v1's shallow constraint set would produce a false sense of coverage. Better to wait until Atlas is proven (which it now is — F1 0.94–1.00) and build Sentinel with real constraints.

### The broader principle

Scope decisions are architectural decisions. A system that does three things well and is honest about what it doesn't do is more credible than a system that claims five things and delivers none of them reliably. This ADR exists to document that the scope decision was deliberate, not accidental — and to provide the reasoning that a CDO or a technical interviewer would need to evaluate it.

The eval results confirm the decision: with three focused agents, Mosaic achieves F1 of 1.00 (UK), 0.94 (India), 0.94 (Brazil) on a 5,000-SKU benchmark. Those are numbers worth shipping. Five half-built agents would not have produced them.

---

## Alternatives Considered

**Option A — Build all five agents in 14 days:** Rejected. The realistic outcome is that Alchemist and Sentinel would have been implemented at 30–40% of the required quality, with no time for eval or calibration. The benchmark — the core differentiator of this project — would not exist.

**Option B — Build Scout + Atlas only, skip Scribe:** Considered briefly. Rejected because Scribe's lineage records and executive summary are the audit-grade outputs that make the system trustworthy to a CDO. They are also the easiest agent to implement (no embeddings, no LLM tie-breaking), so the marginal cost was low.

**Option C — Implement Alchemist but skip Sentinel:** Considered. Rejected because transformation generation without validation is higher-risk than mapping generation without transformation. A wrong mapping is caught at human review. A wrong transformation silently corrupts the master.

---

## Consequences

**What v1 does not do — explicitly:**
- Does not generate transformation code (e.g., `DD/MM/YY → ISO 8601`, `oz → grams`). Every `LineageRecord.transformations_applied` is `[]`.
- Does not validate the harmonized master beyond Pydantic schema checks.
- Does not resolve cross-market entity duplicates.
- Does not detect allergens from free-text ingredient lists.
- Does not check EU CLP or Indian FSSAI compliance.

**What this buys:**
- A benchmarked, evaluated, documented system that ships on Day 14.
- A clear v2 roadmap (`docs/v2-roadmap.md`) with rationale for every deferral.
- An honest README that passes the "90-second CDO test" without overstating capabilities.
- A foundation that v2 Alchemist and Sentinel can build on without architectural rework.

**Revisit trigger:** v2, after at least one real-world or near-real-world dataset is profiled and mapped using v1. The Alchemist and Sentinel designs should be informed by the actual transformation patterns and constraint violations that appear in real data, not anticipated ones.

---

> **Note to future maintainers:** This ADR is the most important one in this directory. Not because of what it decided, but because of what it demonstrates: that the scope was chosen deliberately, with explicit trade-offs evaluated and documented. When v2 Alchemist ships, this ADR should be updated to record that the deferral was resolved — not deleted.
