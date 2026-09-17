---
description: Find the seam, then pay for exactly one. Diagnose structural problems (god class, growing if/switch chain, broken subclass, fat interface, hardwired dependency) and make the smallest refactor that fixes them, with a hard gate against speculative abstraction. Use when the user asks to refactor, clean up, redesign, or review the structure of code; says a file or class "is a mess", "smells", "is too big"; asks for SOLID, clean architecture, or "make it extensible"; wants a new variant/format/provider/type added; or when writing new code in a place that already has more than one implementation of something. Do not use for bug fixes, renames, formatting, performance work, or one-off scripts.
metadata:
    github-path: skills/solid
    github-ref: refs/heads/main
    github-repo: https://github.com/dheerapat/skills
    github-tree-sha: a15896b4d6661f77c6e02a228ec2d34289762f52
name: solid
---
# SOLID Refactoring

The rules are not the deliverable. **The seam is the deliverable** — the one place a change will arrive, opened exactly once.

Read this skill knowing what it is guarding against. A model told "code must follow SOLID" reliably does the opposite of good design: it writes a `Protocol`, an abstract base class, a factory, and an interface for code that has one implementation and will never have a second. That is not SOLID. It is unused indirection, and it costs every future reader an extra hop through a layer that answers nothing. So the order below starts with a gate, not with the letters.

## Step 1 — Gate: does this need to change?

Look for the evidence in the code that is actually in front of you. "Someone might need it later" is not evidence.

| Evidence present in the code | Principle to apply |
| --- | --- |
| Class/module whose members are used by **disjoint sets of callers**, and it has been edited for unrelated reasons (persistence change + formatting change in the same file) | **S** |
| An `if`/`match`/`switch` chain where **git history shows a branch added per new variant** | **O** |
| A subtype that raises `NotImplementedError`, adds validation, returns a narrower/looser type, or would break a caller of the base | **L** |
| An implementer that `pass`es, throws, or stubs a method it doesn't do; a client that touches 2 of 8 methods | **I** |
| A concrete I/O, network, clock, storage, or vendor client constructed **inside** business logic, so the logic can't be tested without the real thing | **D** |

Nothing matched → **write or keep the concrete version.** Do not add an abstraction. If the user asked for SOLID and the code already qualifies, say which evidence is missing in one line and stop. Refactoring clean code into abstract code is a regression.

Explicitly closed sets are fine as they are: an `Enum`, a discriminated union with an exhaustive `match`, a sealed class. Closed for extension is a *design choice*, not a violation of **O**. Only open the set when variants actually keep arriving.

## Step 2 — Pick the smallest seam that holds

Ascend this ladder one rung at a time and stop at the first rung that satisfies the gate. Skipping rungs is the most common way this goes wrong.

0. **Nothing.** Inline branch over a closed set.
1. **A parameter.** A value, a callable, or a narrow type passed in. A seam needs no class — `def charge(cart, *, price_of)` is injectable and testable with zero boilerplate.
2. **An extracted function or module.** Fixes most **S** problems: two jobs become two files, no new types.
3. **A concrete class for the varying part** (strategy as a plain function or small class). Still no interface.
4. **An abstraction** — `Protocol` / `interface` / trait / ABC. Only now, and only when **at least two real implementations exist**, or one exists and the second is a test double you will actually write. One method per client need, not one per implementer capability.
5. **A registry or factory.** Only when variants get added by people who cannot edit the dispatch site (plugins, per-tenant handlers, driver systems). Otherwise the dispatch site editing itself is correct and simpler.

Rungs 4 and 5 are where the abstraction cost is paid; rungs 1–3 are where the flexibility usually already lives.

## Step 3 — The letters, operationally

Full before/after refactors with realistic Python and TypeScript: `references/worked-refactors.md`. Per-language idiom (what an abstraction is *called* and *where it lives* in Python / TS / Go / Rust / Java): `references/language-idioms.md`. Read the one you need; don't read both by default.

- **S — one reason to change.** Count the *kinds of reason* the file has been edited. More than one, and it is more than one module. Name each resulting class after its single job; if you can't name it without the word "and", the split is wrong.
- **O — extension replaces modification.** The move is from "edit the chain" to "add a thing the chain looks up". Prefer data (a dict/registry of handlers) over control flow when variants are open-ended; prefer control flow when they aren't.
- **L — substitutable without callers noticing.** Four ways it breaks: throws on a base method; strengthens a precondition; weakens a postcondition; changes observable semantics the caller relies on. Fix by **narrowing the base type** (a `FlyingBird` above `Bird`, validation in the base constructor so subtypes inherit it) or by **composition over inheritance** — usually the right answer when you find yourself reaching for a fix.
- **I — interfaces split by client.** Derive the shape from what one call site actually touches, not from what the implementer can do. If every implementer of your interface implements every method, you didn't need to split it.
- **D — the consumer owns the abstraction.** Declare the narrow protocol next to the code that *uses* it, not next to the code that implements it. Concrete things (DB, HTTP, clock, filesystem) get constructed at the edge — `main`, adapters, wiring — and passed inward. Pure logic in the middle, effects at the boundary. That shape satisfies S and D at once and is the practical payload of clean architecture.

## Step 4 — Refactor without breaking behavior

1. Get tests around the seam *before* moving code. If none exist and the logic is non-trivial, add a characterization test that pins current output for a few inputs.
2. Land it as two steps: **extract** (behavior-preserving, no signature changes callers care about), then **use** the new seam. Two small diffs beat one that mixes both.
3. Update every call site — grep for the old name, don't trust an IDE-less memory.
4. Don't fix what wasn't asked. If you spot a violation outside the scope of the request, report it in one line and leave the code alone; the user can ask for it.
5. Match the language and style of the repo. Examples here are Python/TS; a Rust or Go task gets Rust and Go idioms, not Python classes with `self`.

## Step 5 — Report

Lead with the diagnosis, then the change. Under 10 lines:

```text
S: OrderService was edited for pricing and for email delivery — two jobs.
Seam: extracted PriceCalculator (rung 2), left notification concrete — single implementation, no protocol needed.
Not done: CheckoutFacade still hardwires StripeClient. Worth inverting if a second provider lands.
```

The "Not done" line matters: it records the seam you deliberately declined, which is the decision a reviewer needs to check.

## Quick check before committing

- Did I add an abstraction with only one implementation? → delete it, go down a rung.
- Can I name the second caller of the interface I just wrote? → if no, delete it.
- Did behavior change anywhere a test wouldn't notice? → add the test first.
- Does the diff touch files the request didn't mention? → back them out.
- Did the *existing* dispatch site get edited, or did the new variant get added alongside it? → that is the whole test for O.
