# Language Idioms

Same five rules, different vocabulary, and — more important — different *default postures*. Read only the section for the language you're working in.

## Where the abstraction goes

| Language | Abstraction mechanism | Who declares it | Default posture |
| --- | --- | --- | --- |
| Python | `typing.Protocol` (structural), `abc.ABC` (nominal) | Consumer module | Duck typing first. Add a `Protocol` only at a boundary you test across; `ABC` when you need to *prevent* instantiation or share enforced machinery. Never `abc` for a single implementation. |
| TypeScript | `interface` / type literal / function type | Consumer module | Structural, so a type literal in a parameter position is usually enough. Reach for `interface` when several files share the shape. No runtime check exists — don't fake one. |
| Go | `interface` (implicitly satisfied) | **The consuming package**, always | Smallest possible: 1–3 methods. `io.Reader` is the model. A type with methods but no interface, and one caller, needs nothing. `interface` in the provider package is the classic misread of this rule. |
| Rust | `trait`; `impl Trait` / `dyn Trait` for passing | Defining crate, but sized to the consumer's needs | Generics (`impl Trait`) for monomorphized, zero-cost seams; `Box<dyn Trait>` only when the set must be heterogeneous or growable. `async fn` in a trait needs `dyn` care — consider returning `impl Future` or boxing. |
| Java / Kotlin | `interface`; abstract class when there's shared state | Client package | `interface` over abstract class unless you carry fields. Kotlin: a function type parameter replaces most one-method interfaces (SAM is Java's workaround for the same fact). |

## Posture differences worth knowing

**Python** has no interface enforcement and no compile step, so **L** is the rule most likely to bite: a subtype that raises or tightens a precondition surfaces as a production failure several modules away. The check is behavioral, not textual — "would an existing caller of the base crash, loop, or silently mis-handle this?" Prefer `Protocol` over `ABC` when you only care about shape, and never inherit an ABC just to "declare intent"; that's `isinstance` machinery you now owe forever.

**TypeScript** structural typing means **D** is often already satisfied: pass the function, not the wrapper. The failure mode here is the reverse of Python's — a `class FooService implements IFooService` with a single implementation and a test-only mock, where `FooService({ load })` would have done. Type-only seams cost nothing; class seams cost a file.

**Go** makes **I** nearly automatic because interfaces are declared where they're used and are small by convention. The Go-specific form of speculative abstraction is an `Interface` type in the same package as its single implementation, plus `NewXxx(dep Interface)` — usually replaced by accepting a `func` or a concrete type and letting the *caller* define the interface. Also: `error` wrapping is a substitution question — a function that documents returning `io.EOF` and returns a custom sentinel breaks `errors.Is` callers.

**Rust** expresses **O** with enums + `match` far more often than with dyn-dispatch, and that is correct: an enum is a *closed* set, so a new variant is a compile error at every `match`. Treat that as a feature. Move to `trait` objects when the set is genuinely open across crates, not because a `match` has four arms.

**Java** inherits the most interface boilerplate risk of any language here — the `DefaultFooManagerImpl` naming culture is a monument to rungs 4 and 5 applied at rung 0. If an interface has one implementation, no test double, and no cross-module boundary, delete it. Constructor injection is the default; a DI framework is a rung-5 decision and should be justified by the number of seams, not by convention.

## Naming

Whatever the language, let the name carry the rung. A concrete class is `CsvExporter`; a role/abstraction is `Exporter`. If you find yourself needing `Exporter` for the only implementation, that is the gate in Step 1 telling you to keep the specific name — and the specific name is the one that survives when the second implementation appears.
