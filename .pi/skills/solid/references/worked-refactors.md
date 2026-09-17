# Worked Refactors

One realistic before/after per letter, plus the over-applied version to avoid. Python and TypeScript alternate because the shape of the mistake is the same in both.

## Table of contents

- [S — a class edited for two reasons](#s)
- [O — a chain that grows a branch per variant](#o)
- [L — the subclass that can't do it](#l)
- [I — the interface nobody fully implements](#i)
- [D — business logic that news up a client](#d)
- [The one to decline — abstraction with one implementation](#decline)

---

<a name="s"></a>
## S — a class edited for two reasons

Before. One class, edited whenever pricing changes *or* delivery changes *or* the schema changes. Note that it is already "well factored" by name — `OrderService` sounds like a service, so nothing looks wrong until you count the reasons.

```python
class OrderService:
    def __init__(self):
        self.db = PostgresDB(DSN)

    def total(self, order):
        sub = sum(l.qty * price(l.sku) for l in order.lines)
        if order.country == "TH":
            sub *= 1.07                      # VAT, edited when tax law changes
        return round(sub, 2)

    def confirm(self, order):
        self.db.execute("INSERT INTO orders ...", order.id, self.total(order))
        smtp.send(to=order.email, subject="Confirmed", body=render(order))
        order.status = "CONFIRMED"           # state machine, edited for unrelated reasons
```

After. Split on the reasons, not on the method count. `total` moves because tax law is a different axis of change than delivery and a different axis than SQL. The two things that *were* legitimately coupled (write + notify, both "confirming an order") stay together.

```python
class ThaiVat:                              # one reason to change: tax rules
    RATE = 1.07

    def apply(self, subtotal: Decimal) -> Decimal:
        return (subtotal * self.RATE).quantize(Decimal("0.01"))


class OrderConfirmer:                       # one reason to change: what confirming means
    def __init__(self, orders, notifier, pricing):
        self.orders, self.notifier, self.pricing = orders, notifier, pricing

    def confirm(self, order):
        order.total = self.pricing.total(order)
        self.orders.save(order)
        self.notifier.sent(order)
```

Pricing is a plain class, not a protocol: one implementation exists. When `ThaiVat` gains siblings, the seam is already a constructor parameter.

---

<a name="o"></a>
## O — a chain that grows a branch per variant

Before. Every export format added costs an edit here, and the edit is in the middle of code that already worked.

```ts
function exportReport(rows: Row[], format: string): string {
  if (format === "csv") return rows.map(r => Object.values(r).join(",")).join("\n")
  if (format === "tsv") return rows.map(r => Object.values(r).join("\t")).join("\n")
  if (format === "json") return JSON.stringify(rows)
  if (format === "ndjson") return rows.map(r => JSON.stringify(r)).join("\n")
  throw new Error(`unknown format ${format}`)
}
```

After. The variants become data; adding one means adding a table entry and nothing else.

```ts
type Format = { ext: string; render: (rows: Row[]) => string }

const FORMATS: Record<string, Format> = {
  csv: { ext: "csv", render: rows => delimit(rows, ",") },
  tsv: { ext: "tsv", render: rows => delimit(rows, "\t") },
  json: { ext: "json", render: rows => JSON.stringify(rows, null, 2) },
  ndjson: { ext: "ndjson", render: rows => rows.map(r => JSON.stringify(r)).join("\n") },
}

export function exportReport(rows: Row[], format: keyof typeof FORMATS): string {
  return FORMATS[format].render(rows)
}
```

`csv`/`tsv` collapsed into one parameterized renderer — the open set got *smaller*, which is the tell that this refactor earned its keep. Kept the registry as a plain object rather than a class hierarchy: no state, no lifecycle, so no interface. If formats ever ship as plugins from packages that can't import this file, rung 5 (a registry with a `register()` call) is the next step; until then, editing `FORMATS` is fine.

---

<a name="l"></a>
## L — the subclass that can't do it

Before. The `NotImplementedError` is the loud version; the quiet version is `VerifiedBot` strengthening a precondition and `ReadOnlyBot` weakening what callers can assume.

```python
class Bot:
    def send(self, text: str) -> None: ...
    def retract(self, msg_id: str) -> None: ...

class WebhookBot(Bot):
    def send(self, text): ...
    def retract(self, msg_id):
        raise NotImplementedError("webhooks cannot retract")   # caller's try/except now lies

class VerifiedBot(Bot):
    def send(self, text):
        if len(text) > 200:                # stricter than the base allows
            raise ValueError(text)
```

After. The base is narrowed to what both variants genuinely guarantee; the capability that isn't universal becomes a second, smaller type. Callers that need `retract` ask for it in their signature and get it checked.

```python
class Sender(Protocol):
    def send(self, text: str) -> None: ...

class Retractability(Protocol):
    def retract(self, msg_id: str) -> None: ...

class WebhookBot:
    def __init__(self, url: str, limit: int = 500): self.url, self.limit = url, limit
    def send(self, text: str) -> None: ...       # honors the base's promise

class PushBot(WebhookBot, Retractability):
    def retract(self, msg_id: str) -> None: ...
```

`VerifiedBot`'s length check is not a substitution violation if the limit is declared in the base contract (e.g. `Bot.send` documents `max_len`, exposed as an attribute callers read). A subtype is allowed to be *as strict*, never *stricter than advertised*, and never *looser in what it returns*.

When inheritance is only there to reuse code, drop it: hold a `WebhookBot` and add `retract` (composition) instead of subclassing.

---

<a name="i"></a>
## I — the interface nobody fully implements

Before. Designed from the implementer's capabilities. Every new device pays for the methods it can't do.

```ts
interface Device {
  print(doc: Doc): void
  scan(): Doc
  fax(to: string, doc: Doc): void
  staple(count: number): void
}

class CheapPrinter implements Device {
  print(doc: Doc) { ... }
  scan(): Doc { throw new Error("no scanner") }
  fax() { throw new Error("no fax") }
  staple() { throw new Error("no stapler") }
}
```

After. Split by what each *client* calls. The spillover benefit: `PrintJob` can now be tested with an object literal instead of a mock.

```ts
interface Printer { print(doc: Doc): void }
interface Scanner { scan(): Doc }

class PrintJob { constructor(private out: Printer) {} }
class Inbox    { constructor(private in_: Scanner) {} }
```

`Device` is gone, not widened. Two clients, two interfaces. If a device implements both, it implements both — no `throw` anywhere, and **I** has nothing left to do.

---

<a name="d"></a>
## D — business logic that news up a client

Before. The logic is untestable without a live S3 bucket, and swapping buckets means finding every construction site.

```python
class ReportArchiver:
    def __init__(self):
        self.s3 = boto3.client("s3")        # concrete, constructed deep inside

    def archive(self, name, body):
        self.s3.put_object(Bucket=BKT, Key=name, Body=body)
        log.info("archived %s", name)
```

After. The abstraction is declared by the *consumer* (`ReportArchiver`'s module), sized to what the consumer uses, and the concrete client is constructed once at the edge.

```python
class Stores(Protocol):
    def put(self, key: str, body: bytes) -> None: ...

class S3Stores:                            # adapter: the only place boto3 is imported
    def __init__(self, client, bucket): self.client, self.bucket = client, bucket
    def put(self, key, body): self.client.put_object(Bucket=self.bucket, Key=key, Body=body)

class ReportArchiver:
    def __init__(self, stores: Stores): self.stores = stores
    def archive(self, name, body):
        self.stores.put(name, body)
        log.info("archived %s", name)
```

`archive(ReportArchiver(StoresThatWritesToTempDir()))` in a test — no mocks, no monkeypatching, and `Stores` never had to be published as part of a library's public API.

---

<a name="decline"></a>
## The one to decline — abstraction with one implementation

Asked to make this "follow SOLID", the reflex answer is below. **Do not write this.**

```python
class IGreetingStrategy(ABC):
    @abstractmethod
    def greet(self, name: str) -> str: ...

class FormalGreetingStrategy(IGreetingStrategy):
    def greet(self, name): return f"Good day, {name}"

class GreetingStrategyFactory:
    _REGISTRY = {"formal": FormalGreetingStrategy}
    @classmethod
    def create(cls, kind: str) -> IGreetingStrategy: ...

class Greeter:
    def __init__(self, strategy: IGreetingStrategy): self.strategy = strategy
```

What's wrong: one strategy, so **O** had no open set to serve; `Greeter` never varies, so **D** inverted nothing; the factory has one entry and no callers who can register; the `Protocol` is a pass-through of a function that returns an f-string. Four types and two indirections replace one function.

The SOLID answer is the function, with the seam at rung 1:

```python
def greet(name: str, formatter: Callable[[str], str] = lambda n: f"Hello, {n}") -> str:
    return formatter(name)
```

Injectable, testable, extensible by passing something. When a second real formatter shows up in the codebase *and* callers need to select it, the registry earns its place.
