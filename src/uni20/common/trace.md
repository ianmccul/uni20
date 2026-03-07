#### 1. **`TRACE_SCOPE` / `ScopedTrace` RAII class**

To trace entry/exit automatically:

```cpp
struct TraceScope {
  std::string name;
  TraceScope(std::string_view label) : name(label) {
    TRACE("enter", name);
  }
  ~TraceScope() {
    TRACE("exit", name);
  }
};
```

Macro:

```cpp
#define TRACE_SCOPE(label) trace::TraceScope __trace_scope__(label)
```

Optionally, include duration measurement.

---

#### 2. **`TraceLifetime` struct member**

For classes that want automatic tracing of construction, move, copy, destruction:

```cpp
struct TraceLifetime {
  std::string name;
  TraceLifetime(std::string_view n) : name(n) { TRACE("construct", name); }
  ~TraceLifetime() { TRACE("destruct", name); }
};
```

Used as a member in user-defined types:

```cpp
struct MyTensor {
  TraceLifetime __trace{"MyTensor"};
};
```

---

#### 3. **Support for heatmap / visual tensor formatting**

When printing `mdspan` or `Tensor` values:

* Color-coded numerical values (linear or log)
* Optional display modes:

  * Raw values
  * Sparsity pattern (`.` or `#`)
  * Min/max-highlight
* Would be implemented as custom `formatter<mdspan<...>>` or via `formatValue()` overloads
* Hooked into `TRACE()` / `CHECK()` print system

