Absolutely — here's a consolidated **TODO checklist** based on our discussion, aimed at enhancing the `uni20::Async<T>` and coroutine-based DAG execution system. This focuses on correctness, debuggability, and DAG introspection.

---

## ✅ **uni20 Async/DAG Infrastructure TODO**

### 🧠 **1. Coroutine Scheduler Architecture**

* [ ] Ensure `AsyncTask`, `GPUTask`, etc., all inherit from a `BasicPromise` that contains `Scheduler* sched_`.
* [ ] Scheduler sets `sched_` before first `.resume()`, and this must be enforced (cannot rely on user discipline).
* [ ] Validate that `sched_` always matches the actual scheduler type to enable safe downcasting.

---

### 🌐 **2. DAG and Epoch Queue Representation**

* [ ] Model each `Async<T>` as a sequence of **epochs**.
* [ ] Represent each `EpochContext` as:

  * One write edge (`Task → A[n]`)
  * One or more read edges (`A[n] → TaskX`)
* [ ] Emit this to `.dot` for DAG visualization.
* [ ] Use one node per epoch: `A0`, `A1`, etc.

---

### 🏷️ **3. DAG Visualization Enhancements**

* [ ] Use `rankdir=LR` for time-flow DAGs; support vertical layout for large graphs.
* [ ] Group tasks using `subgraph cluster_TaskX { ... }` to show internal coroutine structure.
* [ ] Annotate `co_await` access ordering inside each task subgraph.

---

### 📝 **4. Trace Access Metadata**

* [ ] Add `epoch_` and `source_location` fields to `ReadAwaitable` and `WriteAwaitable`.
* [ ] In debug builds only (`UNI20_ENABLE_CORO_TRACE`):

  * Enable recording of source location per `co_await`.
  * Store this in the `promise_type` or DAG emitter.

---

### 🧰 **5. ReadBuffer/WriteBuffer API**

* [ ] Switch to `buffer()` or `buffer.access()` instead of implicit `co_await buffer`.
* [ ] Capture `std::source_location::current()` via a default parameter.
* [ ] Drop `epoch_` and `source_location` fields in release builds.

---

### 🧪 **6. Runtime Stacktrace Support (Optional, Debug Builds)**

* [ ] In `await_suspend()`, use `std::stacktrace::current()` to find the `co_await` caller:

  * Use frame `[1]` as the coroutine callsite.
  * Extract file, line, function for DAG annotation.
* [ ] Filter out internal frames (`uni20::internal::`, `std::`, etc.)
* [ ] Requires `-lstdc++exp` and `-g` for full debug info.

---

### 🧾 **7. DOT DAG Output**

* [ ] Emit `.dot` format in two stages:

  * Static skeleton from promise constructor.
  * Dynamic detail from `co_await` trace (internal steps, access order).
* [ ] Add epoch number to each access edge: `"read A[1]"`, `"write B[2]"`.
* [ ] Use subgraphs for tasks and HTML-style labels if needed.

---

### 🧼 **8. Safety and Clean Build Behavior**

* [ ] Strip `source_location`, `epoch_`, and access logging in release builds.
* [ ] Define `UNI20_ENABLE_CORO_TRACE` to control all instrumentation.

---

Would you like this exported as a Markdown checklist or embedded as a tracked issue or document in the repo?


Yes, here's a **structured summary** of the discussion, including key insights and **actionable design decisions** for `uni20`.

---

## 🧠 Conceptual Summary

### 1. **Reverse-mode autodiff returns a covector**

* Backpropagation yields $\frac{\partial L}{\partial x^i}$, which is a **covector** (element of the cotangent space $T^*_xM$).
* This is appropriate and expected: gradients "live in" the dual space of the inputs.

### 2. **Model inputs $x^i$ as contravariant**

* Inputs to the model are elements of a tangent space $T_xM$, i.e., vectors.
* So the update $x^i \leftarrow x^i - \eta \, \delta x^i$ **only makes sense** if $\delta x^i$ is also contravariant.

### 3. **Update requires an index-raising operation**

* To turn $\frac{\partial L}{\partial x^i}$ into $\delta x^i$, we need to **apply the dual map** (a metric or isomorphism between $T^*_xM$ and $T_xM$).

---

## ✅ Firm Design Decisions for Uni20

### 🧩 1. Define a `dual_to_primal()` or `raise_index()` operation

* This maps an autodiff gradient (a covector) into a tangent vector.
* This will be **required before applying any updates**.

#### ✅ Action Item:

```cpp
template <typename T>
T raise_index(const Covector<T>& grad);  // or Grad<T>, depending on type naming
```

Alternatively:

```cpp
T delta = grad.dual_to_primal();
```

### 🏷️ 2. Consider symmetry label and structure mapping

* In tensor settings with symmetry, `dual_to_primal()`:

  * Must reverse arrows / conjugate labels
  * Possibly apply a symmetry-isometry (like time-reversal or charge conjugation)

#### ✅ Action Item:

Implement `dual_to_primal()` such that:

* For symmetric tensors: returns a new tensor with flipped symmetry labels.
* For scalar types (e.g. `double`): acts as identity.

### 🔀 3. Avoid conflating covectors and vectors in autodiff outputs

* Keep gradient buffers as semantically distinct types (e.g., `Grad<T>` or `Covector<T>`)
* Avoid allowing direct arithmetic with `x.value` without explicit lifting.

#### ✅ Action Item:

Define update APIs such that:

```cpp
x.value = x.value - eta * x.grad().dual_to_primal();
```

is the preferred idiom.

---

## 📚 Optional / Deferred Ideas

* **Generalized inner product / metric map** between tangent and cotangent spaces (for nontrivial geometries)
* **Tagged types** to help enforce correct operations between contravariant and covariant tensors
* **Compile-time traits** to help autodiff or optimization layers understand which quantities live in which space

---

Would you like a prototype `dual_to_primal()` function signature and a unit test case for it using dummy symmetry tags?



Here’s a summary of the discussion distilled into clear **action items for `uni20`**, focused on integrating exception propagation into the async and autodiff mechanisms:

---

## ✅ **Action Items: Exception Support for `uni20::Async<T>`**

### 1. **Add exception support to `WriteBuffer<T>`**

* Add a `std::exception_ptr` member to `WriteBufferBase` or `WriteBuffer<T>`.
* Add methods:

  ```cpp
  void set_error(std::exception_ptr);
  std::exception_ptr get_error() const;
  bool has_error() const;
  ```

### 2. **Extend `Async<T>` / `AsyncTask<T>` to carry exceptions**

* Add `std::exception_ptr error_` to `AsyncTask<T>` or `Async<T>`’s internal state.
* Modify `await_resume()` to rethrow if `error_` is set.
* Add `set_exception(std::exception_ptr)` method to allow upstream propagation.

### 3. **Update coroutine `promise_type`**

* Implement `promise_type::unhandled_exception()` to capture `std::current_exception()` and call `set_exception(...)`.

### 4. **Propagate exceptions during scheduling in `EpochContext`**

* When a `WriteBuffer<T>` is in an error state, copy its `exception_ptr` to all downstream `AsyncTask<T>`s:

  ```cpp
  for (ReadBuffer* r : readers)
      r->get_async_task()->set_exception(write_buf->get_error());
  ```
* This prevents compute from running and ensures all `co_await` calls throw.

### 5. **Ensure shared exception propagation**

* Use a **single `exception_ptr` instance** and fan it out to all consumers.
* Avoid creating multiple `std::make_exception_ptr(...)` instances for the same error.

---

## 🧠 Optional Enhancements (Post-MVP)

* Define custom exception types like:

  ```cpp
  struct autodiff_error : std::runtime_error { using std::runtime_error::runtime_error; };
  struct non_differentiable_error : autodiff_error { ... };
  ```
* Add error origin metadata (e.g., tensor name, kernel name, source location) in exceptions.
* Log exceptions at injection, not at access.
* Consider integrating `std::stacktrace` (C++23) to capture the call stack when injecting exceptions.
* Add API to inspect exception types without rethrowing (e.g., via `dynamic_cast` on a caught copy).

---

## 🧪 Test Coverage Tasks

* [ ] Test a kernel that injects an exception into a `WriteBuffer`.
* [ ] Verify all `ReadBuffer`s co\_awaiting that value rethrow the same exception.
* [ ] Verify exception handling logic in nested coroutines (`co_await` on a `ReadBuffer` inside another `AsyncTask`).
* [ ] Ensure exception cleanup doesn't leak references (e.g., refcounted `TensorRef` or DAG edges).

---

Let me know if you'd like a small test kernel that injects a `non_differentiable_error` at runtime so you can validate the error propagation end-to-end.



