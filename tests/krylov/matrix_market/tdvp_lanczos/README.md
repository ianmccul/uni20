# TDVP Lanczos Exponential Fixture

This fixture is copied from the local `cytnx-fixes` investigation directory:

```text
/home/ian/sync/git/cytnx-fixes/local/jerry_ho_lanczos
```

The files describe a 256-dimensional harmonic-oscillator Hamiltonian and the
initial vector used to reproduce an old `Lanczos_Exp` stopping-criterion
failure:

| File | Description |
| --- | --- |
| `tdvp_ho_n8_hamiltonian.mtx` | 256 x 256 real symmetric Matrix Market Hamiltonian. |
| `tdvp_ho_n8_v0.mtx` | 256 x 1 real Matrix Market initial vector. |
| `metadata.json` | Source parameters and summary of the diagnostic cases. |
| `old_lanczos_exp_scan.csv` | Scan comparing the old error indicator with dense-reference error. |

The most relevant regression row is the `complex<double>` case
`realtime_double_factor10000`:

```text
tau = -0.1968473663975394 i
krylov_dim = 33
old error indicator = 8.47677729418914e-09
true dense expm error = 1.3982341149064179e-08
```

So the old raw projected-tail indicator reports a value below `1e-8`, while
the true action error is still above `1e-8`.

There is also a sharper `complex<float>` row, `realtime_float_factor1000`:

```text
tau = -0.01968473663975394 i
krylov_dim = 10
old error indicator = 4.197080093239507e-09
true dense expm error = 1.5685087557098518e-07
```

That case has a true action error about 37 times larger than the old indicator.

The renamed files correspond to the original source files:

| Current file | Original source file |
| --- | --- |
| `tdvp_ho_n8_hamiltonian.mtx` | `jerry_ho_n8_hamiltonian.mtx` |
| `tdvp_ho_n8_v0.mtx` | `jerry_ho_n8_v0.mtx` |

The original extractor is:

```text
/home/ian/sync/git/cytnx-fixes/local/extract_jerry_ho_lanczos_matrix.py
```
