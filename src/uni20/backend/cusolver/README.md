# `src/uni20/backend/cusolver`

This directory is the cuSOLVER backend-library wiring point. It currently holds
target scaffolding for future cuSOLVER wrappers.

## Contents

- `CMakeLists.txt`: cuSOLVER backend target setup.

## Notes

- cuSOLVER support should be gated by both build-time availability and
  operation-specific runtime capability checks.
- Higher-level linalg entry points belong under `src/uni20/linalg/backends`.
