"""Backtest error type.

Factor evaluation happens in the hardened sandbox (see docker_sandbox.py +
sandbox/runner.py), never here. This module only defines the error the sandbox
raises so the rest of the app can catch it.
"""
from __future__ import annotations


class FactorError(Exception):
    """Agent-written factor code failed in the sandbox (bad code, not infra)."""
