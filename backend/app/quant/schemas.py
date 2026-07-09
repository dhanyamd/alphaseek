from __future__ import annotations

import json

from pydantic import BaseModel, Field
from typing import Any


class DataReport(BaseModel):
    """Free-form report from data profiling step.
    Content is whatever the LLM decides is relevant — no fixed fields.
    Stored as a dict (LLM can include any keys).
    """

    content: dict[str, Any] = Field(default_factory=dict)
    raw_keys: list[str] = Field(default_factory=list)
    source: str = ""


class ExperimentPlan(BaseModel):
    """Structured experiment plan produced by the Synthesist.
    Auto-injected into the Coder's prompt via model_json_schema().
    """

    goal: str = ""
    data_columns: dict[str, str] = Field(default_factory=dict)
    methodology: list[str] = Field(default_factory=list)
    evaluation: str = ""
    constraints: list[str] = Field(default_factory=list)

    @classmethod
    def json_schema(cls) -> str:
        import json

        return json.dumps(cls.model_json_schema(), indent=2)

    @classmethod
    def json_example(cls) -> str:
        return json.dumps(
            {
                "goal": "State the research question here",
                "data_columns": {
                    "<role>": "<actual_column_name_from_data_report>",
                },
                "methodology": [
                    "Step 1 using the actual columns",
                    "Step 2",
                    "Step 3 — compute the metric the question demands",
                ],
                "evaluation": "How success is judged (metric must match the data type)",
                "constraints": [
                    "Use only columns present in the data report",
                    "No forward-looking bias",
                ],
            },
            indent=2,
        )
