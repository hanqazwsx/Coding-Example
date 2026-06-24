"""
Stage 3: Tool Schema
====================
Pydantic models for tool definitions, parameters, and return values.
Provides a unified contract between the tool registry, executor, and LLM.

All tools return ToolResult — a standardised envelope:
    {"success": bool, "result": Any, "error": str, "duration_ms": float}
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Schema for a single tool parameter."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(default="string", description="Parameter type (string, integer, boolean, etc.)")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether this parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value (if any)")


class ToolSpec(BaseModel):
    """
    Specification for a tool that the LLM can call.
    This is converted into an OpenAI-compatible JSON schema for function calling.
    """
    name: str = Field(..., description="Unique tool name")
    description: str = Field(default="", description="What this tool does")
    parameters: List[ToolParameter] = Field(
        default_factory=list,
        description="List of parameter schemas",
    )
    category: str = Field(default="general", description="Tool category for grouping")
    tags: List[str] = Field(default_factory=list, description="Tags for search/filtering")
    safe_for_sandbox: bool = Field(default=True, description="Whether this tool is safe to expose")

    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Convert this spec to an OpenAI-compatible tool definition
        (usable with ChatOpenAI.bind_tools).
        """
        properties = {}
        required_params = []

        for p in self.parameters:
            param_schema: Dict[str, Any] = {"type": p.type}
            if p.description:
                param_schema["description"] = p.description
            if p.default is not None:
                param_schema["default"] = p.default
            properties[p.name] = param_schema
            if p.required:
                required_params.append(p.name)

        parameters_schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required_params:
            parameters_schema["required"] = required_params

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters_schema,
            },
        }


class ToolResult(BaseModel):
    """
    Unified return envelope for every tool execution.
    """
    success: bool = Field(default=False, description="Whether the tool call succeeded")
    result: Any = Field(default=None, description="The tool's output (if successful)")
    error: str = Field(default="", description="Error message (if failed)")
    duration_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    tool_name: str = Field(default="", description="Name of the tool that was called")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (warnings, truncation info, etc.)",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict for JSON serialisation."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tool_name": self.tool_name,
            "metadata": self.metadata,
        }

    def __bool__(self) -> bool:
        """A ToolResult is truthy when it succeeded."""
        return self.success


# ── JSON Schema for parameter validation ───────────────────────────

# Shorthand: define common parameter patterns
PARAM_PATH = ToolParameter(
    name="path",
    type="string",
    description="File path (absolute or relative to project root)",
    required=True,
)
PARAM_CONTENT = ToolParameter(
    name="content",
    type="string",
    description="File content to write",
    required=True,
)
PARAM_COMMAND = ToolParameter(
    name="command",
    type="string",
    description="Shell command to execute",
    required=True,
)
