"""
Stage 4: Skill Catalog — 3-level Hierarchical Skill Directory
==============================================================
Organises skills as: domain > capability > skill

Provides search and query functions to locate the right skill for a
given user intent.

Skill entry structure:
    {
        "name": str,
        "description": str,
        "parameters": [ToolParameter, ...],
        "examples": [str, ...],
        "prompt_template": str,  # optional LLM prompt for this skill
    }
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ── Catalogue data ─────────────────────────────────────────────────

SKILL_CATALOG: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "backend": {
        "api_dev": {
            "create_endpoint": {
                "name": "create_endpoint",
                "description": "Create a new REST API endpoint (FastAPI / Flask / Django)",
                "parameters": [
                    {"name": "framework", "type": "string", "description": "Target framework (fastapi/flask/django)", "required": True},
                    {"name": "method", "type": "string", "description": "HTTP method (GET/POST/PUT/DELETE)", "required": True},
                    {"name": "path", "type": "string", "description": "URL path", "required": True},
                    {"name": "summary", "type": "string", "description": "Short description of the endpoint", "required": False},
                ],
                "examples": [
                    'create_endpoint(framework="fastapi", method="GET", path="/items/{id}")',
                ],
                "prompt_template": "Create a {method} endpoint at {path} using {framework}.",
            },
            "generate_model": {
                "name": "generate_model",
                "description": "Generate a Pydantic / SQLAlchemy data model",
                "parameters": [
                    {"name": "orm", "type": "string", "description": "SQLAlchemy / Pydantic / Django", "required": True},
                    {"name": "fields", "type": "string", "description": "Field definitions as JSON", "required": True},
                ],
                "examples": [
                    'generate_model(orm="sqlalchemy", fields=\'{"id": "int", "name": "str"}\')',
                ],
                "prompt_template": "Generate a {orm} model with fields: {fields}",
            },
        },
        "db_dev": {
            "write_query": {
                "name": "write_query",
                "description": "Write a SQL query for the given database type",
                "parameters": [
                    {"name": "db_type", "type": "string", "description": "postgresql / mysql / sqlite", "required": True},
                    {"name": "requirement", "type": "string", "description": "Natural language query requirement", "required": True},
                ],
                "examples": [
                    'write_query(db_type="postgresql", requirement="select all users created last week")',
                ],
                "prompt_template": "Write a {db_type} query to: {requirement}",
            },
            "migrate_schema": {
                "name": "migrate_schema",
                "description": "Generate a database migration script",
                "parameters": [
                    {"name": "db_type", "type": "string", "description": "Database type", "required": True},
                    {"name": "changes", "type": "string", "description": "Description of schema changes", "required": True},
                ],
                "examples": [],
                "prompt_template": "Generate a {db_type} migration: {changes}",
            },
        },
        "code_review": {
            "review_code": {
                "name": "review_code",
                "description": "Review source code for bugs, security issues, and style",
                "parameters": [
                    {"name": "file_path", "type": "string", "description": "Path to the file to review", "required": True},
                ],
                "examples": [],
                "prompt_template": "Review the code in {file_path} for issues.",
            },
        },
    },
    "frontend": {
        "component_gen": {
            "create_component": {
                "name": "create_component",
                "description": "Create a React/Vue component skeleton",
                "parameters": [
                    {"name": "framework", "type": "string", "description": "react / vue", "required": True},
                    {"name": "component_name", "type": "string", "description": "Component name (PascalCase)", "required": True},
                    {"name": "props", "type": "string", "description": "Comma-separated prop names", "required": False},
                ],
                "examples": [],
                "prompt_template": "Generate a {framework} component named {component_name}.",
            },
        },
    },
    "data": {
        "pipeline": {
            "etl_script": {
                "name": "etl_script",
                "description": "Generate an ETL pipeline script",
                "parameters": [
                    {"name": "source", "type": "string", "description": "Data source type", "required": True},
                    {"name": "target", "type": "string", "description": "Data target type", "required": True},
                ],
                "examples": [],
                "prompt_template": "Create an ETL pipeline from {source} to {target}.",
            },
        },
    },
    "devops": {
        "container": {
            "write_dockerfile": {
                "name": "write_dockerfile",
                "description": "Generate a Dockerfile for the specified base image and dependencies",
                "parameters": [
                    {"name": "base_image", "type": "string", "description": "Base image (e.g., python:3.10-slim)", "required": True},
                    {"name": "packages", "type": "string", "description": "Comma-separated packages to install", "required": False},
                ],
                "examples": [],
                "prompt_template": "Write a Dockerfile based on {base_image}.",
            },
        },
    },
}


# ── Search and query functions ─────────────────────────────────────

def get_domains() -> List[str]:
    """Return all top-level domain names."""
    return list(SKILL_CATALOG.keys())


def get_capabilities(domain: str) -> List[str]:
    """Return capability names for a given domain."""
    domain_data = SKILL_CATALOG.get(domain, {})
    return list(domain_data.keys())


def get_skills(domain: str, capability: str) -> List[Dict[str, Any]]:
    """Return all skill entries under a (domain, capability) pair."""
    capability_data = SKILL_CATALOG.get(domain, {}).get(capability, {})
    return list(capability_data.values())


def get_skill_by_path(path: str) -> Optional[Dict[str, Any]]:
    """
    Look up a skill by its dotted path, e.g. "backend.api_dev.create_endpoint".

    Returns the skill dict or None if not found.
    """
    parts = path.strip().split(".")
    if len(parts) != 3:
        return None
    domain, capability, skill_name = parts
    skill = SKILL_CATALOG.get(domain, {}).get(capability, {}).get(skill_name)
    return skill


def search_skills(query: str) -> List[Tuple[str, float]]:
    """
    Simple keyword-based skill search.
    Returns list of (skill_path, score) tuples sorted by relevance.

    Args:
        query: Natural language search query.

    Returns:
        List of (path, score) ordered descending by score.
    """
    query_lower = query.lower()
    query_terms = set(query_lower.split())
    results: List[Tuple[str, float]] = []

    for domain, cap_dict in SKILL_CATALOG.items():
        for capability, skill_dict in cap_dict.items():
            for skill_name, skill_data in skill_dict.items():
                path = f"{domain}.{capability}.{skill_name}"
                text = (
                    f"{skill_data['name']} {skill_data['description']} "
                    f"{' '.join(skill_data.get('examples', []))} "
                    f"{skill_data.get('prompt_template', '')}"
                ).lower()

                # Score: count matching terms + exact phrase bonus
                term_matches = sum(1 for t in query_terms if t in text)
                exact_bonus = 5.0 if query_lower in text else 0.0
                score = term_matches + exact_bonus

                if score > 0:
                    results.append((path, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def flatten_catalog() -> List[Dict[str, Any]]:
    """
    Flatten the 3-level catalog into a list of entries with a 'path' field.
    Useful for building embedding indices.
    """
    entries = []
    for domain, cap_dict in SKILL_CATALOG.items():
        for capability, skill_dict in cap_dict.items():
            for skill_name, skill_data in skill_dict.items():
                entry = dict(skill_data)
                entry["path"] = f"{domain}.{capability}.{skill_name}"
                entry["domain"] = domain
                entry["capability"] = capability
                entries.append(entry)
    return entries


def catalog_summary() -> Dict[str, int]:
    """Return summary statistics for the catalog."""
    domains = get_domains()
    total_capabilities = 0
    total_skills = 0
    for d in domains:
        caps = get_capabilities(d)
        total_capabilities += len(caps)
        for c in caps:
            total_skills += len(get_skills(d, c))
    return {
        "domains": len(domains),
        "capabilities": total_capabilities,
        "skills": total_skills,
    }
