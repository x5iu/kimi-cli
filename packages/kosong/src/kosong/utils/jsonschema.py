from __future__ import annotations

import copy
from typing import cast

from kosong.utils.typing import JsonType

type JsonDict = dict[str, JsonType]


def deref_json_schema(schema: JsonDict) -> JsonDict:
    """Expand local `$ref` entries in a JSON Schema without infinite recursion."""
    # Work on a deep copy so we never mutate the caller's schema.
    full_schema: JsonDict = copy.deepcopy(schema)

    def resolve_pointer(root: JsonDict, pointer: str) -> JsonType:
        """Resolve a JSON Pointer (e.g. ``#/$defs/User``) inside the schema."""
        parts = pointer.lstrip("#/").split("/")
        current: JsonType = root
        try:
            for part in parts:
                if isinstance(current, dict):
                    current = current[part]
                else:
                    raise ValueError
            return current
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Unable to resolve reference path: {pointer}") from None

    def traverse(node: JsonType, root: JsonDict) -> JsonType:
        """Recursively traverse every node to inline local references."""
        if isinstance(node, dict):
            # Replace local ``$ref`` entries with their referenced payload.
            if "$ref" in node and isinstance(node["$ref"], str):
                ref_path = node["$ref"]
                if ref_path.startswith("#"):
                    # Resolve the local reference target.
                    target = resolve_pointer(root, ref_path)
                    # Recursively inline the target in case it contains more refs.
                    ref = traverse(target, root)
                    if not isinstance(ref, dict):
                        msg = "Local $ref must resolve to a JSON object"
                        raise TypeError(msg)
                    node.pop("$ref")
                    node.update(ref)
                    return node
                else:
                    # Ignore remote references such as http://...
                    return node

            # Traverse the remaining mapping entries.
            return {k: traverse(v, root) for k, v in node.items()}

        elif isinstance(node, list):
            # Traverse list members (e.g. allOf, oneOf, items).
            return [traverse(item, root) for item in node]

        else:
            return node

    def _detect_discriminator(branches: list[JsonDict]) -> str | None:
        """Return the shared discriminator property name if every *oneOf*
        branch is an ``object`` with exactly one ``const`` property in common,
        otherwise ``None``."""
        candidates: set[str] | None = None
        for branch in branches:
            if branch.get("type") != "object":
                return None
            props = branch.get("properties")
            if not isinstance(props, dict):
                return None
            const_keys = {k for k, v in props.items() if isinstance(v, dict) and "const" in v}
            if not const_keys:
                return None
            candidates = const_keys if candidates is None else candidates & const_keys
            if not candidates:
                return None
        # Exactly one shared const property is the discriminator.
        if candidates and len(candidates) == 1:
            return next(iter(candidates))
        return None

    def flatten_oneof(node: JsonType) -> JsonType:
        """Flatten ``oneOf`` discriminated-union patterns into a single object.

        When every ``oneOf`` branch is an object sharing one ``const``
        discriminator property, merge them into:
        - discriminator becomes ``enum``
        - all properties are merged (descriptions kept from first seen)
        - ``required`` is the intersection (only the discriminator key)
        """
        if isinstance(node, dict):
            # Recurse first so nested structures are already simplified.
            node = {k: flatten_oneof(v) for k, v in node.items()}

            one_of = node.get("oneOf")
            if isinstance(one_of, list) and len(one_of) >= 2:
                branches = cast(list[JsonDict], one_of)
                disc_key = _detect_discriminator(branches)
                if disc_key is not None:
                    enum_values: list[JsonType] = []
                    merged_props: JsonDict = {}
                    merged_required: set[str] | None = None

                    for branch in branches:
                        props = cast(JsonDict, branch.get("properties", {}))
                        req = set(cast(list[str], branch.get("required", [])))

                        # Collect the const value for the discriminator.
                        disc_prop = cast(JsonDict, props[disc_key])
                        enum_values.append(disc_prop["const"])

                        # Merge properties (first-seen wins for descriptions).
                        for k, v in props.items():
                            if k == disc_key:
                                continue
                            if k not in merged_props:
                                merged_props[k] = v

                        # required = intersection across branches + disc_key
                        if merged_required is None:
                            merged_required = req
                        else:
                            merged_required &= req

                    # Build the discriminator property.
                    disc_schema: JsonDict = {
                        "type": "string",
                        "enum": enum_values,
                    }
                    # Keep description from any branch if present.
                    for branch in branches:
                        bp = cast(JsonDict, branch.get("properties", {}))
                        bd = cast(JsonDict, bp.get(disc_key, {}))
                        if "description" in bd:
                            disc_schema["description"] = bd["description"]
                            break

                    final_props: JsonDict = {disc_key: disc_schema, **merged_props}
                    final_required = sorted({disc_key} | (merged_required or set()))

                    flat: JsonDict = {"type": "object", "properties": final_props}
                    if final_required:
                        flat["required"] = cast(JsonType, final_required)
                    # Preserve sibling keys (e.g. description) from the parent,
                    # but drop oneOf (replaced) and discriminator (now redundant).
                    for k, v in node.items():
                        if k not in ("oneOf", "discriminator"):
                            flat[k] = v
                    return flat

            return node

        elif isinstance(node, list):
            return [flatten_oneof(item) for item in node]

        return node

    # Remove definition buckets to keep the resolved schema minimal.
    resolved = cast(JsonDict, traverse(full_schema, full_schema))

    # Comment these lines if you want to keep the emitted definitions.
    resolved.pop("$defs", None)
    resolved.pop("definitions", None)

    # Flatten discriminated oneOf unions into plain objects.
    resolved = cast(JsonDict, flatten_oneof(resolved))

    return resolved
