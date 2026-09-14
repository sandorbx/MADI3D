"""Explicit comment-copy planning, independent of Qt and project I/O."""
from dataclasses import dataclass

from .summary import is_derived


@dataclass(frozen=True)
class CommentTarget:
    key: object
    name: str
    metadata: dict
    comments: tuple[str, ...]
    match_key: tuple[str, str] | None


@dataclass(frozen=True)
class CommentChange:
    target_key: object
    name: str
    column: int
    before: str
    after: str


@dataclass(frozen=True)
class CommentPlan:
    changes: tuple[CommentChange, ...]
    skipped: tuple[str, ...]

    @property
    def replacements(self):
        return sum(bool(c.before) for c in self.changes)


def plan_comments(index, targets, field_columns, *, compact_column=None, replace=False):
    """Plan exact cell assignments (one-based columns), never append text.

    field_columns maps stable EvidenceField keys to columns. For compact output,
    only its keys matter. Missing fields produce no assignment, including when
    replacing. An ambiguous object requires a key explicitly chosen by the user.
    """
    columns = [compact_column] if compact_column is not None else list(field_columns.values())
    if any(type(c) is not int or c < 1 for c in columns):
        raise ValueError("Choose a comment column for every selected field.")
    if compact_column is None and len(set(columns)) != len(columns):
        raise ValueError("Separate fields must use different comment columns.")
    changes, skipped, seen = [], [], set()
    for target in targets:
        if target.key in seen:
            continue
        seen.add(target.key)
        matches = index.for_object(target.metadata)
        match = next((m for m in matches if m.key == target.match_key), None)
        if match is None:
            skipped.append(f"{target.name}: choose a search/match; unchanged.")
            continue
        fields = {f.key: f for f in match.fields}
        values = [(fields[key], column) for key, column in field_columns.items()
                  if key in fields and fields[key].text not in (None, "")]
        derived = is_derived(target.metadata)
        if compact_column is not None:
            prefix = "NeuronBridge source-match evidence" if derived else "NeuronBridge"
            assignments = [(compact_column, prefix + ": " + "; ".join(
                f"{field.label}={field.text}" for field, _ in values))] if values else []
        else:
            assignments = [(column, ("Source-match evidence: " if derived else "") + field.text)
                           for field, column in values]
        for column, after in assignments:
            if column > len(target.comments):
                raise ValueError("A target comment column no longer exists.")
            before = target.comments[column - 1]
            if before == after:
                continue
            if before and not replace:
                skipped.append(f"{target.name}, column {column}: existing text retained.")
                continue
            changes.append(CommentChange(target.key, target.name, column, before, after))
    return CommentPlan(tuple(changes), tuple(skipped))
