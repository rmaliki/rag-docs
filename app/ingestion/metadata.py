"""Document metadata, read from a `<file>.meta.json` sidecar next to each source document."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.auth.permissions import group_principal, user_principal


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    title: str = Field(min_length=1, max_length=500)
    department: str = Field(min_length=1, max_length=64)
    classification: Literal["public", "internal", "confidential", "restricted"] = "internal"
    version: int = Field(default=1, ge=1)
    # Group IDs must match the `groups` claim in users' tokens (Entra group object IDs,
    # or the demo group names when AUTH_MODE=dev). Users are Entra object IDs.
    allowed_groups: tuple[str, ...] = ()
    allowed_users: tuple[str, ...] = ()
    public: bool = False

    @model_validator(mode="after")
    def _check_access(self) -> "DocumentMetadata":
        if self.public and self.classification in {"confidential", "restricted"}:
            raise ValueError(f"a {self.classification} document cannot be public")
        if not self.public and not (self.allowed_groups or self.allowed_users):
            raise ValueError("a non-public document needs at least one allowed group or user")
        return self

    def acl_principals(self) -> list[str]:
        principals = {group_principal(g) for g in self.allowed_groups}
        principals |= {user_principal(u) for u in self.allowed_users}
        return sorted(principals)


def metadata_path_for(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def load_metadata(path: Path) -> DocumentMetadata:
    return DocumentMetadata.model_validate_json(metadata_path_for(path).read_text(encoding="utf-8"))
