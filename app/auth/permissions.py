"""Turns a validated identity into the set of ACL principals used to filter retrieval."""

from dataclasses import dataclass

from app.auth.entra import AuthenticatedUser


def user_principal(object_id: str) -> str:
    return f"user:{object_id}"


def group_principal(group_id: str) -> str:
    return f"group:{group_id}"


@dataclass(frozen=True)
class Permissions:
    user_id: str
    principals: frozenset[str]
    roles: frozenset[str] = frozenset()

    def has_role(self, role: str) -> bool:
        return role in self.roles


def resolve_permissions(user: AuthenticatedUser) -> Permissions:
    if user.groups_overage:
        # Fail closed: computing permissions without the real groups would silently
        # under- or over-grant depending on how the ACLs are written.
        raise ValueError("group membership must be resolved before computing permissions")
    principals = {user_principal(user.object_id)} | {group_principal(g) for g in user.groups}
    return Permissions(
        user_id=user.object_id,
        principals=frozenset(principals),
        roles=frozenset(user.roles),
    )
