"""Request authentication and authorization dependencies."""

import asyncio
import dataclasses
import logging
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.entra import AuthenticatedUser, AuthError
from app.auth.permissions import Permissions, resolve_permissions
from app.monitoring import metrics

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> AuthenticatedUser:
    if credentials is None:
        metrics.UNAUTHORIZED_REQUESTS.labels(reason="missing_token").inc()
        raise _unauthorized()
    try:
        # JWKS fetches are blocking network calls (cached afterwards): keep them off the event loop.
        user = await asyncio.to_thread(request.app.state.token_validator.validate, credentials.credentials)
    except AuthError as exc:
        metrics.UNAUTHORIZED_REQUESTS.labels(reason="invalid_token").inc()
        logger.info("rejected bearer token", extra={"reason": str(exc)})
        raise _unauthorized() from None

    if user.groups_overage:
        resolver = request.app.state.group_resolver
        if resolver is None:
            metrics.UNAUTHORIZED_REQUESTS.labels(reason="group_overage").inc()
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Group membership could not be resolved")
        try:
            groups = await resolver.member_groups(user.object_id)
        except httpx.HTTPError:
            logger.exception("Microsoft Graph group lookup failed")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Identity provider unavailable") from None
        user = dataclasses.replace(user, groups=groups, groups_overage=False)
    return user


def get_permissions(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> Permissions:
    return resolve_permissions(user)


def require_admin(request: Request, permissions: Annotated[Permissions, Depends(get_permissions)]) -> Permissions:
    if not permissions.has_role(request.app.state.settings.admin_role):
        metrics.UNAUTHORIZED_REQUESTS.labels(reason="missing_role").inc()
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient role")
    return permissions


CurrentPermissions = Annotated[Permissions, Depends(get_permissions)]
AdminPermissions = Annotated[Permissions, Depends(require_admin)]
