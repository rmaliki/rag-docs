"""Print signed dev tokens for the fictional Acme Corp demo users (AUTH_MODE=dev only).

Usage: python -m scripts.create_demo_users
"""

import sys

from app.auth.entra import create_dev_token
from app.config import get_settings

DEMO_USERS: dict[str, dict] = {
    "marketing": {
        "object_id": "00000000-0000-4000-8000-00000000a001",
        "name": "Demo Marketing Employee",
        "groups": ["all-employees", "marketing-team"],
    },
    "hr": {
        "object_id": "00000000-0000-4000-8000-00000000a002",
        "name": "Demo HR Employee",
        "groups": ["all-employees", "hr-team"],
    },
    "finance": {
        "object_id": "00000000-0000-4000-8000-00000000a003",
        "name": "Demo Finance Employee",
        "groups": ["all-employees", "finance-team"],
    },
    "executive": {
        "object_id": "00000000-0000-4000-8000-00000000a004",
        "name": "Demo Executive",
        "groups": ["all-employees", "executives"],
    },
    "admin": {
        "object_id": "00000000-0000-4000-8000-00000000a005",
        "name": "Demo Knowledge Admin",
        "groups": ["all-employees"],
        "roles": ["Documents.Admin"],
    },
}


def main() -> None:
    settings = get_settings()
    if settings.auth_mode != "dev" or settings.dev_jwt_secret is None:
        sys.exit("Demo tokens only work with AUTH_MODE=dev and DEV_JWT_SECRET set.")
    secret = settings.dev_jwt_secret.get_secret_value()
    for key, user in DEMO_USERS.items():
        token = create_dev_token(
            secret,
            object_id=user["object_id"],
            name=user["name"],
            groups=user["groups"],
            roles=user.get("roles"),
        )
        print(f"export TOKEN_{key.upper()}={token}")


if __name__ == "__main__":
    main()
