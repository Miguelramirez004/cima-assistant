"""Verificación de tokens de Supabase Auth con claves ES256 generadas en el test."""

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from server.auth import AuthError, TokenVerifier
from server.supabase_rest import SupabaseError

URL = "https://proj.supabase.co"
KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())


def public_jwk(private_key, kid):
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(private_key.public_key()))
    return {**jwk, "kid": kid, "alg": "ES256", "use": "sig"}


class FakeRest:
    url = URL

    def __init__(self):
        self.jwks_calls = 0
        self.user_calls = 0

    async def get_jwks(self):
        self.jwks_calls += 1
        return {"keys": [public_jwk(KEY, "kid-1")]}

    async def get_user(self, token):
        self.user_calls += 1
        if token != "legacy-valid":
            raise SupabaseError(401, "invalid JWT")
        return {"id": "legacy-user", "email": "legacy@example.test"}


def token(key=KEY, kid="kid-1", **overrides):
    claims = {"sub": "user-1", "email": "u@example.test", "role": "authenticated",
              "aud": "authenticated", "iss": f"{URL}/auth/v1", "exp": int(time.time()) + 300}
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="ES256", headers={"kid": kid})


@pytest.fixture
def rest():
    return FakeRest()


async def test_valid_es256_token(rest):
    user = await TokenVerifier(rest).verify(token())
    assert user.id == "user-1" and user.email == "u@example.test"


async def test_jwks_is_cached(rest):
    verifier = TokenVerifier(rest)
    await verifier.verify(token())
    await verifier.verify(token())
    assert rest.jwks_calls == 1


@pytest.mark.parametrize("bad, reason", [
    (lambda: token(exp=int(time.time()) - 10), "expired"),
    (lambda: token(aud="other"), "wrong audience"),
    (lambda: token(iss="https://evil.supabase.co/auth/v1"), "wrong issuer"),
    (lambda: token(role="anon"), "anon role"),
    (lambda: token(key=OTHER_KEY), "signed with another key"),
])
async def test_rejected_tokens(rest, bad, reason):
    with pytest.raises(AuthError):
        await TokenVerifier(rest).verify(bad())


async def test_unknown_kid_rejected_without_hammering_jwks(rest):
    verifier = TokenVerifier(rest)
    await verifier.verify(token())
    for _ in range(3):
        with pytest.raises(AuthError):
            await verifier.verify(token(kid="made-up"))
    assert rest.jwks_calls == 1


async def test_hs256_tokens_are_checked_with_auth_server(rest):
    verifier = TokenVerifier(rest)
    legacy = jwt.encode({"sub": "x"}, "shared-secret-at-least-32-bytes-long!!", algorithm="HS256")
    with pytest.raises(AuthError):
        await verifier.verify(legacy)
    assert rest.user_calls == 1


async def test_garbage_token(rest):
    with pytest.raises(AuthError):
        await TokenVerifier(rest).verify("not-a-jwt")
