#!/usr/bin/env python3
"""Install Meno beside one local Hermes profile and verify the connection."""

from __future__ import annotations

import argparse
import getpass
import importlib
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

REPO = Path(__file__).resolve().parents[1]
PROFILE_KEYS = {
    "MENO_API_URL": "http://127.0.0.1:8765",
    "MENO_API_TOKEN": "",
    "MENO_USER_ID": "",
}


def _backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, backup)
    backup.chmod(0o600)


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = shlex.split(value)[0]
        values[key.strip()] = value
    return values


def _env_value(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("environment values cannot contain newlines")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_env(path: Path, additions: dict[str, str], *, mode: int) -> None:
    existing = _parse_env(path)
    for key, value in additions.items():
        if key in existing and existing[key] != value:
            raise ValueError(f"{path} already defines a different {key}; resolve it manually")
    missing = {key: value for key, value in additions.items() if key not in existing}
    if not missing:
        path.chmod(mode)
        return
    _backup(path)
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    if original and not original.endswith("\n"):
        original += "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), mode)
        handle.write(
            original + "".join(f"{key}={_env_value(value)}\n" for key, value in missing.items())
        )


def _set_provider(config_path: Path) -> None:
    config: dict = {}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as source:
            config = yaml.safe_load(source) or {}
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping")
    memory = config.setdefault("memory", {})
    if not isinstance(memory, dict):
        raise TypeError("config.yaml memory must be a mapping")
    if memory.get("provider") == "meno":
        return
    _backup(config_path)
    memory["provider"] = "meno"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def install(home: Path, *, siliconflow_key: str, run_commands: bool = True) -> Path:
    home = home.expanduser().resolve()
    data = home / "meno-data"
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    data.chmod(0o700)
    env_file = data / "meno.env"
    old = _parse_env(env_file)
    profile_env = home / ".env"
    profile = _parse_env(profile_env)
    token = old.get("MENO_API_TOKEN") or profile.get("MENO_API_TOKEN") or secrets.token_urlsafe(32)
    user_id = (
        profile.get("MENO_USER_ID")
        or old.get("MENO_USER_ID")
        or f"profile-{secrets.token_urlsafe(18)}"
    )
    key = old.get("MENO_OPENAI_API_KEY") or siliconflow_key
    if len(key) < 20:
        raise ValueError("SiliconFlow API key must contain at least 20 characters")

    _write_env(
        env_file,
        {
            "MENO_ENV": "production",
            "MENO_DATABASE_URL": f"sqlite:////{data.as_posix().lstrip('/')}/meno.db",
            "MENO_VECTOR_MODE": "memory",
            "MENO_VECTOR_MAX_RESIDENT": "10000",
            "MENO_USER_TOKEN_MATERIALIZATION_ENABLED": "true",
            "MENO_CONTEXT_ACTIVATION_ENABLED": "true",
            "MENO_PREFERENCE_DISTRIBUTION_ENABLED": "true",
            "MENO_REFLECTION_ENABLED": "true",
            "MENO_EVIDENCE_SELECTION_ENABLED": "true",
            "MENO_API_HOST": "127.0.0.1",
            "MENO_API_PORT": "8765",
            "MENO_API_TOKEN": token,
            "MENO_OPENAI_API_KEY": key,
            "MENO_AUDIT_SPILL_PATH": str(data / "audit-spill.jsonl"),
        },
        mode=0o600,
    )

    venv = data / ".venv"
    if run_commands:
        if not venv.exists():
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / "bin" / "python"
        subprocess.run([str(python), "-m", "pip", "install", str(REPO)], check=True)
        env = os.environ | _parse_env(env_file)
        env["HERMES_HOME"] = str(home)
        subprocess.run([str(python), "-m", "meno.cli", "migrate"], env=env, check=True)

    _write_env(
        profile_env,
        {
            "MENO_API_URL": PROFILE_KEYS["MENO_API_URL"],
            "MENO_API_TOKEN": token,
            "MENO_USER_ID": user_id,
        },
        mode=0o600,
    )

    plugin = home / "plugins" / "meno" / "__init__.py"
    plugin.parent.mkdir(parents=True, exist_ok=True)
    shim = f"""# Generated by Meno's Hermes installer.
import sys

_meno_src = {str(REPO / "src")!r}
if _meno_src not in sys.path:
    sys.path.insert(0, _meno_src)

from meno.hermes_plugin import MenoMemoryProvider

def register(ctx):
    ctx.register_memory_provider(MenoMemoryProvider())
"""
    if not plugin.exists() or plugin.read_text(encoding="utf-8") != shim:
        _backup(plugin)
        plugin.write_text(shim, encoding="utf-8")

    _set_provider(home / "config.yaml")

    return env_file


def _hermes_source(home: Path, explicit: str | None) -> None:
    candidates = [Path(explicit).expanduser()] if explicit else []
    executable = Path(sys.executable)
    candidates.extend([executable.parent, *executable.parents])
    candidates.append(home.parent / "hermes-agent")
    candidates.append(home / "hermes-agent")
    for candidate in candidates:
        if (candidate / "plugins" / "memory" / "__init__.py").is_file():
            sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError("Hermes source containing plugins/memory/__init__.py was not found")


def _wait_ready(base: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=2.0, trust_env=False) as client:
                client.get(f"{base.rstrip('/')}/health/ready").raise_for_status()
            return
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Meno did not become healthy within {timeout:g}s") from last_error


def check(home: Path, hermes_dir: str | None = None) -> None:
    home = home.expanduser().resolve()
    os.environ["HERMES_HOME"] = str(home)
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
    selected = (config.get("memory") or {}).get("provider")
    if selected != "meno":
        raise RuntimeError(f"Hermes selected memory.provider={selected!r}, expected 'meno'")
    for key, value in _parse_env(home / ".env").items():
        if key.startswith("MENO_"):
            os.environ[key] = value

    _hermes_source(home, hermes_dir)
    loader = importlib.import_module("plugins.memory").load_memory_provider
    provider = loader("meno", register_skills=False)
    if not provider or not provider.is_available():
        raise RuntimeError("Hermes loaded Meno, but the provider is unavailable")

    base = os.environ.get("MENO_API_URL", PROFILE_KEYS["MENO_API_URL"]).rstrip("/")
    token = os.environ.get("MENO_API_TOKEN", "")
    user_id = os.environ.get("MENO_USER_ID", "")
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        client.get(f"{base}/health/ready").raise_for_status()
        response = client.post(
            f"{base}/v1/retrieve",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "user_id": user_id,
                "purpose": "response_personalization",
                "context": {"query": "Meno connection check", "platform": "hermes-check"},
                "constraints": {"max_facets": 1, "max_rendered_tokens": 64},
            },
        )
        response.raise_for_status()
        if response.json().get("degraded"):
            raise RuntimeError(
                "Meno retrieval is degraded; check embedding credentials and service logs"
            )
    print("Hermes loaded Meno; local health and authenticated retrieve passed.")


def _unit_quote(path: Path) -> str:
    value = str(path).replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def _write_unit(home: Path, unit_dir: Path) -> Path:
    unit = _unit_path(home, unit_dir)
    unit_dir.mkdir(parents=True, exist_ok=True)
    _backup(unit)
    unit.write_text(
        "[Unit]\nDescription=Meno memory sidecar\nAfter=network.target\n\n"
        "[Service]\nType=simple\n"
        f"EnvironmentFile={_unit_quote(home / 'meno-data/meno.env')}\n"
        f"WorkingDirectory={_unit_quote(REPO)}\n"
        f"ExecStart={_unit_quote(home / 'meno-data/.venv/bin/python')} -m meno.cli serve\n"
        "Restart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n",
        encoding="utf-8",
    )
    return unit


def _unit_path(home: Path, unit_dir: Path) -> Path:
    unit = unit_dir / "meno.service"
    if unit.exists() and (
        f"EnvironmentFile={_unit_quote(home / 'meno-data/meno.env')}"
        not in unit.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "Existing meno.service uses different storage; follow the manual upgrade guide"
        )
    return unit


def _start(
    home: Path,
    *,
    is_root: bool,
    hermes_dir: str | None,
    unit_dir: Path | None = None,
) -> None:
    if is_root:
        _write_unit(home, unit_dir or Path("/etc/systemd/system"))
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", "meno.service"], check=True)
        subprocess.run(["systemctl", "restart", "meno.service"], check=True)
    else:
        _write_unit(home, unit_dir or Path.home() / ".config/systemd/user")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "meno.service"], check=True)
        subprocess.run(["systemctl", "--user", "restart", "meno.service"], check=True)
    _wait_ready(PROFILE_KEYS["MENO_API_URL"])
    check(home, hermes_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", default=os.environ.get("HERMES_HOME", "~/.hermes"))
    parser.add_argument("--hermes-dir", help="Hermes source root when it is not on Python's path")
    parser.add_argument(
        "--check", action="store_true", help="load the provider and test the local API"
    )
    parser.add_argument(
        "--start", action="store_true", help="start the installed sidecar with systemd"
    )
    args = parser.parse_args()
    home = Path(args.hermes_home).expanduser().resolve()
    os.environ["HERMES_HOME"] = str(home)
    if args.check:
        check(home, args.hermes_dir)
        return
    _hermes_source(home, args.hermes_dir)
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    unit_dir = Path("/etc/systemd/system") if is_root else Path.home() / ".config/systemd/user"
    _unit_path(home, unit_dir)
    key = (
        _parse_env(home / "meno-data/meno.env").get("MENO_OPENAI_API_KEY")
        or os.environ.get("MENO_INSTALL_SILICONFLOW_KEY")
        or getpass.getpass("SiliconFlow API key: ")
    )
    env_file = install(home, siliconflow_key=key)
    if args.start:
        _start(home, is_root=is_root, hermes_dir=args.hermes_dir)
    else:
        _write_unit(home, unit_dir)
    print(f"Installed Meno for one Hermes profile. Env file: {env_file}")
    if not args.start:
        scope = "system" if is_root else "user"
        prefix = "systemctl" if is_root else "systemctl --user"
        print(
            f"Start with: {prefix} daemon-reload && {prefix} enable --now meno.service ({scope} service)"
        )
    print("Restart Hermes to load the new profile config; the installer does not restart Hermes.")


if __name__ == "__main__":
    main()
