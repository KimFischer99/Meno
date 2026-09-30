from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "install-hermes.py"
SPEC = importlib.util.spec_from_file_location("install_hermes", SCRIPT)
installer = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(installer)


class HermesInstallerTests(unittest.TestCase):
    def test_install_reuses_profile_identity_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "profile % one"
            home.mkdir()
            (home / ".env").write_text("KEEP_ME=yes\n", encoding="utf-8")
            (home / "config.yaml").write_text("memory:\n  provider: builtin\n", encoding="utf-8")

            installer.install(home, siliconflow_key="s" * 32, run_commands=False)
            profile_env = installer._parse_env(home / ".env")
            user_id = profile_env["MENO_USER_ID"]
            shim = (home / "plugins/meno/__init__.py").read_text(encoding="utf-8")
            installer.install(home, siliconflow_key="ignored-existing-key", run_commands=False)

            self.assertEqual(installer._parse_env(home / ".env")["MENO_USER_ID"], user_id)
            self.assertEqual(installer._parse_env(home / ".env")["KEEP_ME"], "yes")
            self.assertIn("from meno.hermes_plugin import MenoMemoryProvider", shim)
            self.assertIn("ctx.register_memory_provider(MenoMemoryProvider())", shim)
            self.assertTrue(list(home.glob(".env.bak-*")))
            self.assertTrue(list(home.glob("config.yaml.bak-*")))

    def test_hermes_source_checks_real_memory_package_path(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "plugins/memory").mkdir(parents=True)
            (source / "plugins/memory/__init__.py").touch()
            with patch.object(installer.sys, "executable", str(source / ".venv/bin/python")):
                installer._hermes_source(Path(directory) / ".hermes", None)
            self.assertEqual(installer.sys.path[0], str(source))
            installer.sys.path.pop(0)

    def test_unit_quotes_paths_and_backs_up_existing_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / 'profile " 100%'
            unit_dir = Path(directory) / "systemd"
            unit_dir.mkdir()
            unit = unit_dir / "meno.service"
            unit.write_text(
                f"EnvironmentFile={installer._unit_quote(home / 'meno-data/meno.env')}\nold unit\n",
                encoding="utf-8",
            )
            installer._write_unit(home, unit_dir)
            content = unit.read_text(encoding="utf-8")
            self.assertIn(
                '"' + str(home / "meno-data/meno.env").replace("%", "%%").replace('"', '\\"') + '"',
                content,
            )
            self.assertTrue(list(unit_dir.glob("meno.service.bak-*")))
            unit.write_text("EnvironmentFile=/other/service.env\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different storage"):
                installer._write_unit(home, unit_dir)

    def test_check_uses_hermes_loader_and_authenticated_retrieve(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "hermes profile"
            home.mkdir()
            (home / "config.yaml").write_text("memory:\n  provider: meno\n", encoding="utf-8")
            (home / ".env").write_text(
                'MENO_API_TOKEN="secret-token-value"\nMENO_USER_ID="stable-profile"\n',
                encoding="utf-8",
            )
            source = Path(directory) / "hermes-source"
            (source / "plugins/memory").mkdir(parents=True)
            (source / "plugins/memory/__init__.py").touch()
            provider = types.SimpleNamespace(is_available=lambda: True)
            response = types.SimpleNamespace(raise_for_status=lambda: None, json=dict)
            client = MagicMock()
            client.__enter__.return_value = client
            client.get.return_value = response
            client.post.return_value = response
            module = types.SimpleNamespace(load_memory_provider=lambda name, **kwargs: provider)
            out = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=False),
                patch.object(installer.importlib, "import_module", return_value=module),
                patch.object(installer.httpx, "Client", return_value=client),
                contextlib.redirect_stdout(out),
            ):
                os.environ.pop("MENO_API_TOKEN", None)
                os.environ.pop("MENO_USER_ID", None)
                installer.check(home, str(source))
                self.assertEqual(os.environ["HERMES_HOME"], str(home.resolve()))

            args, kwargs = client.post.call_args
            self.assertEqual(args[0], "http://127.0.0.1:8765/v1/retrieve")
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-token-value")
            self.assertEqual(kwargs["json"]["user_id"], "stable-profile")
            self.assertNotIn("rendered_context", out.getvalue())

    def test_start_waits_and_checks_after_systemd_start(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "profile"
            unit_dir = Path(directory) / "systemd"
            with (
                patch.object(installer.subprocess, "run") as run,
                patch.object(installer, "_wait_ready") as wait_ready,
                patch.object(installer, "check") as check,
            ):
                installer._start(home, is_root=True, hermes_dir="/tmp/hermes", unit_dir=unit_dir)
            self.assertEqual(run.call_args_list[0].args[0], ["systemctl", "daemon-reload"])
            self.assertEqual(run.call_args_list[1].args[0], ["systemctl", "enable", "meno.service"])
            self.assertEqual(
                run.call_args_list[2].args[0], ["systemctl", "restart", "meno.service"]
            )
            wait_ready.assert_called_once_with(installer.PROFILE_KEYS["MENO_API_URL"])
            check.assert_called_once_with(home, "/tmp/hermes")


if __name__ == "__main__":
    unittest.main()
