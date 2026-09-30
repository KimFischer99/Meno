# Install Meno for one Hermes profile

The installer puts the sidecar and its virtual environment under the selected
Hermes profile's `meno-data/`, adds a profile-local `meno` provider, and sets
`memory.provider: meno` in that profile's `config.yaml`. It does not edit Hermes
source or restart Hermes.

## Install

Run from a Linux host with Python, systemd, and the Meno checkout available:

```bash
/path/to/hermes/venv/bin/python deploy/install-hermes.py --hermes-home ~/.hermes --start
```

The installer prompts for the SiliconFlow API key with `getpass`. To supply it
non-interactively, set `MENO_INSTALL_SILICONFLOW_KEY`; the installer does not
print the key. It creates a separate venv in `<hermes-home>/meno-data/.venv`,
installs the checkout and its declared dependencies, then runs the Meno
migration. Existing config, profile `.env`, plugin shim, and systemd unit files
are backed up before edits. Existing environment values are retained; a
conflicting Meno setting stops installation for manual resolution. The stable
`MENO_USER_ID` is stored in the profile `.env` and reused on later installs.
The installed profile uses the deployed state-materialization, context,
preference, reflection, and evidence-selection settings, with a 10,000-vector
cap. Semantic routing and layered retrieval remain disabled.
If an existing `meno.service` points to another environment/database, the installer
stops before changing the profile. Use the [manual upgrade guide](README.md) to
preserve that service's canonical state.

The installer selects a system service when run as root and a user service
otherwise. Starting waits for the local health endpoint, loads the provider
through Hermes, then makes an authenticated
retrieve request. It reports success only after all three checks pass. Without
`--start`, start the generated service using the command printed by the
installer. Restart Hermes after installation to load the new profile config.

## Diagnose

Run the provider and authenticated API check with the same profile and Hermes
source tree used by the Hermes process:

```bash
/path/to/hermes/venv/bin/python deploy/install-hermes.py --hermes-home ~/.hermes --hermes-dir /path/to/hermes-agent --check
```

`--hermes-dir` is optional when `plugins/memory/__init__.py` is already
discoverable from the Python executable's parent paths. The check sets
`HERMES_HOME` to the selected profile, verifies `memory.provider`, asks Hermes'
real `load_memory_provider('meno', register_skills=False)` loader to load the
provider, then checks `/health/ready` and an authenticated `/v1/retrieve`. It
does not print retrieved memory text.

For foreground startup, use the sidecar interpreter and environment created by
the installer:

```bash
MENO_ENV_FILE="$HERMES_HOME/meno-data/meno.env" \
MENO_PYTHON="$HERMES_HOME/meno-data/.venv/bin/python" deploy/start.sh
```

Set `HERMES_HOME` to your profile directory first. A non-root systemd user service
needs a user manager; enable lingering if it should run after logout.

For service issues, inspect the matching system or user unit:

```bash
systemctl status meno.service
systemctl --user status meno.service
```

Use the command matching how the installer was run. A failed health check means
the sidecar is not ready; a provider load error usually means the Hermes source
directory or profile plugin path is wrong; an authenticated retrieve error
usually means the sidecar and profile tokens differ. Do not paste either `.env`
file into diagnostics because they contain credentials.

To disable the integration, set `memory.provider` back to the prior provider in
`config.yaml`, then restart Hermes. The sidecar data remains in
`<hermes-home>/meno-data/`.
