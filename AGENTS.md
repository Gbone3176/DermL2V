# AGENTS.md

## Path Handling

- This repository uses the `local_info/` directory to store frequently used local file paths and machine-specific path references.
- When updating, replacing, or adding file paths in scripts, configs, or commands, check `local_info/` first and prefer those recorded paths over introducing new hardcoded paths.
- If a required path is missing from `local_info/`, add or update the relevant entry there before reusing that path broadly in the project.
- Treat `local_info/` as the default source of truth for local path lookup in this repository.
