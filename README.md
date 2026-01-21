# ⚙️ Environment Setup 

The project uses **Conda** for environment management.
To automatically create and configure the `srt-anom` environment:

## 🪟 Windows systems
Run:
```bash
scripts\win\setup_env.bat
```
This script:

- creates or updates the `srt-anom` Conda environment from `env.yml`;

- installs all required dependencies;

- registers the project for direct execution.
  
## ▶️ Running the project

After the initial setup, the pipeline can be executed with:

```bash
scripts\win\run.bat
```

There is no need to manually activate the Conda environment:
`run.bat` automatically runs the project in the correct environment.

## 🐧🍎 Linux / macOS systems

Use the equivalent scripts located in the `scripts/unix` directory:

```bash
scripts/unix/setup_env.sh
scripts/unix/run.sh
```