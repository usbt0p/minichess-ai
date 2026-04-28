## TIL
### 250426
- export environment variables in fish: `set --export PYTHONPATH /home/usbt0p/TFG/`. To make this permanent, put `set -xg PYTHONPATH /path/to/test/folder $PYTHONPATH` at the end of `~/.config/fish/config.fish`
- to make vscode auto export vars in `.env` files, create an env file add 
    ```
    {
    "python.envFile": "${workspaceFolder}/.env"
    }
    ```
    to the `.vscode/settings.json` file.