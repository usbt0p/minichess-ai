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

### 290426
Un heredoc es una forma de pasarle datos a un programa como si se le pasaras por entrada estándar.
En Bash, cuando pones comillas al identificador del heredoc (<< 'EOF'), le estás diciendo al shell que no expanda las variables que hay dentro. Así que para strings literales es mejor usar comillas, y para cosas como esta no:

```bash
input_file=$1
output_file=$2

./variant-nnue-tools/src/stockfish << EOF
setoption name UCI_Variant value gardner
gather_statistics all input_file $input_file output_file_name $output_file
quit
EOF
```