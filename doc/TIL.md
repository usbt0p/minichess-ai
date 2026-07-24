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

## 020526

descargo una librería. no tengo autocompletado por lo tanto no tengo ni idea ni de las funciones que hay disponibles en la librería. mirando en .venv/.../site-packages/ parece que la librería es un binario de cpython . cómo puedo hacer para que tenga autocompletado?

Para solucionar este problema con las librerías compiladas en C (.so), he utilizado una herramienta estándar llamada stubgen (del paquete mypy).Esta herramienta inspecciona dinámicamente el binario de CPython importado y genera un archivo de "stubs" de tipado con extensión .pyi.

```
uv add mypy
.venv/bin/stubgen -m pyffish -o <your-output-dir>
```

Pero a veces visual studio y sus forks siguen dando por culo:

```
Cannot find module `...`
  Looked in these locations (from default config for project root marked by `/home/usbt0p/project/pyproject.toml`):
  Import root (inferred from project layout): "/home/usbt0p/project/src"
  Site package path queried from interpreter: ["/home/usbt0p/project/.venv/lib/python3.14/site-packages"] 
```

Esto es una cuestión de rutas. asegúrate de que el entorno virtual esté activado (ej: `source .venv/bin/activate`) y que la ruta sea la correcta.

## 050526

Existe [esto](https://docs.python.org/3/library/functools.html#functools.lru_cache), que es la ostia:

``` 
@functools.lru_cache(user_function)
@functools.lru_cache(maxsize=128, typed=False)

    Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls. It can save time when an expensive or I/O bound function is periodically called with the same arguments.

    The cache is threadsafe so that the wrapped function can be used in multiple threads. This means that the underlying data structure will remain coherent during concurrent updates.

    It is possible for the wrapped function to be called more than once if another thread makes an additional call before the initial call has been completed and cached.
```

## config pattern for configuration-heavy classes
use a dataclass as config. use __post_init__ to validate and normalize the config
your classes and funcs needing config accept it and take what they need from it, so they're agnostic to how the config was created. 
this makes the calls and arguments cleaner too, and with proper validation ensures your configs are always good

## profiling in pytorch, checking kernel usage and gpu usage
TODO write...

## optimizing attention with torch.compile, sdpa_backends, torch.autocast...
TODO write...

## 0307
Using `git info/exclude` lets you exclude files locally without using a `.gitignore`, in case you want some local stuff to stay in your repo untracked, but not commit them

## 1207

just learned about shell brace expansion... this is so useful!
```bash
scp -r lucas@1.2.3.4:~/tfg/minichess-ai/results/hypothesis3/tabula_rasa_seed_{1,32,999} results/hypothesis3/
```
also, wildcard works there too!

## 1307

find the latest edited files in a directory (recursively) using
```bash
find /path -printf '%T+ %p\n' | sort -r | head
```

## 2107

This is some bread and butter stuff. I already know this but I have to note it because i use it so much, but always forget the syntax and end up looking it up.

This is how to connect two computers trough ethernet to transfer files.

1. This is a temporal setup. It will only last until the next reboot. For both computers to see each other, they must be on the same subnet. For this just set them up so that the first two octets of their IPs are the same, and the mask is 24. For example, `192.168.100.x/24` and `192.168.100.y/24`. 

2. You can do this two ways:
 - `nmtui`, edit the cable connection and set IPv4 to manual, put the address you want in. Then save, deactivate the connection, and reactivate it. 
 - The command line one, with `sudo ip addr add 192.168.100.1/24 dev enp3s0`. This assumes your cable connection is named enp3s0, you can check it with `ip a`. You might also want to check if it's up if you are having problems.

3. Always ping to make sure the connection is working: `ping 192.168.100.y`, or the other way around.

4. Now you are connected. You can go a couple ways for file transfer / computer access:
  - Just ssh into the other computer with `ssh user@ip`
  - Use scp to bring or send files:
    - Bring: `scp user@ip:/path/to/file .`
    - Send: `scp /path/to/file user@ip:/path/to/destination`
    - For recursive, just `scp -r ...`
  - Use rsync for more complicated stuff, filtering, excluding files, etc... Works similar to scp but with more options. `rsync -ahP /path/to/file user@ip:/path/to/destination`. For filtering and stuff: 
    - exclude files: `rsync -ahP --exclude 'file1' --exclude 'file2' /path/to/file user@ip:/path/to/destination`
    - include only specific files: `rsync -ahP --include 'file1' --include 'file2' /path/to/file user@ip:/path/to/destination`
    - Wildcard and expansions: `rsync -ahP --include='*.py' --exclude='*' /path/to/file user@ip:/path/to/destination`. Note the order. 
  - Another goated one is the good ol' python server: `python3 -m http.server 8000` on the host computer, and then you can just `wget http://ip:8000/file` or just download by right-clicking, like any other website.