# compatible setup script for Bash, Zsh, and Fish shell

# this MUST be run on the project root for correct path export
# so for example: cd /home/lucas/tfg/minichess-ai && source setup.sh 
export PYTHONPATH="." 

# Conditionally activate the virtual environment based on the shell
[ -n "$BASH_VERSION" ] && source .venv/bin/activate
[ -n "$ZSH_VERSION" ] && source .venv/bin/activate
[ -n "$FISH_VERSION" ] && source .venv/bin/activate.fish

true