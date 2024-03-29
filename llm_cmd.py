import click
import llm
import subprocess
import os
from collections import deque

from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.shell import BashLexer


SYSTEM_PROMPT = """
Please return a valid bash command. I will supply some environment variables (# env), 
a shell history context (# history) and a user instruction (# now:) like:
# env
USER=someuser
SHELL=bash
PWD=/home/someuser/tmp
# history 3
ls -lah
cd ~/tmp
touch somefile
# now: undo last git commit

You should return only a valid bash command to be executed as a raw string, no example output,
no string delimiters, no wrapping it, no yapping, no markdown, no fenced code blocks, 
do not wrap output in ```.
What you return will be passed to Python subprocess.check_output() directly.

For example, if the user asks: ### now: undo last git commit

You return only: git reset --soft HEAD~1
""".strip()

ENV_VARS_CONTEXT = ["SHELL", "USER", "HOME", "PWD"]
N_SHELL_HISTORY_LINES = 3


@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("args", nargs=-1)
    @click.option("-m", "--model", default=None, help="Specify the model to use")
    @click.option("-s", "--system", help="Custom system prompt")
    @click.option("--key", help="API key to use")
    def cmd(args, model, system, key):
        """Generate and execute commands in your shell"""
        from llm.cli import get_default_model

        prompt = " ".join(args)
        prompt = generate_context_prompt(prompt)

        model_id = model or get_default_model()

        model_obj = llm.get_model(model_id)
        if model_obj.needs_key:
            model_obj.key = llm.get_key(key, model_obj.needs_key, model_obj.key_env_var)

        result = model_obj.prompt(prompt, system=system or SYSTEM_PROMPT)
        result = clean_codeblock(str(result))

        interactive_exec(result)


def tail(filename, n):
    """Returns the last n lines of a file, efficiently."""
    with open(filename, "r") as f:
        lines = deque(f, maxlen=n)
    return list(lines)


def shell_history_context(n=3, history_file=None):
    """
    Retrieves the last `n` lines from the specified shell history file.

    :param n: The number of lines to retrieve from the end of the history file. Defaults to 3.
    :param history_file: The path to the shell history file. Defaults to the value of $HISTFILE.
                         If $HISTFILE is not set, fallback to using ~/.bash_history.
    :return: A list containing the last `n` lines of the specified shell history file.

    Note: For accurate results, ensure the shell is configured to append commands to
    the history file immediately (e.g., via `shopt -s histappend`). Otherwise,
    the history file may not reflect the most recent commands until after executing
    `history -a` or exiting the shell.
    """
    if history_file is None:
        history_file = os.environ.get("HISTFILE", "~/.bash_history")

    history_file = os.path.expanduser(history_file)
    return [l.strip() for l in tail(history_file, n)]


def clean_codeblock(text):
    """
    Removes enclosing triple-backticks from a multi-line string.

    :param text: The input string.
    :return: The processed string without enclosing triple-backticks.
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("```") and lines[-1].startswith("```"):
        lines = lines[1:-1]
    return "\n".join(lines)


def env_var_context(env_vars):
    """
    Returns a list of environment variables like ENVVAR=value.

    :param env_vars: A list of environment variable names.
    :return: A list with each environment variable in the format ENVVAR=value
    """
    env_var_values = []
    for var in env_vars:
        if var in os.environ:
            env_var_values.append(f"{var}={os.environ[var].strip()}")
    return env_var_values


def generate_context_prompt(user_prompt):
    env_vars = "\n".join(env_var_context(ENV_VARS_CONTEXT))
    shell_history = "\n".join(shell_history_context(n=N_SHELL_HISTORY_LINES))
    prompt = f"# env\n{env_vars}\n# history {N_SHELL_HISTORY_LINES}\n{shell_history}\n# now: {user_prompt}".strip()
    return prompt


def interactive_exec(command):
    if "\n" in command:
        print("Multiline command - Meta-Enter or Esc Enter to execute")
        edited_command = prompt(
            "> ", default=command, lexer=PygmentsLexer(BashLexer), multiline=True
        )
    else:
        edited_command = prompt("> ", default=command, lexer=PygmentsLexer(BashLexer))
    try:
        output = subprocess.check_output(
            edited_command, shell=True, stderr=subprocess.STDOUT
        )
        print(output.decode())
    except subprocess.CalledProcessError as e:
        print(
            f"Command failed with error (exit status {e.returncode}): {e.output.decode()}"
        )
