import os

import typer

from letta.cli.cli import server
from letta.cli.cli_messages import app as messages_app

app = typer.Typer(pretty_exceptions_enable=False)

# Register server as both the default command and as a subcommand
app.command(name="server")(server)

# Register messages subcommand
app.add_typer(messages_app, name="messages")


# Also make server the default when no command is specified
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        # If no subcommand is specified, run the server
        server()
