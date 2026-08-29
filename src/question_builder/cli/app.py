import typer

app = typer.Typer(
    name="qbuilder",
    help="Build traceable, precision-first question records from DOCX inputs.",
    no_args_is_help=True,
)


@app.callback()
def root() -> None:
    """Question Builder V1 command root."""
