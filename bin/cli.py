import typer

main = typer.Typer()


@main.command()
def train():
    print("This is where the training code will go")


@main.command()
def infer():
    print("this is where the inference code will go")
