import datetime

from rich.console import Console
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown


def print_warning(msg, method_name = '', end_program=False):
    print_stuff(msg, "A error occured", method_name=method_name, end_program=end_program)

def print_message(msg, method_name = '', end_program=False):
    print_stuff(msg, "A message", method_name=method_name, end_program=end_program)

def print_stuff(msg, msg_type, method_name = '', end_program=False):
    print(f"\n\n------ {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{msg_type}:")
    print(f" - {msg}")
    if not method_name == '':
        print("The error occured in method:")
        print(f" - {method_name}")
    if end_program:
        print("The program will now exit")
    print("------\n\n")


def print_hyperparameters(parameters):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    
    table.add_column("Name")
    table.add_column("Default Value")
    table.add_column("Description")
    
    for p in parameters:
        table.add_row(
            p.arg_name,
            str(p.arg_default),
            str(p.arg_help)
        )
    
    console.print(table)

def print_md(md_path):
    console = Console()
    with open(md_path) as md:
        markdown = Markdown(md.read())
    console.print(markdown)

def introduce_biksup(parameters, parsed_keys, data_dict):
    print_md("text/start.md")
    print_hyperparameters(parameters)
    print_keys(parsed_keys)
    print_seeds(data_dict)
    introduce_start()

def introduce_start():
    print_md("text/execute.md")

def introduce_keys():
    print_md("text/introduce_keys.md")

def introduce_seeds():
    print_md('text/seed.md')

def print_keys(keys):
    introduce_keys()
    
    console = Console()
    keyTable = Table(show_header=True, header_style="bold magenta")
    
    keyTable.add_column("Key")
    
    
    for k in keys:
        keyTable.add_row(
            k.get_key()
        )
    
    
    console.print(keyTable)

def print_seeds(data_dict):
    console = Console()
    introduce_seeds()
    
    seedTable = Table(show_header=True, header_style="bold magenta")
    
    seedTable.add_column("Index")
    seedTable.add_column("Name")
    seedTable.add_column("Description")
    
    for i in range(len(data_dict.keys())):
        seedTable.add_row(
            str(i),
            str(list(data_dict.keys())[i]),
            str(list(data_dict.values())[i]),
        )
    
    console.print(seedTable)

def print_msg():
    pass