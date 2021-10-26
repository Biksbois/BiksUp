import datetime

import sys
from rich.console import Console
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown
from biksLog import get_logger
from icecream import ic
from rich import print
from rich.panel import Panel
from rich.padding import Padding

from global_items import TRUE_LIST
from global_items import DEFAULT_INPUTKEY

log = get_logger()

def get_default(opt, p):
    if p.arg_name == '--dataset':
        return opt.dataset
    elif p.arg_name == '--batchSize':
        return opt.batchSize
    elif p.arg_name == '--hiddenSize':
        return opt.hiddenSize
    elif p.arg_name == '--epoch':
        return opt.epoch
    elif p.arg_name == '--lr':
        return opt.lr
    elif p.arg_name == '--lr_dc':
        return opt.lr_dc
    elif p.arg_name == '--lr_dc_step':
        return opt.lr_dc_step
    elif p.arg_name == '--l2':
        return opt.l2
    elif p.arg_name == '--step':
        return opt.step
    elif p.arg_name == '--patience':
        return opt.patience
    elif p.arg_name == '--nonhybrid':
        return opt.nonhybrid
    elif p.arg_name == '--validation':
        return opt.validation
    elif p.arg_name == '--valid_portion':
        return opt.valid_portion
    elif p.arg_name == '--keys':
        return opt.keys
    elif p.arg_name == '--runall':
        return opt.runall
    elif p.arg_name == '--runlast':
        return opt.runlast
    elif p.arg_name == '--iterations':
        return opt.iterations
    else:
        log.exception(f"The parameter {p.arg_name} has not been implemented correctly when getting default parameters.")
        return p.arg_default

def print_hyperparameters(parameters, opt):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    
    table.add_column("Name")
    table.add_column("Value")
    table.add_column("Description")
    
    for p in parameters:
        default = get_default(opt, p)
        
        table.add_row(
            p.arg_name,
            str(default),
            str(p.arg_help)
        )
    
    console.print(table)

def print_md(md_path):
    console = Console()
    with open(md_path) as md:
        markdown = Markdown(md.read())
    console.print(markdown)

def invalid_arguments(opt):
    count = 0
    
    if opt.runall in TRUE_LIST :
        count += 1
    if opt.runlast in TRUE_LIST:
        count += 1
    if len(opt.keys) > 0 and not opt.keys == DEFAULT_INPUTKEY:
        count += 1
    return False if count <= 1 else True

def print_exit_keys():
    log.exception(f"Incorrect parameters give. Please only initialize on of the following three parameters: '--runall', '--runlast' and '--keys'")
    sys.exit()

def check_if_valid(opt):
    if invalid_arguments(opt):
        print_exit_keys()

def introduce_biksup(parameters, parsed_keys, data_dict, opt):
    print_md("text/start.md")
    print_hyperparameters(parameters, opt)
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
    
    key_str = "[bright_yellow]\n"
    
    for i in range(len(keys)):
        key_str += f"Index: {str(i).rjust(3, '0')} | Key: {keys[i].get_key()}\n"
    
    print(Padding(Panel(key_str, title="All input keys contains", subtitle="Let the biksing being!"), (4,4)))

def print_seeds(data_dict):
    introduce_seeds()
    
    seed_str = "\n"
    
    for i in range(len(data_dict.keys())):
        seed_str += f"[bright_yellow]Index: {str(i).rjust(3, '0')} | Name: {str(list(data_dict.keys())[i]).ljust(10, ' ')} | Desc: {str(list(data_dict.values())[i])}\n"
    
    print(Padding(Panel(seed_str, title="All keys contains", subtitle="Let the biksing being!"), (4,4)))
