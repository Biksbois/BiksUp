from biksLog import get_logger
import sys

log = get_logger()

class parameterObj():
    def __init__(self, arg_name=None, arg_default='', arg_help=None, arg_type=None, arg_nargs=None, arg_action=None) -> None:
        self.arg_name = arg_name
        self.arg_help = arg_help
        
        self.arg_type = arg_type
        self.arg_default = arg_default

        self.arg_action = arg_action
        self.arg_nargs = arg_nargs
    
    def add_new_argument(self, parser):
        if not self.arg_type == None and not self.arg_default == '' and self.arg_action == None and self.arg_nargs == None:
            parser.add_argument(self.arg_name, default=self.arg_default, help=self.arg_help)
        elif self.arg_type == None and self.arg_default == '' and not self.arg_action == None and self.arg_nargs == None:
            parser.add_argument(self.arg_name, action=self.arg_action, help=self.arg_help)
        elif self.arg_type == None and not self.arg_default == '' and self.arg_action == None and not self.arg_nargs == None:
            parser.add_argument(self.arg_name, nargs=self.arg_nargs, default=self.arg_default, help=self.arg_help)
        else:
            log.exception("This is an invalid type of argument, and is not handled. parameter_class.py")
            sys.exit()


def get_parameters():
    parameter_list = []
    
    parameter_list.append(parameterObj(arg_name='--dataset', arg_default='sample', arg_type=str, arg_help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample'))
    parameter_list.append(parameterObj(arg_name='--batchSize', arg_type=int, arg_default=100, arg_help='input batch size'))
    parameter_list.append(parameterObj(arg_name='--hiddenSize', arg_type=int, arg_default=100, arg_help='hidden state size'))
    parameter_list.append(parameterObj(arg_name='--epoch', arg_type=int, arg_default=30, arg_help='the number of epochs to train for'))
    parameter_list.append(parameterObj(arg_name='--lr', arg_type=float, arg_default=0.001, arg_help='learning rate'))  # [0.001, 0.0005, 0.0001]
    parameter_list.append(parameterObj(arg_name='--lr_dc', arg_type=float, arg_default=0.1, arg_help='learning rate decay rate'))
    parameter_list.append(parameterObj(arg_name='--lr_dc_step', arg_type=int, arg_default=3, arg_help='the number of steps after which the learning rate decay'))
    parameter_list.append(parameterObj(arg_name='--l2', arg_type=float, arg_default=1e-5, arg_help='l2 penalty'))  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parameter_list.append(parameterObj(arg_name='--step', arg_type=int, arg_default=1, arg_help='gnn propogation steps'))
    parameter_list.append(parameterObj(arg_name='--patience', arg_type=int, arg_default=10, arg_help='the number of epoch to wait before early stop '))
    parameter_list.append(parameterObj(arg_name='--nonhybrid', arg_action='store_true', arg_help='only use the global preference to predict'))
    parameter_list.append(parameterObj(arg_name='--validation', arg_action='store_true', arg_help='validation'))
    parameter_list.append(parameterObj(arg_name='--valid_portion', arg_type=float, arg_default=0.1, arg_help='split the portion of training set as validation set'))
    parameter_list.append(parameterObj(arg_name='--keys', arg_nargs='+', arg_default=['1'], arg_help="List of boolean keys of what permutation to execute, '1' = True, '0'=False, '_' = True and False. Example: ['1110_00']"))
    parameter_list.append(parameterObj(arg_name='--runall', arg_type=bool, arg_default=False, arg_help="Run all permutations of key combinations"))
    parameter_list.append(parameterObj(arg_name='--runlast', arg_type=bool, arg_default=False, arg_help="Run the last executed variation of the --keys argument"))

    return parameter_list


