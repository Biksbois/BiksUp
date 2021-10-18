# parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')

# parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
# parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
# parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
# parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
# parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
# parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
# parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
# parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
# parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
# parser.add_argument('--runall', type=bool, default=False, help="Run all permutations of key combinations")

# parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
# parser.add_argument('--validation', action='store_true', help='validation')

# parser.add_argument('--l', nargs='+', default=[''], help="List of boolean keys of what permutation to execute, '1' = True, '0'=False, '-' = True and False. Example: ['1110-00']")




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
            # TODO: End program
            print("this should not happen. parameter_class.py")


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
    parameter_list.append(parameterObj(arg_name='--l', arg_nargs='+', arg_default=[''], arg_help="List of boolean keys of what permutation to execute, '1' = True, '0'=False, '-' = True and False. Example: ['1110-00']"))
    parameter_list.append(parameterObj(arg_name='--runall', arg_type=bool, arg_default=False, arg_help="Run all permutations of key combinations"))

    return parameter_list


