from prettytable import PrettyTable

def logging(msg, name_study):
    fpath = name_study + '/log_' + name_study + ".txt" 
    with open( fpath, "a" ) as fw:
        fw.write("%s\n" % msg)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    return total_params, table