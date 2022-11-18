

def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items() :
        if k != "weights_prior_params":
            info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")
    return info
