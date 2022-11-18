

def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")
    return info
