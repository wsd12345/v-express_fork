def get_cfg(cfg):
    if isinstance(cfg, dict):
        return cfg

    import yaml  # for torch hub
    with open(cfg, encoding='ascii', errors='ignore') as f:
        data = yaml.safe_load(f)  # model dict
    return data
