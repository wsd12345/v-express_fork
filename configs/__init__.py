from .lip_sync import CFG_LIP_SYNC


def update_config(cfg, args_cfg):
    cfg.defrost()

    cfg.merge_from_file(args_cfg)  # update cfg
    # cfg.merge_from_list(args.opts)

    cfg.freeze()
