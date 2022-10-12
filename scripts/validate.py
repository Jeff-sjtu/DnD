import logging

import torch
import torch.nn as nn
from dnd.datasets.loaders import get_test_loaders
from dnd.evaluator import Evaluator
from dnd.models import builder
from dnd.args import cfg, logger, args


def main():
    print('...Evaluating on test set...')
    torch.cuda.set_device(args.gpu)

    evaluator = Evaluator

    # Model Initialize
    m = builder.build_model(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    try:
        print(f'Loading model from {args.ckpt}...')
        m.load_state_dict(torch.load(args.ckpt, map_location='cpu')['gen_state_dict'], strict=True)
    except RuntimeError:
        m = nn.parallel.DataParallel(m)
        m.load_state_dict(torch.load(args.ckpt, map_location='cpu')['gen_state_dict'], strict=True)
        m = m.module

    m.cuda(args.gpu)

    test_loader = get_test_loaders(args, cfg)

    streamhandler = logging.StreamHandler()

    logger.setLevel(logging.INFO)
    logger.addHandler(streamhandler)

    if isinstance(test_loader, list):
        for testset in test_loader:
            print('Evaluating on ======', testset.dataset.dataset_name.upper(), '======')
            evaluator(
                args,
                cfg,
                logger,
                model=m,
                test_loader=testset,
            ).run_new()
    else:
        evaluator(
            args,
            cfg,
            logger,
            model=m,
            test_loader=test_loader,
        ).run()


if __name__ == '__main__':
    main()
