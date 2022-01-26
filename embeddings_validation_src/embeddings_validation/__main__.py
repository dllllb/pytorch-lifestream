import os

import luigi
import argparse
import datetime

from embeddings_validation import ReportCollect
from embeddings_validation.config import Config
from embeddings_validation.tasks.fold_splitter import FoldSplitter


def read_extra_conf(file_name, conf_extra):
    conf = Config.read_file(file_name, conf_extra)
    name, ext = os.path.splitext(file_name)
    tmp_file_name = f'{name}_{datetime.datetime.now().timestamp():.0f}{ext}'
    conf.save_tmp_copy(tmp_file_name)
    return tmp_file_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--conf', required=True)
    parser.add_argument('--split_only', action='store_true', required=False)
    parser.add_argument('--total_cpu_count', type=int, required=True)
    parser.add_argument('--local_scheduler', action='store_true', required=False)
    parser.add_argument('--conf_extra', required=False)
    parser.add_argument('--log_level', default='INFO')
    args = parser.parse_args()

    conf_file_name = args.conf
    if args.conf_extra is not None:
        conf_file_name = read_extra_conf(conf_file_name, args.conf_extra)

    if args.split_only:
        task = FoldSplitter(
            conf=conf_file_name,
        )
    else:
        task = ReportCollect(
            conf=conf_file_name,
            total_cpu_count=args.total_cpu_count,
        )
    luigi.build([task], workers=args.workers, local_scheduler=args.local_scheduler, log_level=args.log_level)
