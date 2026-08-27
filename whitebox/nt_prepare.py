"""M7 rung 2 — fetch Nucleotide Transformer downstream tasks (revised,
chromosome-held-out, real negatives) and mirror them into the
~/.genomic_benchmarks/<task>/{train,test}/<label>/ layout that
whitebox.dna_train already consumes.

python3 -m whitebox.nt_prepare --task splice_sites_donors
Priority tasks (M7-DNA.md): splice_sites_donors (NEGATIVE CONTROL),
H3K4me3, enhancers.
"""

import argparse
import pathlib

from datasets import load_dataset

HF = 'InstaDeepAI/nucleotide_transformer_downstream_tasks_revised'
OUT = pathlib.Path.home() / '.genomic_benchmarks'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default=None)
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.list:
        ds = load_dataset(HF, split='test')
        names = sorted(set(ds['task']))
        for n in names:
            print(n, flush=True)
        return

    assert args.task
    for split in ('train', 'test'):
        ds = load_dataset(HF, split=split)
        ds = ds.filter(lambda r: r['task'] == args.task)
        assert len(ds) > 0, f'no rows for task {args.task}'
        root = OUT / f'nt_{args.task}' / split
        counts = {}
        for row in ds:
            lab = str(row['label'])
            d = root / lab
            d.mkdir(parents=True, exist_ok=True)
            i = counts.get(lab, 0)
            (d / f'{i}.txt').write_text(row['sequence'])
            counts[lab] = i + 1
        print(f'nt_{args.task} {split}: {counts} '
              f'len~{len(ds[0]["sequence"])}', flush=True)


if __name__ == '__main__':
    main()
