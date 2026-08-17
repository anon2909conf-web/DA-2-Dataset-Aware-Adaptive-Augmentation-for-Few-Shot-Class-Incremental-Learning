"""
build_timgnet200.py
====================
Builds an FSCIL session split for Tiny-ImageNet-200, mirroring CUB200's
exact session config (base_class=100, num_classes=200, way=10, shot=5,
sessions=11) so it's a structurally apples-to-apples COARSE counterpart to
your fine-grained datasets (cub200/mix200/dogs120/flowers102): 200 broad,
visually-distinct object categories (goldfish, school bus, teddy bear,
sports car, ...) instead of one narrow super-category split into subtypes.

--- Getting the raw data (run on your machine) ---
    mkdir -p /path/to/dataset/timgnet200_raw && cd /path/to/dataset/timgnet200_raw
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip
    # -> tiny-imagenet-200/train/<wnid>/images/<wnid>_<n>.JPEG  (500 imgs/class)
    #    tiny-imagenet-200/wnids.txt                            (200 class ids)

We only use the `train/` split (500 images/class, already organized one
folder per class) and pool+re-split it ourselves, same convention as
mix120/mix200/flowers102/dogs120: reserve --test-per-class images for test,
rest go to train (base classes get all of it = many-shot, incremental
classes get exactly `shot`). The dataset's own val/ and test/ folders are
ignored (val requires parsing val_annotations.txt and test is unlabeled --
neither is needed since we're building our own split).

--- What this script does ---
  1. Reads wnids.txt for the 200 class ids.
  2. Globs each class's train/<wnid>/images/*.JPEG.
  3. Reserves --test-per-class images per class for test (default 50), rest
     for train (~450 remain/class -- plenty for a many-shot base session).
  4. Randomly selects 100 classes for the base session (many-shot) and
     splits the remaining 100 into 10 incremental sessions of 10-way 5-shot.
  5. Symlinks images into the same layout used by your other built datasets:
       images/<wnid>/<wnid>__<filename>
       split/train.csv, split/test.csv       (filename,wnid)
       index_list/session_1.txt ... session_10.txt
       class_order.txt                       (wnid<TAB>global_label, 0..199)
"""

import argparse
import glob
import os
import os.path as osp
import random


def symlink_image(src_dir, wnid, fname, out_images_dir):
    """Symlinks <src_dir>/<fname> -> <out_images_dir>/<wnid>/<wnid>__<fname>.
    Returns the new filename (with wnid__ prefix) used inside the output dataset."""
    src = osp.join(src_dir, fname)
    dst_dir = osp.join(out_images_dir, wnid)
    os.makedirs(dst_dir, exist_ok=True)
    new_fname = f'{wnid}__{fname}'
    dst = osp.join(dst_dir, new_fname)
    if not osp.exists(dst):
        if not osp.exists(src):
            raise FileNotFoundError(f'Missing source image: {src}')
        os.symlink(osp.abspath(src), dst)
    return new_fname


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-root', required=True,
                     help='Folder containing wnids.txt and train/<wnid>/images/*.JPEG '
                          '(i.e. the extracted tiny-imagenet-200.zip)')
    ap.add_argument('--out-root', required=True,
                     help='Output directory, e.g. <dataroot>/timgnet200')

    ap.add_argument('--num-classes', type=int, default=200)
    ap.add_argument('--base-size', type=int, default=100)
    ap.add_argument('--sessions', type=int, default=10)
    ap.add_argument('--way', type=int, default=10)
    ap.add_argument('--shot', type=int, default=5)
    ap.add_argument('--test-per-class', type=int, default=50,
                     help='Images held out per class for the test split')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    assert args.base_size + args.sessions * args.way == args.num_classes, \
        f'base-size ({args.base_size}) + sessions*way ' \
        f'({args.sessions}*{args.way}={args.sessions * args.way}) must equal ' \
        f'num-classes ({args.num_classes})'

    random.seed(args.seed)

    wnids_file = osp.join(args.raw_root, 'wnids.txt')
    with open(wnids_file, 'r') as f:
        all_wnids = sorted(x.strip() for x in f if x.strip())
    assert len(all_wnids) == args.num_classes, \
        f'Found {len(all_wnids)} classes in {wnids_file}, expected {args.num_classes}'

    train_root = osp.join(args.raw_root, 'train')

    train_by_class = {}
    test_by_class = {}
    class_src_dir = {}
    for wnid in all_wnids:
        src_dir = osp.join(train_root, wnid, 'images')
        class_src_dir[wnid] = src_dir
        files = sorted(osp.basename(p) for p in glob.glob(osp.join(src_dir, '*.JPEG')))
        random.shuffle(files)
        assert len(files) > args.test_per_class, \
            f'{wnid} only has {len(files)} images, need > {args.test_per_class} (test-per-class)'
        test_by_class[wnid] = files[:args.test_per_class]
        train_by_class[wnid] = files[args.test_per_class:]

    # ---- assign classes to base / incremental sessions ----
    shuffled = all_wnids[:]
    random.shuffle(shuffled)
    base_classes = shuffled[:args.base_size]
    incr_pool = shuffled[args.base_size:]
    random.shuffle(base_classes)

    sessions = []
    idx = 0
    for s in range(args.sessions):
        sess_classes = incr_pool[idx:idx + args.way]
        idx += args.way
        sessions.append(sess_classes)
    assert idx == len(incr_pool)

    for sess_classes in sessions:
        for wnid in sess_classes:
            assert len(train_by_class[wnid]) >= args.shot, \
                f'{wnid} has only {len(train_by_class[wnid])} train images, need {args.shot}'

    # ---- assign the fixed global labels: base first, then session order ----
    label_of = {}
    label = 0
    for wnid in base_classes:
        label_of[wnid] = label
        label += 1
    for sess_classes in sessions:
        for wnid in sess_classes:
            label_of[wnid] = label
            label += 1
    assert label == args.num_classes

    # ---- write output ----
    images_out = osp.join(args.out_root, 'images')
    split_out = osp.join(args.out_root, 'split')
    index_out = osp.join(args.out_root, 'index_list')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(split_out, exist_ok=True)
    os.makedirs(index_out, exist_ok=True)

    train_csv_rows = []
    test_csv_rows = []

    # base classes: many-shot -- use every train image reserved for them
    for wnid in base_classes:
        src_dir = class_src_dir[wnid]
        for fname in train_by_class[wnid]:
            new_fname = symlink_image(src_dir, wnid, fname, images_out)
            train_csv_rows.append((new_fname, wnid))
        for fname in test_by_class[wnid]:
            new_fname = symlink_image(src_dir, wnid, fname, images_out)
            test_csv_rows.append((new_fname, wnid))

    # incremental classes: `shot` images for train, held-out test images for eval
    session_index_lines = [[] for _ in range(args.sessions)]
    for s, sess_classes in enumerate(sessions):
        for wnid in sess_classes:
            src_dir = class_src_dir[wnid]
            chosen = random.sample(train_by_class[wnid], args.shot)
            for fname in chosen:
                new_fname = symlink_image(src_dir, wnid, fname, images_out)
                train_csv_rows.append((new_fname, wnid))
                session_index_lines[s].append(f'{wnid}/{new_fname}')
            for fname in test_by_class[wnid]:
                new_fname = symlink_image(src_dir, wnid, fname, images_out)
                test_csv_rows.append((new_fname, wnid))

    with open(osp.join(split_out, 'train.csv'), 'w') as f:
        f.write('filename,label\n')
        for fname, wnid in train_csv_rows:
            f.write(f'{fname},{wnid}\n')
    with open(osp.join(split_out, 'test.csv'), 'w') as f:
        f.write('filename,label\n')
        for fname, wnid in test_csv_rows:
            f.write(f'{fname},{wnid}\n')

    # NOTE: data_utils.py's get_new_dataloader computes the path as
    # "session_{session+1}.txt" where `session` is the loop counter starting
    # at 1 for the first incremental round (0 is the base session, and its
    # slot -- session_1.txt -- is left unused). So the first incremental
    # session's file must be session_2.txt, not session_1.txt.
    for s in range(args.sessions):
        with open(osp.join(index_out, f'session_{s + 2}.txt'), 'w') as f:
            f.write('\n'.join(session_index_lines[s]) + '\n')

    with open(osp.join(args.out_root, 'class_order.txt'), 'w') as f:
        for wnid in base_classes + [w for sess in sessions for w in sess]:
            f.write(f'{wnid}\t{label_of[wnid]}\n')

    print('Done.')
    print(f'  base session: {len(base_classes)} classes, many-shot')
    for s, sess_classes in enumerate(sessions):
        print(f'  session {s + 1}: {len(sess_classes)} classes, {args.shot}-shot')
    print(f'  total classes: {label} -> written to {osp.join(args.out_root, "class_order.txt")}')


if __name__ == '__main__':
    main()