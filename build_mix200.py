"""
build_mix200.py
================
Builds a 200-class FSCIL dataset that mirrors CUB200's own session layout
(base_class=100, way=10, shot=5, sessions=11) but is composed mostly of
CUB200 classes with a MiniImageNet minority mixed into both the base
session and every incremental session.

Default composition (all overridable via CLI):
  - 200 total classes = 160 CUB200 + 40 MiniImageNet
  - Base session (100 classes)      = 80 CUB + 20 MiniImageNet, many-shot
  - 10 incremental sessions (10-way) = 8 CUB + 2 MiniImageNet each, 5-shot

Session-file numbering (IMPORTANT):
  Matches the convention used by dataloader/data_utils.py and Mix200:
    session_1.txt          = base session (written here for documentation /
                              debugging symmetry -- Mix200's base session
                              actually loads via `index` + SelectfromClasses,
                              not this file).
    session_2.txt .. session_{sessions+1}.txt
                             = the `sessions` incremental sessions.
  i.e. with the default --sessions 10, files session_1.txt through
  session_11.txt are written (1 base + 10 incremental), matching
  Mix200's own docstring ("index_list/session_1.txt ... session_11.txt")
  and args.sessions = 11 for the 'mix200' dataset in
  dataloader/data_utils.py's set_up_datasets().

Output layout under `--out-root` (default: <root>/mix200/):
  images/<wnid>/<wnid>__<filename>          (symlinks into the source datasets)
  split/train.csv, split/test.csv           (filename,wnid  -- filename already
                                              carries the wnid__ prefix)
  index_list/session_1.txt ... session_11.txt
                                             session_1 = base classes' full
                                             train pool (docs only);
                                             session_2..session_11 = the
                                             `shot` images sampled for each
                                             incremental session.
  class_order.txt                           (wnid<TAB>global_label, 0..199)

Assumed source layout:
  <mini-root>/split/train.csv, test.csv       (rows: filename,wnid -- MoCo/FSCIL
                                                convention, same as build_mix120.py)
  <mini-root>/images/<wnid>/<filename>
  <cub-root>/CUB_200_2011/images.txt            ("<id> <class_folder>/<filename>")
  <cub-root>/CUB_200_2011/image_class_labels.txt ("<id> <class_id 1..200>")
  <cub-root>/CUB_200_2011/train_test_split.txt   ("<id> <1=train,0=test>")
  <cub-root>/CUB_200_2011/images/<class_folder>/<filename>

CUB200 is read directly from these native Caltech-format files (the same
ones the original CUB200 Dataset class uses) -- no csv conversion needed.
The "wnid" used internally for CUB classes is just the class folder name
from images.txt, e.g. '001.Black_footed_Albatross'.

Usage:
  python build_mix200.py --root /home/rb/FACL-main/dataset \
      --out-root /home/rb/FACL-main/dataset/mix200 \
      --mini-total 40 --cub-total 160 --seed 1
"""

import argparse
import os
import os.path as osp
import random
from collections import defaultdict


def read_split_csv(path):
    """Returns list of (filename, wnid) rows, skipping the header."""
    with open(path, 'r') as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    rows = []
    for line in lines[1:]:
        fname, wnid = line.split(',')
        rows.append((fname, wnid))
    return rows


def group_by_class(rows):
    """wnid -> list[filename]"""
    out = defaultdict(list)
    for fname, wnid in rows:
        out[wnid].append(fname)
    return out


def read_cub_native(cub_root):
    """Reads CUB200's raw Caltech-format annotation files directly (the same
    files the original CUB200 Dataset class uses) -- no csv conversion needed.

    Returns (train_by_class, test_by_class, images_dir):
      train_by_class / test_by_class : wnid -> list[filename]
        where wnid is the class folder name from images.txt
        (e.g. '001.Black_footed_Albatross') and filename is the basename
        inside that folder.
      images_dir : <cub-root>/CUB_200_2011/images, i.e. where
        images_dir/<wnid>/<filename> resolves to an actual file.
    """
    base = osp.join(cub_root, 'CUB_200_2011')
    image_file = osp.join(base, 'images.txt')
    split_file = osp.join(base, 'train_test_split.txt')
    images_dir = osp.join(base, 'images')

    def read_lines(path):
        with open(path, 'r') as f:
            return [x.strip() for x in f.readlines() if x.strip()]

    id2image = {}
    for line in read_lines(image_file):
        idx, relpath = line.split(' ', 1)
        id2image[int(idx)] = relpath

    id2train = {}
    for line in read_lines(split_file):
        idx, flag = line.split(' ')
        id2train[int(idx)] = flag

    train_by_class = defaultdict(list)
    test_by_class = defaultdict(list)
    for idx, relpath in id2image.items():
        wnid, fname = relpath.split('/', 1)
        bucket = train_by_class if id2train[idx] == '1' else test_by_class
        bucket[wnid].append(fname)

    return train_by_class, test_by_class, images_dir


def symlink_image(src_dir, wnid, fname, out_images_dir, is_mini):
    """Symlinks the source image -> <out_images_dir>/<wnid>/<wnid>__<fname>.
    MiniImageNet images are stored FLAT (src_dir/<fname>, no per-class
    subfolder); CUB200 images are nested (src_dir/<wnid>/<fname>). Returns
    the new filename (with wnid__ prefix) used inside mix200."""
    src = osp.join(src_dir, fname) if is_mini else osp.join(src_dir, wnid, fname)
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
    ap.add_argument('--root', required=True,
                     help='Directory containing miniimagenet/ and cub200/ subfolders')
    ap.add_argument('--mini-root', default=None,
                     help='Override path to MiniImageNet dataset (default: <root>/miniimagenet)')
    ap.add_argument('--cub-root', default=None,
                     help='Path to the directory CONTAINING CUB_200_2011/ '
                          '(default: <root> itself, i.e. expects <root>/CUB_200_2011/images.txt)')
    ap.add_argument('--out-root', default=None,
                     help='Output directory (default: <root>/mix200)')

    ap.add_argument('--mini-total', type=int, default=40,
                     help='Total MiniImageNet classes to use (minority)')
    ap.add_argument('--cub-total', type=int, default=160,
                     help='Total CUB200 classes to use (majority)')

    ap.add_argument('--base-size', type=int, default=100,
                     help='Number of classes in the base session')
    ap.add_argument('--mini-in-base', type=int, default=20,
                     help='MiniImageNet classes placed in the base session')
    ap.add_argument('--cub-in-base', type=int, default=80,
                     help='CUB200 classes placed in the base session')

    ap.add_argument('--sessions', type=int, default=10,
                     help='Number of INCREMENTAL sessions')
    ap.add_argument('--way', type=int, default=10,
                     help='Classes per incremental session')
    ap.add_argument('--mini-per-session', type=int, default=2,
                     help='MiniImageNet classes per incremental session')
    ap.add_argument('--cub-per-session', type=int, default=8,
                     help='CUB200 classes per incremental session')

    ap.add_argument('--shot', type=int, default=5,
                     help='Images per class sampled for each incremental session')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)

    mini_root = args.mini_root or osp.join(args.root, 'miniimagenet')
    cub_root = args.cub_root or args.root
    out_root = args.out_root or osp.join(args.root, 'mix200')

    # ---- sanity checks on the composition numbers before touching any files ----
    assert args.mini_in_base + args.cub_in_base == args.base_size, \
        f'mini-in-base ({args.mini_in_base}) + cub-in-base ({args.cub_in_base}) ' \
        f'must equal base-size ({args.base_size})'
    assert args.mini_per_session + args.cub_per_session == args.way, \
        f'mini-per-session ({args.mini_per_session}) + cub-per-session ' \
        f'({args.cub_per_session}) must equal way ({args.way})'
    mini_incremental_total = args.mini_per_session * args.sessions
    cub_incremental_total = args.cub_per_session * args.sessions
    assert args.mini_in_base + mini_incremental_total == args.mini_total, \
        f'mini-in-base + mini-per-session*sessions ' \
        f'({args.mini_in_base} + {mini_incremental_total}) must equal ' \
        f'mini-total ({args.mini_total})'
    assert args.cub_in_base + cub_incremental_total == args.cub_total, \
        f'cub-in-base + cub-per-session*sessions ' \
        f'({args.cub_in_base} + {cub_incremental_total}) must equal ' \
        f'cub-total ({args.cub_total})'
    # NOTE: unlike build_mix120.py, this does NOT assume a minority class
    # contributes at most 1 slot per session -- both mini-per-session and
    # cub-per-session can be any value as long as they sum to `way`.

    print(f'MiniImageNet root: {mini_root}')
    print(f'CUB200 root:       {cub_root}')
    print(f'Output root:       {out_root}')

    # ---- load source splits ----
    # MiniImageNet: csv convention (filename,wnid)
    mini_train_rows = read_split_csv(osp.join(mini_root, 'split', 'train.csv'))
    mini_test_rows = read_split_csv(osp.join(mini_root, 'split', 'test.csv'))
    mini_train_by_class = group_by_class(mini_train_rows)
    mini_test_by_class = group_by_class(mini_test_rows)
    mini_images_dir = osp.join(mini_root, 'images')

    # CUB200: native Caltech-format files (images.txt / image_class_labels.txt /
    # train_test_split.txt) -- no csv conversion required
    cub_train_by_class, cub_test_by_class, cub_images_dir = read_cub_native(cub_root)

    mini_all_wnids = sorted(mini_train_by_class.keys())
    cub_all_wnids = sorted(cub_train_by_class.keys())
    assert len(mini_all_wnids) >= args.mini_total, \
        f'Requested {args.mini_total} MiniImageNet classes but only ' \
        f'{len(mini_all_wnids)} available'
    assert len(cub_all_wnids) >= args.cub_total, \
        f'Requested {args.cub_total} CUB200 classes but only ' \
        f'{len(cub_all_wnids)} available'

    # ---- pick which classes participate ----
    mini_selected = random.sample(mini_all_wnids, args.mini_total)
    cub_selected = random.sample(cub_all_wnids, args.cub_total)
    random.shuffle(mini_selected)
    random.shuffle(cub_selected)

    mini_base = mini_selected[:args.mini_in_base]
    mini_incr_pool = mini_selected[args.mini_in_base:]
    cub_base = cub_selected[:args.cub_in_base]
    cub_incr_pool = cub_selected[args.cub_in_base:]

    # base session: mix the two pools together, shuffled
    base_classes = mini_base + cub_base
    random.shuffle(base_classes)

    # incremental sessions: draw mini-per-session / cub-per-session from each
    # pool per session, in order (pools were already shuffled), then shuffle
    # the composition within each session so mini/cub aren't fixed slots
    sessions = []
    mi, ci = 0, 0
    for s in range(args.sessions):
        m_chunk = mini_incr_pool[mi:mi + args.mini_per_session]
        c_chunk = cub_incr_pool[ci:ci + args.cub_per_session]
        mi += args.mini_per_session
        ci += args.cub_per_session
        sess_classes = m_chunk + c_chunk
        random.shuffle(sess_classes)
        sessions.append(sess_classes)
    assert mi == len(mini_incr_pool) and ci == len(cub_incr_pool)

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
    assert label == args.mini_total + args.cub_total

    def source_for(wnid):
        return ('mini', mini_images_dir, mini_train_by_class, mini_test_by_class, True) \
            if wnid in mini_selected else \
            ('cub', cub_images_dir, cub_train_by_class, cub_test_by_class, False)

    # ---- build output dirs ----
    images_out = osp.join(out_root, 'images')
    split_out = osp.join(out_root, 'split')
    index_out = osp.join(out_root, 'index_list')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(split_out, exist_ok=True)
    os.makedirs(index_out, exist_ok=True)

    train_csv_rows = []  # (new_fname, wnid)
    test_csv_rows = []

    # base classes: many-shot -- symlink every train image available.
    # Also collect the relative paths for session_1.txt (docs only, see below).
    base_session_lines = []
    for wnid in base_classes:
        _, images_dir, train_by_class, test_by_class, is_mini = source_for(wnid)
        for fname in train_by_class[wnid]:
            new_fname = symlink_image(images_dir, wnid, fname, images_out, is_mini)
            train_csv_rows.append((new_fname, wnid))
            base_session_lines.append(f'{wnid}/{new_fname}')
        for fname in test_by_class.get(wnid, []):
            new_fname = symlink_image(images_dir, wnid, fname, images_out, is_mini)
            test_csv_rows.append((new_fname, wnid))

    # incremental classes: `shot` images for train, full test set for eval
    session_index_lines = [[] for _ in range(args.sessions)]
    for s, sess_classes in enumerate(sessions):
        for wnid in sess_classes:
            _, images_dir, train_by_class, test_by_class, is_mini = source_for(wnid)
            avail = train_by_class[wnid]
            assert len(avail) >= args.shot, \
                f'{wnid} has only {len(avail)} train images, need {args.shot}'
            chosen = random.sample(avail, args.shot)
            for fname in chosen:
                new_fname = symlink_image(images_dir, wnid, fname, images_out, is_mini)
                train_csv_rows.append((new_fname, wnid))
                session_index_lines[s].append(f'{wnid}/{new_fname}')
            for fname in test_by_class.get(wnid, []):
                new_fname = symlink_image(images_dir, wnid, fname, images_out, is_mini)
                test_csv_rows.append((new_fname, wnid))

    # ---- write split csvs ----
    with open(osp.join(split_out, 'train.csv'), 'w') as f:
        f.write('filename,label\n')
        for fname, wnid in train_csv_rows:
            f.write(f'{fname},{wnid}\n')
    with open(osp.join(split_out, 'test.csv'), 'w') as f:
        f.write('filename,label\n')
        for fname, wnid in test_csv_rows:
            f.write(f'{fname},{wnid}\n')

    # ---- session_1.txt: base-session training images (documentation only --
    #      Mix200's base session actually loads via `index` + SelectfromClasses,
    #      NOT this file, but we still write it for symmetry/debugging, matching
    #      Mix200's own docstring which lists session_1.txt as part of the
    #      expected index_list/ layout) ----
    with open(osp.join(index_out, 'session_1.txt'), 'w') as f:
        f.write('\n'.join(base_session_lines) + '\n')

    # ---- write incremental session index files, numbered session_2 .. session_{sessions+1} ----
    # (session_1 is reserved for the base session above; the trainer's
    # get_new_dataloader looks up "session_" + str(session_index + 1) + ".txt"
    # where session_index runs 1..args.sessions-1 for the incremental loop,
    # i.e. it expects files session_2.txt .. session_{args.sessions}.txt with
    # args.sessions == 1 + <number of incremental sessions> == 11 here.)
    for s in range(args.sessions):
        with open(osp.join(index_out, f'session_{s + 2}.txt'), 'w') as f:
            f.write('\n'.join(session_index_lines[s]) + '\n')

    # ---- write class_order.txt ----
    with open(osp.join(out_root, 'class_order.txt'), 'w') as f:
        for wnid in base_classes + [w for sess in sessions for w in sess]:
            f.write(f'{wnid}\t{label_of[wnid]}\n')

    print('Done.')
    print(f'  base session (session_1.txt, docs only): {len(base_classes)} classes '
          f'({len(mini_base)} mini + {len(cub_base)} cub), many-shot')
    for s, sess_classes in enumerate(sessions):
        n_mini = sum(1 for w in sess_classes if w in mini_selected)
        n_cub = len(sess_classes) - n_mini
        print(f'  session_{s + 2}.txt (incremental #{s + 1}): {len(sess_classes)} classes '
              f'({n_mini} mini + {n_cub} cub), {args.shot}-shot')
    print(f'  total classes: {label} -> written to {osp.join(out_root, "class_order.txt")}')
    print(f'=> set args.sessions = {1 + args.sessions} for \'mix200\' in dataloader/data_utils.py')


if __name__ == '__main__':
    main()