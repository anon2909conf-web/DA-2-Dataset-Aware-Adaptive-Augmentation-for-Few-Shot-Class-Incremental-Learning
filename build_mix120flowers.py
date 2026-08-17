"""
Build a 120-class FSCIL dataset that mixes MiniImageNet and Oxford 102
Flowers in BOTH the base session and every incremental session. Mirrors the
structure of build_mix120dogs.py exactly, swapping Stanford Dogs' native
folder-per-breed layout for Flowers102's raw .mat annotation format.

Composition (defaults, all overridable via CLI):
  - Flowers total classes:        20   (--flowers-total)
  - Flowers classes put in base:   8   (--flowers-in-base)  -> base gets 52 mini + 8 flowers = 60
  - Flowers classes in incremental:12  (= flowers-total - flowers-in-base)
  - Sessions: 12 incremental sessions, 5-way each (--sessions / --way)
    -> each session gets 1 Flowers class + 4 MiniImageNet classes (5-way total)
  - Shot: 5 images/class for every incremental-session class (--shot)
  => 100 mini + 20 flowers = 120 total classes

Base session classes are "many-shot": ALL their training images are used,
regardless of whether the class came from MiniImageNet or Flowers.
Incremental classes are always few-shot (--shot images each).

Session-file numbering (IMPORTANT -- same convention as build_mix120.py /
build_mix120dogs.py / dataloader/data_utils.py):
    session_1.txt          = base session (documentation only)
    session_2.txt .. session_{sessions+1}.txt
                            = the `sessions` incremental sessions.
  i.e. with the default --sessions 12, files session_1.txt through
  session_13.txt are written (1 base + 12 incremental), and
  args.sessions for the 'mix120flowers' dataset must be set to 13 in
  dataloader/data_utils.py's set_up_datasets().

Result layout (created under --out-root, default dataset/mix120flowers):

  mix120flowers/
    images/<cid>/<cid>__<filename>            # symlinks to the original files
    split/train.csv                           # filename,cid (ALL train images of the 120 classes)
    split/test.csv                             # filename,cid (ALL test images of the 120 classes)
    index_list/session_1.txt ... session_13.txt   # session_1 = base (docs only), 2..13 = incremental (5-shot)
    class_order.txt                            # cid <TAB> global_label  (0..119, base classes first)

ASSUMPTIONS (adjust if your layout differs):
  - MiniImageNet: <root>/miniimagenet/images/<file>.jpg, split csvs at
    <root>/miniimagenet/split/{train,test}.csv with rows "filename,wnid"
    (or flat <root>/miniimagenet/{train,test}.csv -- both are checked).
  - Flowers102: <root>/flowers102_raw/jpg/image_00001.jpg ... and
    <root>/flowers102_raw/imagelabels.mat (the extracted 102flowers.tgz +
    imagelabels.mat, official Oxford VGG mirror -- see build_flowers102.py
    for the download commands). We pool all images per class and build our
    own train/test split (Flowers102's own train/val/test only gives 10
    train images/class, too few for many-shot base classes).

Usage:
  python build_mix120flowers.py --root /home/rb/FACL-main/dataset \
      --out-root /home/rb/FACL-main/dataset/mix120flowers \
      --flowers-total 20 --flowers-in-base 8 --sessions 12 --way 5 --shot 5 --seed 1
"""
import argparse
import csv
import os
import os.path as osp
import random
from collections import OrderedDict, defaultdict

import scipy.io as sio


def find_csv(root, name):
    """Looks for `name` directly under root, then under root/split, then root/splits.
    Raises a clear error listing what it checked if none exist."""
    candidates = [osp.join(root, name), osp.join(root, 'split', name), osp.join(root, 'splits', name)]
    for c in candidates:
        if osp.exists(c):
            return c
    raise FileNotFoundError(f"Couldn't find {name} in any of: {candidates}")


def read_split_csv(path):
    with open(path, 'r') as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    return [tuple(l.split(',')) for l in lines[1:]]


def unique_wnids_in_order(rows):
    seen = OrderedDict()
    for _, wnid in rows:
        seen[wnid] = True
    return list(seen.keys())


def group_by_wnid(rows):
    d = defaultdict(list)
    for fname, wnid in rows:
        d[wnid].append(fname)
    return d


def symlink_image(src, dst):
    os.makedirs(osp.dirname(dst), exist_ok=True)
    if not osp.exists(dst):
        os.symlink(osp.abspath(src), dst)


def load_flowers_raw(flowers_raw_root, test_per_class=15, seed=1):
    """Reads Flowers102's raw imagelabels.mat and pools all images per class
    (train+val+test), then builds our own train/test split, same convention
    as build_flowers102.py: reserve `test_per_class` for test, rest to train.

    Returns (train_rows, test_rows, img_dir) in the same (filename, cid) row
    format used elsewhere in this script -- img_dir is the 'jpg' folder, so
    images live at img_dir/<filename> (NOT nested by class on disk; cid is
    just our own class label, e.g. 'c001').
    """
    rng = random.Random(seed)
    jpg_dir = osp.join(flowers_raw_root, 'jpg')
    labels_mat = sio.loadmat(osp.join(flowers_raw_root, 'imagelabels.mat'))
    labels = labels_mat['labels'][0]

    by_class = defaultdict(list)
    for i, cls in enumerate(labels, start=1):
        fname = f'image_{i:05d}.jpg'
        cid = f'c{int(cls):03d}'
        by_class[cid].append(fname)

    train_rows, test_rows = [], []
    for cid, files in by_class.items():
        files = sorted(files)
        rng.shuffle(files)
        assert len(files) > test_per_class, \
            f'{cid} only has {len(files)} images, need > {test_per_class} (test_per_class)'
        for fname in files[:test_per_class]:
            test_rows.append((fname, cid))
        for fname in files[test_per_class:]:
            train_rows.append((fname, cid))
    return train_rows, test_rows, jpg_dir


def distribute(n_items, n_bins):
    """Split n_items as evenly as possible across n_bins, returns list of counts."""
    base, rem = divmod(n_items, n_bins)
    return [base + (1 if i < rem else 0) for i in range(n_bins)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./dataset', help='folder containing miniimagenet/ and flowers102_raw/')
    ap.add_argument('--flowers-raw-root', default=None,
                     help='defaults to <root>/flowers102_raw (containing jpg/ and imagelabels.mat)')
    ap.add_argument('--out-root', default=None, help='defaults to <root>/mix120flowers')
    ap.add_argument('--base-size', type=int, default=60)
    ap.add_argument('--flowers-total', type=int, default=20)
    ap.add_argument('--flowers-in-base', type=int, default=8)
    ap.add_argument('--sessions', type=int, default=12, help='number of INCREMENTAL sessions')
    ap.add_argument('--way', type=int, default=5)
    ap.add_argument('--shot', type=int, default=5)
    ap.add_argument('--flowers-test-per-class', type=int, default=15,
                     help='images held out per Flowers class for test')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = args.out_root or osp.join(args.root, 'mix120flowers')
    out_images = osp.join(out_root, 'images')
    out_split = osp.join(out_root, 'split')
    out_index = osp.join(out_root, 'index_list')
    os.makedirs(out_split, exist_ok=True)
    os.makedirs(out_index, exist_ok=True)

    novel_total = args.sessions * args.way
    flowers_novel_total = args.flowers_total - args.flowers_in_base
    mini_in_base = args.base_size - args.flowers_in_base
    mini_novel_total = novel_total - flowers_novel_total
    assert flowers_novel_total >= 0 and mini_in_base >= 0 and mini_novel_total >= 0, \
        "flowers-in-base / flowers-total / base-size / sessions*way don't add up -- check your numbers"
    assert flowers_novel_total <= args.sessions, \
        "need at most 1 flowers novel class per session with these defaults; " \
        "increase --sessions or lower --flowers-total/--flowers-in-base, or edit the per-session split below"

    # ---- load MiniImageNet ----
    mini_root = osp.join(args.root, 'miniimagenet')
    mini_img_dir = osp.join(mini_root, 'images')
    mini_train_csv = find_csv(mini_root, 'train.csv')
    mini_test_csv = find_csv(mini_root, 'test.csv')
    mini_train = read_split_csv(mini_train_csv)
    mini_test = read_split_csv(mini_test_csv)
    mini_wnids = unique_wnids_in_order(mini_train)
    mini_needed = mini_in_base + mini_novel_total
    assert len(mini_wnids) >= mini_needed, \
        f"MiniImageNet has {len(mini_wnids)} classes, need {mini_needed}"
    random.shuffle(mini_wnids)
    mini_base_wnids = mini_wnids[:mini_in_base]
    mini_novel_wnids = mini_wnids[mini_in_base: mini_in_base + mini_novel_total]

    # ---- load Flowers102 ----
    flowers_raw_root = args.flowers_raw_root or osp.join(args.root, 'flowers102_raw')
    flowers_train, flowers_test, flowers_img_dir = load_flowers_raw(
        flowers_raw_root, test_per_class=args.flowers_test_per_class, seed=args.seed)
    flowers_wnids_all = unique_wnids_in_order(flowers_train)
    assert len(flowers_wnids_all) >= args.flowers_total, \
        f"Flowers102 has {len(flowers_wnids_all)} classes, need {args.flowers_total}"
    flowers_chosen = random.sample(flowers_wnids_all, args.flowers_total)
    flowers_base_wnids = flowers_chosen[:args.flowers_in_base]
    flowers_novel_wnids = flowers_chosen[args.flowers_in_base:]

    print(f"Base session: {len(mini_base_wnids)} mini + {len(flowers_base_wnids)} flowers = "
          f"{len(mini_base_wnids) + len(flowers_base_wnids)} classes (many-shot)")
    print(f"Incremental: {args.sessions} sessions x {args.way}-way "
          f"({len(mini_novel_wnids)} mini + {len(flowers_novel_wnids)} flowers novel classes, "
          f"{args.shot}-shot each)")

    # ---- global class order / label map (base classes first, then novel in session order) ----
    base_wnids = mini_base_wnids + flowers_base_wnids
    random.shuffle(base_wnids)  # so mini/flowers aren't just block-concatenated in label space
    class_order = list(base_wnids)  # placeholder, novel appended after sessions are built below

    keep_wnids = set(mini_base_wnids + mini_novel_wnids + flowers_base_wnids + flowers_novel_wnids)
    mini_train_by_wnid = group_by_wnid([r for r in mini_train if r[1] in keep_wnids])
    flowers_train_by_wnid = group_by_wnid([r for r in flowers_train if r[1] in keep_wnids])
    mini_wnids_set = set(mini_wnids)

    # ---- build per-session wnid lists: 1 flowers (if available) + rest mini, per session ----
    flowers_per_session = distribute(len(flowers_novel_wnids), args.sessions)  # e.g. [1,1,...,1,0,0]
    mini_pool = list(mini_novel_wnids)
    flowers_pool = list(flowers_novel_wnids)
    sessions = []
    for s in range(args.sessions):
        n_flowers = flowers_per_session[s]
        sess_wnids = flowers_pool[:n_flowers] + mini_pool[:args.way - n_flowers]
        flowers_pool = flowers_pool[n_flowers:]
        mini_pool = mini_pool[args.way - n_flowers:]
        random.shuffle(sess_wnids)
        sessions.append(sess_wnids)
    assert not mini_pool and not flowers_pool, "leftover novel classes not assigned to any session"

    for sess_wnids in sessions:
        class_order.extend(sess_wnids)
    label_of = {wnid: i for i, wnid in enumerate(class_order)}
    with open(osp.join(out_root, 'class_order.txt'), 'w') as f:
        for wnid, lb in label_of.items():
            f.write(f"{wnid}\t{lb}\n")

    # ---- symlink images + write train/test csvs ----
    # NOTE: unlike MiniImageNet (already nested img_dir/<wnid>/<file> via
    # dogs-style layout) Flowers102's raw jpg/ folder is FLAT -- all images
    # sit directly in jpg_dir with no per-class subfolder, so unlike
    # build_mix120dogs.py the source path for flowers is img_dir/fname, not
    # img_dir/wnid/fname. is_mini already distinguishes mini vs flowers, and
    # both mini and flowers happen to use a flat source layout here.
    def process(rows, img_dir, writer, nested):
        for fname, wnid in rows:
            if wnid not in keep_wnids:
                continue
            src = osp.join(img_dir, wnid, fname) if nested else osp.join(img_dir, fname)
            dst_name = f"{wnid}__{fname}"  # avoid filename collisions across datasets
            dst = osp.join(out_images, wnid, dst_name)
            symlink_image(src, dst)
            writer.writerow([dst_name, wnid])

    with open(osp.join(out_split, 'train.csv'), 'w', newline='') as f_train, \
         open(osp.join(out_split, 'test.csv'), 'w', newline='') as f_test:
        w_train, w_test = csv.writer(f_train), csv.writer(f_test)
        w_train.writerow(['filename', 'wnid'])
        w_test.writerow(['filename', 'wnid'])
        process(mini_train, mini_img_dir, w_train, nested=False)
        process(flowers_train, flowers_img_dir, w_train, nested=False)
        process(mini_test, mini_img_dir, w_test, nested=False)
        process(flowers_test, flowers_img_dir, w_test, nested=False)

    # ---- session_1.txt: base-session training images (documentation only) ----
    base_lines = []
    for wnid in base_wnids:
        pool = mini_train_by_wnid.get(wnid) or flowers_train_by_wnid.get(wnid)
        assert pool is not None, f"base class {wnid} has no train images"
        for fname in pool:
            dst_name = f"{wnid}__{fname}"
            base_lines.append(f"{wnid}/{dst_name}")
    with open(osp.join(out_index, 'session_1.txt'), 'w') as f:
        f.write('\n'.join(base_lines) + '\n')

    # ---- per-session few-shot index files, numbered session_2 .. session_{sessions+1} ----
    for offset, wnids in enumerate(sessions, start=2):
        lines = []
        for wnid in wnids:
            pool = mini_train_by_wnid.get(wnid) or flowers_train_by_wnid.get(wnid)
            assert pool is not None and len(pool) >= args.shot, \
                f"class {wnid} has only {0 if pool is None else len(pool)} train images, need {args.shot}"
            for fname in random.sample(pool, args.shot):
                dst_name = f"{wnid}__{fname}"
                lines.append(f"{wnid}/{dst_name}")
        with open(osp.join(out_index, f'session_{offset}.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')

    n_incremental = len(sessions)
    print(f"\nDone. Total classes: {len(class_order)} "
          f"(base={len(base_wnids)}, incremental={sum(len(s) for s in sessions)})")
    print(f"Wrote session_1.txt (base, docs only) and "
          f"session_2.txt .. session_{1 + n_incremental}.txt ({n_incremental} incremental sessions).")
    print(f"=> set args.sessions = {1 + n_incremental} for 'mix120flowers' in dataloader/data_utils.py")
    print(f"Output written to: {out_root}")


if __name__ == '__main__':
    main()