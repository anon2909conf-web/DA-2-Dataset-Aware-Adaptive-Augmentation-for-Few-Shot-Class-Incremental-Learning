"""
Build a 120-class FSCIL dataset that mixes MiniImageNet and CUB200 in BOTH
the base session and every incremental session.

Composition (defaults, all overridable via CLI):
  - CUB total classes:        20   (--cub-total)
  - CUB classes put in base:   8   (--cub-in-base)      -> base gets 52 mini + 8 cub = 60
  - CUB classes in incremental:12   (= cub-total - cub-in-base)
  - Sessions: 12 incremental sessions, 5-way each (--sessions / --way)
    -> each session gets 1 CUB class + 4 MiniImageNet classes (5-way total)
  - Shot: 5 images/class for every incremental-session class (--shot)

Base session classes are "many-shot": ALL their training images are used
(same as the original MiniImageNet base session), regardless of whether the
class came from MiniImageNet or CUB200. Incremental classes are always
few-shot (--shot images each), as in standard FSCIL.

Session-file numbering (IMPORTANT):
  Matches the convention used by dataloader/data_utils.py and Mix120:
    session_1.txt         = base session (documentation only -- Mix120's
                             base session actually loads via `index` +
                             SelectfromClasses, not this file, but it is
                             still written here for symmetry/debugging).
    session_2.txt .. session_{sessions+1}.txt
                            = the `sessions` incremental sessions.
  i.e. with the default --sessions 12, files session_1.txt through
  session_13.txt are written (1 base + 12 incremental), and
  args.sessions for the 'mix120' dataset must be set to 13 in
  dataloader/data_utils.py's set_up_datasets().

Result layout (created under --out-root, default dataset/mix120):

  mix120/
    images/<wnid>/<wnid>__<filename>          # symlinks to the original files
    split/train.csv                           # filename,wnid (ALL train images of the 120 classes)
    split/test.csv                             # filename,wnid (ALL test images of the 120 classes)
    index_list/session_1.txt ... session_13.txt   # session_1 = base (docs only), 2..13 = incremental (5-shot)
    class_order.txt                            # wnid <TAB> global_label  (0..119, base classes first)

ASSUMPTIONS (adjust the two path pairs below if your layout differs):
  - MiniImageNet: <root>/miniimagenet/images/<file>.jpg, split csvs at
    <root>/miniimagenet/split/{train,test}.csv with rows "filename,wnid".
  - CUB200: <root>/CUB_200_2011/images/<wnid>/<file>.jpg (raw distribution:
    images.txt, classes.txt, image_class_labels.txt, train_test_split.txt).

Usage:
  python build_mix120.py --root /home/rb/FACL-main/dataset \
      --out-root /home/rb/FACL-main/dataset/mix120 \
      --cub-total 20 --cub-in-base 8 --sessions 12 --way 5 --shot 5 --seed 1
"""
import argparse
import csv
import os
import os.path as osp
import random
from collections import OrderedDict, defaultdict

from PIL import Image


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


def crop_and_save_image(src, dst, bbox):
    """bbox = (x, y, w, h) in pixels, as given by CUB's bounding_boxes.txt."""
    os.makedirs(osp.dirname(dst), exist_ok=True)
    if osp.exists(dst):
        return
    x, y, w, h = bbox
    with Image.open(src) as im:
        im = im.convert('RGB').crop((x, y, x + w, y + h))
        im.save(dst, quality=95)


def load_cub_raw(cub_root):
    """Parses the raw CUB_200_2011 distribution (images.txt, classes.txt,
    image_class_labels.txt, train_test_split.txt) into the same
    (filename, wnid) row format used elsewhere in this script.
    wnid = class folder name (e.g. '001.Black_footed_Albatross'),
    matching images/<wnid>/<filename> on disk.
    """
    img_dir = osp.join(cub_root, 'images')

    id_to_name = {}
    with open(osp.join(cub_root, 'images.txt')) as f:
        for line in f:
            img_id, name = line.strip().split(' ', 1)
            id_to_name[img_id] = name  # "001.Black_footed_Albatross/Black_Footed_..._0046.jpg"

    id_to_classid = {}
    with open(osp.join(cub_root, 'image_class_labels.txt')) as f:
        for line in f:
            img_id, class_id = line.strip().split()
            id_to_classid[img_id] = class_id

    classid_to_name = {}
    with open(osp.join(cub_root, 'classes.txt')) as f:
        for line in f:
            class_id, class_name = line.strip().split(' ', 1)
            classid_to_name[class_id] = class_name

    id_to_split = {}
    with open(osp.join(cub_root, 'train_test_split.txt')) as f:
        for line in f:
            img_id, is_train = line.strip().split()
            id_to_split[img_id] = int(is_train)

    bbox_by_id = {}  # img_id -> (x, y, w, h)
    bbox_path = osp.join(cub_root, 'bounding_boxes.txt')
    if osp.exists(bbox_path):
        with open(bbox_path) as f:
            for line in f:
                img_id, x, y, w, h = line.strip().split()
                bbox_by_id[img_id] = tuple(float(v) for v in (x, y, w, h))

    train_rows, test_rows = [], []
    bboxes = {}  # (wnid, fname) -> (x, y, w, h), only used if --cub-crop-bbox is passed
    for img_id, name in id_to_name.items():
        wnid = classid_to_name[id_to_classid[img_id]]
        fname = osp.basename(name)
        row = (fname, wnid)
        (train_rows if id_to_split[img_id] == 1 else test_rows).append(row)
        if img_id in bbox_by_id:
            bboxes[(wnid, fname)] = bbox_by_id[img_id]

    return train_rows, test_rows, img_dir, bboxes


def distribute(n_items, n_bins):
    """Split n_items as evenly as possible across n_bins, returns list of counts."""
    base, rem = divmod(n_items, n_bins)
    return [base + (1 if i < rem else 0) for i in range(n_bins)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./dataset', help='folder containing miniimagenet/ and CUB_200_2011/')
    ap.add_argument('--out-root', default=None, help='defaults to <root>/mix120')
    ap.add_argument('--base-size', type=int, default=60)
    ap.add_argument('--cub-total', type=int, default=20)
    ap.add_argument('--cub-in-base', type=int, default=8)
    ap.add_argument('--sessions', type=int, default=12, help='number of INCREMENTAL sessions')
    ap.add_argument('--way', type=int, default=5)
    ap.add_argument('--shot', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--cub-crop-bbox', action='store_true',
                     help='crop CUB images to their bounding_boxes.txt box before saving '
                          '(writes real cropped files instead of symlinks for CUB only; '
                          'MiniImageNet images are always symlinked)')
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = args.out_root or osp.join(args.root, 'mix120')
    out_images = osp.join(out_root, 'images')
    out_split = osp.join(out_root, 'split')
    out_index = osp.join(out_root, 'index_list')
    os.makedirs(out_split, exist_ok=True)
    os.makedirs(out_index, exist_ok=True)

    novel_total = args.sessions * args.way
    cub_novel_total = args.cub_total - args.cub_in_base
    mini_in_base = args.base_size - args.cub_in_base
    mini_novel_total = novel_total - cub_novel_total
    assert cub_novel_total >= 0 and mini_in_base >= 0 and mini_novel_total >= 0, \
        "cub-in-base / cub-total / base-size / sessions*way don't add up -- check your numbers"
    assert cub_novel_total <= args.sessions, \
        "need at most 1 cub novel class per session with these defaults; " \
        "increase --sessions or lower --cub-total/--cub-in-base, or edit the per-session split below"

    # ---- load MiniImageNet ----
    # supports either layout: <root>/miniimagenet/{train,test}.csv (flat) or
    # <root>/miniimagenet/split/{train,test}.csv (nested) -- whichever exists.
    # Ignores stray '._images' AppleDouble folders / 'splits' leftovers.
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

    # ---- load CUB200 (raw CUB_200_2011 distribution: images.txt, classes.txt,
    #      image_class_labels.txt, train_test_split.txt -- bounding_boxes.txt used
    #      only if --cub-crop-bbox is set; parts/ and attributes/ are unused here) ----
    cub_root = osp.join(args.root, 'CUB_200_2011')
    cub_train, cub_test, cub_img_dir, cub_bboxes = load_cub_raw(cub_root)
    cub_wnids_all = unique_wnids_in_order(cub_train)
    cub_chosen = random.sample(cub_wnids_all, args.cub_total)
    cub_base_wnids = cub_chosen[:args.cub_in_base]
    cub_novel_wnids = cub_chosen[args.cub_in_base:]

    print(f"Base session: {len(mini_base_wnids)} mini + {len(cub_base_wnids)} cub = "
          f"{len(mini_base_wnids) + len(cub_base_wnids)} classes (many-shot)")
    print(f"Incremental: {args.sessions} sessions x {args.way}-way "
          f"({len(mini_novel_wnids)} mini + {len(cub_novel_wnids)} cub novel classes, "
          f"{args.shot}-shot each)")

    # ---- global class order / label map (base classes first, then novel in session order) ----
    base_wnids = mini_base_wnids + cub_base_wnids
    random.shuffle(base_wnids)  # so mini/cub aren't just block-concatenated in label space
    class_order = list(base_wnids)  # placeholder, novel appended after sessions are built below

    keep_wnids = set(mini_base_wnids + mini_novel_wnids + cub_base_wnids + cub_novel_wnids)
    mini_train_by_wnid = group_by_wnid([r for r in mini_train if r[1] in keep_wnids])
    cub_train_by_wnid = group_by_wnid([r for r in cub_train if r[1] in keep_wnids])
    mini_wnids_set = set(mini_wnids)

    # ---- build per-session wnid lists: 1 cub (if available) + rest mini, per session ----
    cub_per_session = distribute(len(cub_novel_wnids), args.sessions)  # e.g. [1,1,...,1,0,0]
    mini_pool = list(mini_novel_wnids)
    cub_pool = list(cub_novel_wnids)
    sessions = []
    for s in range(args.sessions):
        n_cub = cub_per_session[s]
        sess_wnids = cub_pool[:n_cub] + mini_pool[:args.way - n_cub]
        cub_pool = cub_pool[n_cub:]
        mini_pool = mini_pool[args.way - n_cub:]
        random.shuffle(sess_wnids)
        sessions.append(sess_wnids)
    assert not mini_pool and not cub_pool, "leftover novel classes not assigned to any session"

    for sess_wnids in sessions:
        class_order.extend(sess_wnids)
    label_of = {wnid: i for i, wnid in enumerate(class_order)}
    with open(osp.join(out_root, 'class_order.txt'), 'w') as f:
        for wnid, lb in label_of.items():
            f.write(f"{wnid}\t{lb}\n")

    # ---- symlink images + write train/test csvs ----
    def process(rows, img_dir, writer, bboxes=None):
        for fname, wnid in rows:
            if wnid not in keep_wnids:
                continue
            is_mini = wnid in mini_wnids_set
            src = osp.join(img_dir, fname) if is_mini else osp.join(img_dir, wnid, fname)
            dst_name = f"{wnid}__{fname}"  # avoid filename collisions across datasets
            dst = osp.join(out_images, wnid, dst_name)
            if (not is_mini) and args.cub_crop_bbox and bboxes and (wnid, fname) in bboxes:
                crop_and_save_image(src, dst, bboxes[(wnid, fname)])
            else:
                symlink_image(src, dst)
            writer.writerow([dst_name, wnid])

    with open(osp.join(out_split, 'train.csv'), 'w', newline='') as f_train, \
         open(osp.join(out_split, 'test.csv'), 'w', newline='') as f_test:
        w_train, w_test = csv.writer(f_train), csv.writer(f_test)
        w_train.writerow(['filename', 'wnid'])
        w_test.writerow(['filename', 'wnid'])
        process(mini_train, mini_img_dir, w_train)
        process(cub_train, cub_img_dir, w_train, bboxes=cub_bboxes)
        process(mini_test, mini_img_dir, w_test)
        process(cub_test, cub_img_dir, w_test, bboxes=cub_bboxes)

    # ---- session_1.txt: base-session training images (documentation only --
    #      Mix120's base session actually loads via `index` + SelectfromClasses,
    #      NOT this file, but we still write it for symmetry/debugging) ----
    base_lines = []
    for wnid in base_wnids:
        pool = mini_train_by_wnid.get(wnid) or cub_train_by_wnid.get(wnid)
        assert pool is not None, f"base class {wnid} has no train images"
        for fname in pool:
            dst_name = f"{wnid}__{fname}"
            base_lines.append(f"{wnid}/{dst_name}")
    with open(osp.join(out_index, 'session_1.txt'), 'w') as f:
        f.write('\n'.join(base_lines) + '\n')

    # ---- per-session few-shot index files, numbered session_2 .. session_{sessions+1} ----
    # (session_1 is reserved for the base session above; the trainer's
    # get_new_dataloader looks up "session_" + str(session_index + 1) + ".txt"
    # where session_index runs 1..args.sessions-1 for the incremental loop,
    # i.e. it expects files session_2.txt .. session_{args.sessions}.txt with
    # args.sessions == 1 + <number of incremental sessions>.)
    for offset, wnids in enumerate(sessions, start=2):
        lines = []
        for wnid in wnids:
            pool = mini_train_by_wnid.get(wnid) or cub_train_by_wnid.get(wnid)
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
    print(f"=> set args.sessions = {1 + n_incremental} for 'mix120' in dataloader/data_utils.py")
    print(f"Output written to: {out_root}")


if __name__ == '__main__':
    main()