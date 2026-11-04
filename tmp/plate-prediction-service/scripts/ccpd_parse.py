"""Parse CCPD filenames into ground-truth plate strings.

CCPD filename fields (dash-separated), per PaddleOCR CCPD doc and
detectRecog/CCPD README:
  0: plate-area ratio
  1: horizontal_vertical tilt
  2: left-top & right-bottom box (x&y_x&y)
  3: four plate vertices (x&y_x&y_x&y_x&y)
  4: character indices, underscore-separated:
     [0]=province, [1]=letter, [2:]=alphanumerics
  5: brightness
  6: blur
'O' is the placeholder for "no character" in CCPD index tables.
"""
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
             "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
             "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
             'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q',
       'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def parse_ccpd_filename(filename):
    """Return (plate_str, char_indices, vertices, brightness, blur)."""
    stem = filename.rsplit('.', 1)[0]
    parts = stem.split('-')
    if len(parts) < 7:
        raise ValueError(f"unexpected CCPD filename format: {filename}")
    idx = [int(x) for x in parts[4].split('_')]
    chars = [PROVINCES[idx[0]], ALPHABETS[idx[1]]]
    chars += [ADS[i] for i in idx[2:]]
    # Drop 'O' placeholders (no-character slots); keep real characters only.
    plate = ''.join(c for c in chars if c != 'O')
    verts = []
    for tok in parts[3].split('_'):
        x, y = tok.split('&')
        verts.append((int(x), int(y)))
    return plate, idx, verts, parts[5], parts[6]


if __name__ == '__main__':
    # Sanity check against the documented example: 皖AY339S
    fn = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    plate, _, _, _, _ = parse_ccpd_filename(fn)
    assert plate == "皖AY339S", plate
    print("parser OK:", plate)
