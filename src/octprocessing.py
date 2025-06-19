import numpy as np


def get_id(manualLayer):
    idx = []
    for i in range(0, 61):
        temp = manualLayer[:, :, i]
        if np.sum(temp) != 0:
            idx.append(i)
    return idx


def get_unlabelled(manualLayer):
    idx = []
    for i in range(0, 61):
        temp = manualLayer[:, :, i]
        if np.sum(temp) == 0:
            idx.append(i)
    return idx


def get_img_segmentation(mat):
    fluid_class = 9

    manualLayer = np.array(mat["manualLayers1"], dtype=np.uint16)
    manualFluid = np.array(mat["manualFluid1"], dtype=np.uint16)
    img = np.array(mat["images"], dtype=np.uint8)
    valid_idx = get_id(manualLayer)

    manualFluid = manualFluid[:, :, valid_idx]
    manualLayer = manualLayer[:, :, valid_idx]

    seg = np.zeros((496, 768, 11))
    seg[manualFluid > 0] = fluid_class

    max_col = -100
    min_col = 900
    for b_scan_idx in range(0, 11):
        for col in range(768):
            cur_col = manualLayer[:, col, b_scan_idx]
            if np.sum(cur_col) == 0:
                continue

            max_col = max(max_col, col)
            min_col = min(min_col, col)

            labels_idx = cur_col.tolist()
            last_st = None
            last_ed = None
            for label, (st, ed) in enumerate(zip([0] + labels_idx, labels_idx + [-1])):

                if st == 0 and ed == 0:
                    st = last_ed
                    print("val", seg[st, col, b_scan_idx])
                    while seg[st, col, b_scan_idx] == fluid_class:
                        st += 1

                    while seg[st, col, b_scan_idx] != fluid_class:
                        seg[st, col, b_scan_idx] = label
                        st += 1
                        if st >= 496:
                            break
                    continue
                if ed == 0:
                    ed = st + 1
                    while seg[ed, col, b_scan_idx] != fluid_class:
                        ed += 1

                if st == 0 and label != 0:
                    st = ed - 1
                    while seg[st, col, b_scan_idx] != fluid_class:
                        st -= 1
                    st += 1

                seg[st:ed, col, b_scan_idx] = label
                last_st = st
                last_ed = ed

    seg[manualFluid > 0] = fluid_class

    seg = seg[:, min_col : max_col + 1]
    img = img[:, min_col : max_col + 1]

    img = img[:, :, valid_idx]
    return img, seg

def get_valid_img_seg_reimpl(scan_obj):
    fluid_class = 9

    manual_layers = np.array(scan_obj["manualLayers1"], dtype=np.uint16)
    manual_fluid = np.array(scan_obj["manualFluid1"], dtype=np.uint16)
    images = np.array(scan_obj["images"], dtype=np.uint8)
    valid_indices = get_id(manual_layers)

    manual_fluid = manual_fluid[:, :, valid_indices]
    manual_layers = manual_layers[:, :, valid_indices]

    segmentation = np.zeros_like(manual_fluid, dtype=np.uint8)

    for b_scan_idx in range(segmentation.shape[2]):
        for a_scan_idx in range(segmentation.shape[1]):
            class_indices = manual_layers[:, a_scan_idx, b_scan_idx]

            for i, _ in enumerate(class_indices):
                if i > 0 and class_indices[i] < class_indices[i - 1]:
                    class_indices[i] = class_indices[i - 1]

            for label, (start, end) in enumerate(
                zip([0, *class_indices], [*class_indices, segmentation.shape[0]])
            ):
                segmentation[start:end, a_scan_idx, b_scan_idx] = label

    segmentation[manual_fluid > 0] = fluid_class

    _, a_scan_used = np.where(np.sum(manual_layers, axis=(0, 2)) != 0)
    segmentation = segmentation[:, a_scan_used[0] : a_scan_used[-1] + 1]
    images = images[:, a_scan_used[0] : a_scan_used[-1] + 1]


    images = images[:, :, valid_indices]

    return images, segmentation


def get_unlabelled_bscans(scan_obj):
    manualLayer = np.array(scan_obj["manualLayers1"], dtype=np.uint16)
    img = np.array(scan_obj["images"], dtype=np.uint8)

    valid_idx = get_unlabelled(manualLayer)
    return img[:, :, valid_idx]
