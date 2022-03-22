import cv2
import numpy as np
from tqdm import tqdm


def water_filling(img, neta=0.22, iter=3):

    h, w = img.shape

    # Water
    w_ = np.zeros((h, w), dtype=np.float)

    # Overall height
    G_ = np.zeros((h, w), dtype=np.float)

    h_ = img.copy()
    h_ = img.astype(np.float)

    x = np.linspace(1, w-2, w-2)
    y = np.linspace(1, h-2, h-2)

    X, Y = np.meshgrid(x, y)
    X = X.astype(np.uint)
    Y = Y.astype(np.uint)

    # Left (x-delta)
    lx, ly = X-1, Y
    # Right (x+delta)
    rx, ry = X+1, Y
    # Top (y-delta)
    tx, ty = X, Y-1
    # Btm (y+delta)
    bx, by = X, Y+1

    print("[MSG] Water filling in progress ...")
    for t in tqdm(range(iter)):
        G_ = w_ + h_

        # Find local maximum using neighboring pixels
        left = G_[ly, lx]
        right = G_[ry, rx]
        top = G_[ty, tx]
        btm = G_[by, bx]

        stacked = np.stack([left, right, top, btm])

        G_peak = np.amax(stacked, axis=0)
        G_peak = np.pad(G_peak, ((1, 1), (1, 1)),
                        'constant', constant_values=0)

        pouring = np.exp(-t) * (G_peak - G_)

        left = -G_[Y, X] + left
        left[left > 0] = 0

        right = -G_[Y, X] + right
        right[right > 0] = 0

        top = -G_[Y, X] + top
        top[top > 0] = 0

        btm = -G_[Y, X] + btm
        btm[btm > 0] = 0

        del_w = neta * (left + right + top + btm)

        # del_w : (w-2) * (h-2)
        # pouring : w * h
        # w_ : w * h

        # To match the shape of del_w, padding is required
        del_w = np.pad(del_w, ((1, 1), (1, 1)),
                       'constant', constant_values=0)

        temp = del_w + pouring + w_

        temp[temp < 0] = 0

        w_[1: h - 2, 1: w - 2] = temp[1: h - 2, 1: w - 2]

    G_ = G_.astype(np.uint8)

    return G_


def binarize(img):
    """
        Helper function for umbra and penumbra extraction

        Median filtering -> OTSU binarization
    """

    ksize = 3

    median = cv2.medianBlur(img, ksize)

    _, binary = cv2.threshold(
        median, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return binary


def extract_umbra_and_penumbra(S):

    b, g, r = cv2.split(S)

    r = binarize(r)
    g = binarize(g)
    b = binarize(b)

    umbra = r + g + b

    # For a point, at least one of the three channels must be classified as shadow
    # i.e., if the sum of three channels is not three, it's shadow
    idx = umbra < 3
    umbra[idx] = 0
    umbra[~idx] = 255
    umbra = 255 - umbra

    kernel = np.ones((5, 5), np.uint8)

    dilated = cv2.dilate(umbra, kernel, iterations=2)

    penumbra = dilated - umbra

    return umbra, penumbra


def enhance_umbra(umbra, S, I):

    unshadowed_mask = umbra == 0

    g_color = []
    enhanced = []

    # For each channel, find a global background color
    for S_channel, I_channel in zip(cv2.split(S), cv2.split(I)):

        unshadowed = S_channel[unshadowed_mask]

        Gi = np.mean(unshadowed)

        g_color.append(Gi)

        n = Gi / S_channel

        # Remove umbra
        tmp = I_channel * n

        enhanced.append(tmp)

    enhanced = cv2.merge(enhanced, 3)
    enhanced = enhanced.astype(np.uint8)

    return enhanced, g_color


def main():
    I = cv2.imread("./006_014.jpg")

    adjusted = []

    """ 1. Local Water Filling Algorithm """
    for i, channel in enumerate(cv2.split(I)):
        print(f"[MSG] Processing {i}th channel ...")
        G_ = water_filling(channel)

        adjusted.append(G_)

    # Shading Map
    S = cv2.merge(adjusted, 3)

    """ 2. Extract Umbra and Penumbra """
    umbra, penumbra = extract_umbra_and_penumbra(S)

    """ 3. Umbra Enhancement """
    UI, G = enhance_umbra(umbra, S, I)

    """ 4. Penumbra Removal using LBWF """
    gray = cv2.cvtColor(UI, cv2.COLOR_BGR2GRAY)

    # Binarization Using Integral Image
    B1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 11, 20)

    adjusted = []

    for i, channel in enumerate(cv2.split(UI)):
        print(f"[MSG] Processing {i}th channel ...")
        _B2 = water_filling(channel, neta=1, iter=1)

        adjusted.append(_B2)

    B2 = cv2.merge(adjusted, 3)
    B2 = cv2.cvtColor(B2, cv2.COLOR_BGR2GRAY)

    B3 = cv2.bitwise_xor(B1, B2)
    B3 = 255 - B3
    B3 = cv2.cvtColor(B3, cv2.COLOR_GRAY2BGR)

    G = np.array(G, np.uint8)

    B3 += G

    cv2.imshow("B1", B1)
    cv2.imshow("B2", B2)
    cv2.imshow("B3", B3)
    cv2.imshow("UI", UI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
