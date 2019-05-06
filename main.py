import imutils
import argparse
import urllib.request
import numpy as np
import os
import cv2
from skimage.measure import compare_ssim as ssim, compare_mse as mse
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt

DATA_DIR = 'data'
INVENTORY_IMAGE_PATH = 'zelda-items.jpeg'
GRID_COORDS = [
    [120, 215], [510, 215],  # Top left / right
    [120, 510], [510, 510]   # Bottom left / right
]
GRID_SQUARE_SIZE_PX = 98

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=INVENTORY_IMAGE_PATH, help="path to the input image")
args = vars(ap.parse_args())


def download_zelda_items(data_dir):
    os.makedirs(data_dir)
    f = open('zelda_items.html', 'r')
    html = f.read()
    parsed_html = BeautifulSoup(html, features="html.parser")
    rows = parsed_html.find_all('tr')

    for row in rows[1:]:  # skip header
        name = str(row.td.img.get('alt'))
        source = str(row.td.img.get('src'))
        print("Downloading " + name)
        urllib.request.urlretrieve(source, data_dir + os.sep + name)


def create_item_collage(image_dir):
    collage = []
    image_paths = os.listdir(image_dir)

    for i in range(1, len(image_paths)):
        item_name = image_paths[i]
        item_image = cv2.imread(os.path.join(image_dir, item_name), cv2.IMREAD_COLOR)
        item_image = cv2.resize(item_image, (GRID_SQUARE_SIZE_PX, GRID_SQUARE_SIZE_PX))

        if i == 1:
            collage = item_image
        else:
            collage = np.vstack([collage, item_image])

    return image_paths, collage


def build_item_list(data_dir):
    if not os.path.isdir(data_dir):
        download_zelda_items(data_dir)

    image_paths = os.listdir(data_dir)
    item_list = []
    for i in range(1, len(image_paths)):
        item_name = image_paths[i]
        item_image = cv2.imread(os.path.join(data_dir, item_name), cv2.IMREAD_COLOR)
        item_image = cv2.resize(item_image, (GRID_SQUARE_SIZE_PX, GRID_SQUARE_SIZE_PX))
        item_list.append([item_name, item_image])

    return item_list


def template_matching():
    method = cv2.TM_CCOEFF_NORMED
    threshold = 0.7
    inventory_image = cv2.imread(INVENTORY_IMAGE_PATH, cv2.IMREAD_COLOR)
    inventory_image = inventory_image[
      GRID_COORDS[0][1]:GRID_COORDS[2][1]+GRID_SQUARE_SIZE_PX,
      GRID_COORDS[0][0]:GRID_COORDS[1][0]+GRID_SQUARE_SIZE_PX
    ]
    cv2.imshow('inventory', inventory_image)
    cv2.waitKey(0)
    item_list = build_item_list(DATA_DIR)

    for item_name, item_image in item_list:

        canvas = inventory_image.copy()
        res = cv2.matchTemplate(inventory_image, item_image, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            if min_val >= threshold:
                print("not detected")
                continue
            print(min_val)
            top_left = min_loc
        else:
            if max_val <= threshold:
                print("not detected")
                continue
            print(max_val)
            top_left = max_loc
            bottom_right = (top_left[0] + GRID_SQUARE_SIZE_PX, top_left[1] + GRID_SQUARE_SIZE_PX)

        cv2.rectangle(canvas, top_left, bottom_right, 255, 2)
        cv2.imshow(item_name, canvas)
        cv2.waitKey(0)


def main():
    img = cv2.imread(INVENTORY_IMAGE_PATH, cv2.IMREAD_COLOR)
    item_list = build_item_list(DATA_DIR)
    # Canny params
    thresh = 900
    thresh_2 = thresh/3
    elipse = (5, 5)

    print("extracting coords")
    for x in range(GRID_COORDS[0][0], GRID_COORDS[1][0], GRID_SQUARE_SIZE_PX):
        for y in range(GRID_COORDS[0][1], GRID_COORDS[2][1], GRID_SQUARE_SIZE_PX):
            item_image = img[y:y + GRID_SQUARE_SIZE_PX, x:x + GRID_SQUARE_SIZE_PX]
            item = extract_item(item_image, thresh, thresh_2, elipse)
            best_sim = None
            best_sim_name = None
            best_sim_image = None

            mses = [[mse(item, icon_image), icon_name] for icon_name, icon_image in item_list]
            mses = sorted(mses, key=lambda x: x[0])
            mse_rank_dict = {}
            for idx, [mse_obj, name] in enumerate(mses):
                mse_rank_dict[name] = idx

            ssims = [[ssim(item, icon_image, multichannel=True, gaussian_weights=True), icon_name] for icon_name, icon_image in item_list]
            ssims = sorted(ssims, key=lambda x: x[0], reverse=True)
            ssim_rank_dict = {}
            for idx, [mse_obj, name] in enumerate(ssims):
                ssim_rank_dict[name] = idx

            for icon_name, icon_image in item_list:
                mse_rank = mse_rank_dict[icon_name]
                ssim_rank = ssim_rank_dict[icon_name]
                avg_rank = (mse_rank + ssim_rank) / 2
                is_best_sim = not best_sim or avg_rank < best_sim

                if is_best_sim:
                    best_sim = avg_rank
                    best_sim_name = icon_name
                    best_sim_image = icon_image

            print(best_sim)
            cv2.imshow(best_sim_name, np.hstack([item_image, item, best_sim_image]))
            cv2.waitKey(0)


def histogram():
    img = cv2.imread(INVENTORY_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    img = np.hstack((img,equ))
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    cv2.imshow('image', img)
    cv2.waitKey(0)


def extract_item(image, thresh, thresh_2, elipse):
    edit_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    edit_image = cv2.equalizeHist(edit_image)

    # Extract contours
    edges = cv2.Canny(edit_image, thresh, thresh_2)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt_len = [len(c) for c in cnts]
    max_idx = cnt_len.index(max(cnt_len))
    item_cnt = cnts[max_idx]

    # Extract only item contour
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [item_cnt], -1, (255, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, elipse)
    dilated = cv2.dilate(mask, kernel)
    dilated = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, cnts, -1, (255, 255, 255), thickness=cv2.FILLED)
    item = cv2.bitwise_and(image, mask)
    return item


if __name__ == "__main__":
    main()