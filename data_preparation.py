import os
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from datetime import datetime
from requests import get
import zipfile
import tarfile
import shutil
import config
import dlib
from tqdm import tqdm
from PIL import Image


detector = dlib.get_frontal_face_detector()
DATA_PATH = config.DATA_PATH
VAL_SPLIT = config.VAL_SPLIT
if config.USE_ALL_DATA:
    VAL_SPLIT = 1.0

assert config.PHOTO_DIST in ('CU', 'ECU'), 'Unknown PHOTO_DIST parameter. It must be either "CU" or "ECU"'


def detect_face(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected = detector(img, 1)

    # No multiple faces and no no faces at all
    if len(detected) != 1:
        return None

    d = detected[0]
    x1, y1, x2, y2, = d.left(), d.top(), d.right() + 1, d.bottom() + 1

    return x1, y1, x2, y2


def imcrop(img, x1, y1, x2, y2):

    w, h = x2 - x1, y2 - y1

    x1 = int(x1 - config.CU_MARGIN * w)
    y1 = int(y1 - config.CU_MARGIN * h)
    x2 = int(x2 + config.CU_MARGIN * w)
    y2 = int(y2 + config.CU_MARGIN * h)

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)

    return img, x1, x2, y1, y2


def download_file(url, file_name=None):

    if file_name is None:
        file_name = url.split('/')[-1]

    with open(file_name, 'wb') as f:
        response = get(url)
        f.write(response.content)


def download_and_extract_appa_real():

    # Info
    url = 'http://158.109.8.102/AppaRealAge/appa-real-release.zip'
    save_name = os.path.join(DATA_PATH, 'appa-real-release.zip')
    dir_name = os.path.join(DATA_PATH, 'appa-real-release')

    print('APPA-REAL dataset')

    # Download
    if not os.path.exists(save_name):
        print('Downloading...')
        download_file(url, save_name)
    else:
        print('Dataset already exists')

    # Extract
    if not os.path.exists(dir_name):
        print('Extracting...')
        with zipfile.ZipFile(save_name, 'r') as z:
            z.extractall(path=DATA_PATH)
    else:
        print('Already extracted')


def extract_utk():

    print('UTKFace dataset')
    parts = ['part1.tar.gz', 'part2.tar.gz', 'part3.tar.gz']

    for part in parts:

        print(part)

        # Info
        save_name = os.path.join(DATA_PATH, part)
        dir_name = os.path.join(DATA_PATH, part.split('.')[0])

        assert os.path.exists(save_name), 'Please download the UTKFace in the wild dataset part1.tar.gz, ' \
                                          'part2.tar.gz, part3.tar.gz'

        # Extract
        if not os.path.exists(dir_name):
            print('Extracting...')
            with tarfile.open(save_name, 'r') as t:
                t.extractall(path=DATA_PATH)
        else:
            print('Already extracted')


def download_and_extract_imdb():

    # Info
    url = 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar'
    save_name = os.path.join(DATA_PATH, 'imdb_crop.tar')
    dir_name = os.path.join(DATA_PATH, 'imdb_crop')

    print('IMDB dataset')

    # Download
    if not os.path.exists(save_name):
        print('Downloading...')
        download_file(url, save_name)
    else:
        print('Dataset already exists')

    # Extract
    if not os.path.exists(dir_name):
        print('Extracting...')
        with tarfile.open(save_name, 'r') as t:
            t.extractall(path=DATA_PATH)
    else:
        print('Already extracted')


def extract_sof():

    # Info
    save_name = os.path.join(DATA_PATH, 'original images.rar')
    dir_name = os.path.join(DATA_PATH, 'original images')

    print('SoF dataset')

    assert os.path.exists(save_name), 'Please download the SoF ' \
                                      'drive.google.com/uc?id=0BwO0RMrZJCioaW5TdVJtOEtfYUk&export=download ' \
                                      'and place it into data folder. The Google Drive has a messed API for ' \
                                      'automatic downloading'

    # Extract
    assert os.path.exists(dir_name), 'Please extract the original images.rar folder. The directory must have name ' \
                                     '"original images"'


def prepare_class_dirs(dataset_name, train=True):

    dataset_path = os.path.join(DATA_PATH, 'processed', dataset_name)
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    # Choose the sets

    dataset_sets = ['train'] if config.USE_ALL_DATA else ['train', 'valid']
    sets = dataset_sets if train else ['test']

    # Create class directories
    for part in sets:

        if not os.path.isdir(os.path.join(dataset_path, part)):
            os.mkdir(os.path.join(dataset_path, part))

        for i in range(0, 120):
            target_path = os.path.join(dataset_path, part, str(i))
            if not os.path.isdir(target_path):
                os.mkdir(target_path)


def prepare_appa_real():

    print('Preparing APPA-REAL dataset')

    dir_name = os.path.join(DATA_PATH, 'appa-real-release')

    # List of bad images
    with open('./meta/appa-real-ignore-list.txt', 'rb') as f:
        ignore_imgs = set([_.strip() for _ in f.readlines()])

    # Create processed dataset just in case
    if not os.path.isdir(os.path.join(DATA_PATH, 'processed')):
        os.mkdir(os.path.join(DATA_PATH, 'processed'))

    # Create dataset specific folder if does not exist
    new_name = 'appa-real'
    if os.path.isdir(os.path.join(DATA_PATH, 'processed', new_name)):
        print('APPA-REAL already preprocessed')
        return

    prepare_class_dirs(new_name)

    # Process train and validation sets
    for df_name in ['gt_avg_train.csv', 'gt_avg_valid.csv']:

        df_name = os.path.join(dir_name, df_name)
        df = pd.read_csv(df_name)

        part = 'train' if 'train' in df_name else 'valid'
        part_dst = 'train' if config.USE_ALL_DATA else 'valid'

        print('Processing {} set'.format(part))

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):

            # Ignore list
            if row['file_name'] in ignore_imgs:
                continue

            img_name = row['file_name'] + '_face.jpg'
            age = row['real_age']

            # Restrict ages
            if not 0 <= age < 120:
                continue

            # Copy the files
            src = os.path.join(dir_name, part, img_name)
            dst = os.path.join(DATA_PATH, 'processed', new_name, part_dst, str(age), img_name)

            if config.PHOTO_DIST == 'CU':
                shutil.copyfile(src, dst)
            else:
                img = cv2.imread(src)
                res = detect_face(img)
                if res is None:
                    continue

                x1, y1, x2, y2 = res
                cv2.imwrite(dst, img[y1:y2, x1:x2])

    # Process test set
    for df_name in ['gt_avg_test.csv']:

        df_name = os.path.join(dir_name, df_name)
        df = pd.read_csv(df_name)

        # Set a seed for reproducibility
        np.random.seed(42)

        part = 'train' if VAL_SPLIT > np.random.rand() else 'valid'
        print('Processing test set')

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):

            # Ignore list
            if row['file_name'] in ignore_imgs:
                continue

            img_name = row['file_name'] + '_face.jpg'
            age = row['real_age']

            # Restrict ages
            if not 0 <= age < 120:
                continue

            # Copy the files
            src = os.path.join(dir_name, 'test', img_name)
            dst = os.path.join(DATA_PATH, 'processed', new_name, part, str(age), img_name)

            if config.PHOTO_DIST == 'CU':
                shutil.copyfile(src, dst)
            else:
                img = cv2.imread(src)
                res = detect_face(img)
                if res is None:
                    continue

                x1, y1, x2, y2 = res
                cv2.imwrite(dst, img[y1:y2, x1:x2])


def prepare_utk():

    # Create processed dataset just in case
    if not os.path.isdir(os.path.join(DATA_PATH, 'processed')):
        os.mkdir(os.path.join(DATA_PATH, 'processed'))

    print('Preparing UTKFace dataset')
    # Create dataset specific folder if does not exist
    new_name = 'utk'
    if os.path.isdir(os.path.join(DATA_PATH, 'processed', new_name)):
        print('UTKFace already preprocessed')
        return
    prepare_class_dirs(new_name)

    # Set random seed for reproducibility
    np.random.seed(42)

    parts = ['part1', 'part2', 'part3']

    for part in parts:

        dir_name = os.path.join(DATA_PATH, part)
        print('Processing {}'.format(part))
        # Process sets

        img_names = os.listdir(dir_name)
        for i in tqdm(range(len(img_names))):

            img_name = img_names[i]

            # Ignore non-images
            if not img_name.endswith('.jpg'):
                continue

            # Obtain image from file name
            age = img_name.split('_')[0]

            # Restrict ages
            if not 0 <= int(age) < 120:
                continue

            # Decide randomly where the image goes
            part = 'train' if VAL_SPLIT > np.random.rand() else 'valid'

            # Copy the files
            src = os.path.join(dir_name, img_name)
            dst = os.path.join(DATA_PATH, 'processed', new_name, part, age, img_name)

            # Depending on photo distance decide how to crop
            if config.PHOTO_DIST == 'CU':
                img = cv2.imread(src)
                res = detect_face(img)
                if res is None:
                    continue

                x1, y1, x2, y2 = res
                img = imcrop(img, x1, y1, x2, y2)
                cv2.imwrite(dst, img)

            else:
                img = cv2.imread(src)
                res = detect_face(img)
                if res is None:
                    continue

                x1, y1, x2, y2 = res
                cv2.imwrite(dst, img[y1:y2, x1:x2])


def prepare_imdb():

    print('Preparing IMDB dataset')

    dir_name = os.path.join(DATA_PATH, 'imdb_crop')

    # Create processed dataset just in case
    if not os.path.isdir(os.path.join(DATA_PATH, 'processed')):
        os.mkdir(os.path.join(DATA_PATH, 'processed'))

    # Create dataset specific folder if does not exist
    new_name = 'imdb'
    if os.path.isdir(os.path.join(DATA_PATH, 'processed', new_name)):
        print('IMDB already preprocessed')
        return
    prepare_class_dirs(new_name)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Obtain info from
    meta = loadmat(os.path.join(dir_name, 'imdb.mat'))['imdb'][0][0]
    dobs = meta['dob'][0]
    takens = meta['photo_taken'][0]
    paths = meta['full_path'][0]
    face_scores = meta['face_score'][0]
    second_face_scores = meta['second_face_score'][0]

    # Process sets
    for i in tqdm(range(len(dobs))):

        dob = dobs[i]
        taken = takens[i]
        path = paths[i].tolist()[0]
        age = taken - datetime.fromordinal(max(int(dob) - 366, 1)).year
        img_name = path.split('/')[1]
        face_score = face_scores[i]
        second_face_score = second_face_scores[i]

        # Restrict ages
        if not 18 <= age < 120:
            continue

        # Delete no faces
        if np.isinf(face_score) or face_score < 1.0:
            continue

        # Delete multiple faces
        if not np.isnan(second_face_score):
            continue

        # Decide randomly where the image goes
        part = 'train' if VAL_SPLIT > np.random.rand() else 'valid'

        # Copy the files
        src = os.path.join(dir_name, path)
        dst = os.path.join(DATA_PATH, 'processed', new_name, part, str(age), img_name)

        if config.PHOTO_DIST == 'CU':
            shutil.copyfile(src, dst)
        else:
            img = cv2.imread(src)
            res = detect_face(img)
            if res is None:
                continue

            x1, y1, x2, y2 = res
            cv2.imwrite(dst, img[y1:y2, x1:x2])


def prepare_sof():

    print('Preparing SoF dataset')

    dir_name = os.path.join(DATA_PATH, 'original images')

    # Create processed dataset just in case
    if not os.path.isdir(os.path.join(DATA_PATH, 'processed')):
        os.mkdir(os.path.join(DATA_PATH, 'processed'))

    # Create dataset specific folder
    new_name = 'sof'
    if os.path.isdir(os.path.join(DATA_PATH, 'processed', new_name)):
        print('SOF already preprocessed')
        return
    prepare_class_dirs(new_name, train=False)

    # Load info
    id2coors = {}
    meta = loadmat('./meta/metadata.mat')['metadata'][0]
    for person in meta:
        idx = person[1][0][0].tolist()[0]
        x1, y1, w, h = person[14][0].astype(np.int32)
        x2, y2 = x1 + w, y1 + h
        id2coors[idx] = [x1, y1, x2, y2]

    # Process sets
    for img_name in os.listdir(dir_name):

        # Ignore non-images
        if not img_name.endswith('.jpg'):
            continue

        # Obtain image info and crop image
        img_info = img_name.split('_')
        idx = img_info[1]
        age = img_info[3]
        x1, y1, x2, y2 = id2coors[idx]
        img = cv2.imread(os.path.join(dir_name, img_name))

        if config.PHOTO_DIST == 'CU':
            crop = imcrop(img, x1, y1, x2, y2)
        else:
            crop = img[y1:y2, x1:x2]

        # Restrict ages
        if not 0 <= int(age) < 120:
            continue

        # Save the crop
        dst = os.path.join(DATA_PATH, 'processed', new_name, 'test', age, img_name)

        cv2.imwrite(dst, crop)


def download_and_extract_all():
    print('Downloading and extracting files...\n')
    download_and_extract_appa_real()
    print('\n')
    extract_utk()
    print('\n')
    download_and_extract_imdb()
    print('\n')
    extract_sof()


def prepare_all():
    print('Preparing datasets...\n')
    prepare_appa_real()
    print('\n')
    prepare_utk()
    print('\n')
    prepare_imdb()
    print('\n')
    prepare_sof()


def clean_brokens():
    print('Looking for broken images...')
    # Clean broken images
    path = os.path.join(DATA_PATH, 'processed/')
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.jpg'):
                img_path = os.path.join(root, f)
                try:
                    Image.open(img_path).verify()
                except OSError:
                    print('Removing: {}'.format(img_path))
                    os.remove(img_path)


def symlink_test_set():

    print('Creating symlinks...')

    # Absolute path to source test set
    src = os.path.join(DATA_PATH, 'processed/sof/test')
    src = os.path.realpath(src)

    # According absolute paths to datasets
    suffix = 'valid' if config.USE_ALL_DATA else 'test'

    appa_real_dst = os.path.join(DATA_PATH, 'processed/appa-real/', suffix)
    if os.path.isdir(appa_real_dst):
        os.remove(appa_real_dst)
    appa_real_dst = os.path.realpath(appa_real_dst)
    os.symlink(src, appa_real_dst)

    utk_dst = os.path.join(DATA_PATH, 'processed/utk/', suffix)
    if os.path.isdir(utk_dst):
        os.remove(utk_dst)
    utk_dst = os.path.realpath(utk_dst)
    os.symlink(src, utk_dst)

    imdb_dst = os.path.join(DATA_PATH, 'processed/imdb/', suffix)
    if os.path.isdir(imdb_dst):
        os.remove(imdb_dst)
    imdb_dst = os.path.realpath(imdb_dst)
    os.symlink(src, imdb_dst)


def prepare_data():
    download_and_extract_all()
    prepare_all()
    clean_brokens()
    symlink_test_set()


if __name__ == '__main__':
    prepare_data()
