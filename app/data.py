import pandas as pd
import os
from config import Config
import zipfile
from tqdm import tqdm
from steps import Steps
from pathlib import Path
import shutil
from tabulate import tabulate
from dfs import Dfs
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import albumentations as A
import imgaug
from rectangle import Rectangle
import matplotlib.patches as patches
import wget
import base64
from io import BytesIO


class Data():

    def __init__(self, force_extract=False, mode='notebook'):
        self.mode = mode
        self.df_images_path = 'images.parquet'
        self.df_boxes_path = 'boxes.parquet'
        self.dataset_yaml = 'dataset.yaml'
        self.df_images: pd.DataFrame = None
        self.df_boxes: pd.DataFrame = None

        if mode != 'notebook':
            sns.set_context("talk")
            sns.set_palette('colorblind')

        if force_extract or not os.path.exists(Config.ORIGINAL_DATA_DIR):
            self.clean_data() 
            self.unzip()
            self.get_dfs()

    def clean_data(self):
        if os.path.exists(Config.ORIGINAL_DATA_DIR):
            Path(Config.ORIGINAL_DATA_DIR).rmdir()
        if os.path.exists(Config.PREPARED_DATA_DIR):
            Path(Config.PREPARED_DATA_DIR).rmdir()
        if os.path.exists(self.df_images_path):
            os.remove(self.df_images_path)
        if os.path.exists(self.df_boxes_path):
            os.remove(self.df_boxes_path)
        if os.path.exists(self.dataset_yaml):
            os.remove(self.dataset_yaml)
            
    def unzip(self):
        print('Téléchargement des data')
        file_zip = wget.download(Config.DATA_ZIP_URL)
        
        print('Extraction des data')
        with zipfile.ZipFile(file_zip, 'r') as zf:
                for member in tqdm(zf.infolist(), desc='Extraction '):
                    try:
                        zf.extract(member)
                    except zipfile.error as e:
                        pass
        os.remove(file_zip)

    def createYaml(self):
        lines = [f'path: {os.path.join(Config.ORIGINAL_DATA_DIR)}']
        for step in Steps.ALL:
            step_name = 'val' if step == Steps.VALID else step
            lines.append(f'{step_name}: {step}/images')
        lines.append('')
        lines.append(f'nc: {len(Config.CLASSES)}')
        for class_name in Config.CLASSES:
            lines.append(f'names: [\'{class_name}\']')
        with open(self.dataset_yaml, 'w') as f:
            f.write('\n'.join(lines))
            f.close()

    def prepare_data(self, min_size = Config.BATCH_IMAGE_SIZE):
        print(f'Préparation du dataset')
        self.createYaml()

        dest = os.path.join(Config.PREPARED_DATA_DIR, Config.ORIGINAL_DATA_DIR)

        print(f'Copie des fichiers originaux')
        Path(dest).mkdir(parents=True, exist_ok=True)
        shutil.copytree(Config.ORIGINAL_DATA_DIR, dest, dirs_exist_ok=True)
        os.remove(os.path.join(dest, 'data.yaml'))

        table=[['Step', 'avant', 'après']]
        for step in Steps.ALL:
            dfs = self.get_dfs(filter_valid=True, filter_min_size=min_size, step=step)

            images_destination = os.path.join(dest,step,'images')
            labels_destination = os.path.join(dest,step,'labels')

            images = sorted([img for img in os.listdir(images_destination)])
            labels = sorted([img for img in os.listdir(labels_destination)])
            len_images_before = len(os.listdir(images_destination))
            authorizes_indexes = dfs['images']['index'].to_list()

            for i in tqdm(range(len(images)), desc=f"Filtrage des images et labels {step}"):
                index = images[i][:-4]
                if not index in authorizes_indexes:
                    image_path = os.path.join(images_destination, images[i])
                    label_path = os.path.join(labels_destination, labels[i])
                    os.remove(image_path)
                    os.remove(label_path)

            len_images_after = len(os.listdir(images_destination))
            table.append([step, len_images_before, len_images_after])

        print('')
        print('Nombre d\'images avant et après filtrage:')
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    def get_dfs(self, load_if_exist=True, filter_valid = False, filter_min_size = None, step = None) -> Dfs:
        if load_if_exist and (self.df_images is not None) and (self.df_boxes is not None):
            pass
        elif load_if_exist and os.path.exists(self.df_images_path) and os.path.exists(self.df_boxes_path):
            self.df_images = pd.read_parquet(self.df_images_path)
            self.df_boxes = pd.read_parquet(self.df_boxes_path)
        else:
            infos = {
                'images': [],
                'boxes': []
            }

            max_cell_size = max([Config.LARGEUR_CELLULE, Config.HAUTEUR_CELLULE])

            for step in Steps.ALL:
                print(f'Traitement du dataset {step}')

                images_dir = Config.ORIGINAL_DATA_DIR + '/' + step + '/images/'
                labels_dir = Config.ORIGINAL_DATA_DIR + '/' + step + '/labels/'

                images = sorted([img for img in os.listdir(images_dir)])
                labels = sorted([img for img in os.listdir(labels_dir)])

                total_invalid_boxes = 0
                total_boxes = 0

                for i in tqdm(range(len(images))):
                    # Images = Label?
                    assert images[i][:-3] == labels[i][:-3], f'step: {step}\n images: {images[i]}\n label: {labels[i]}'
                    index = images[i][:-4]

                    # paths
                    image_path = images_dir + images[i]
                    label_path = labels_dir + labels[i]

                    # Image infos
                    image = Image.open(image_path)
                    width = image.size[0]
                    height = image.size[1]
                    mode = image.mode
                    image.close()

                    # Boxes infos
                    boxes = []
                    with open(label_path, 'r') as f:
                        for line in f:
                            boxes.append(line.strip().split())
                    f.close()

                    nb_invalid = 0

                    if (len(boxes) > 0):
                        nb_boxes = len(boxes)
                        nb_invalid = np.sum([1 if len(box) != 5 else 0 for box in boxes])

                        total_boxes = total_boxes + nb_boxes
                        total_invalid_boxes = total_invalid_boxes + nb_invalid

                        if (nb_boxes > 1) and (nb_invalid == 0):
                            # centers box distances
                            centers_pixels = (np.array(boxes, dtype=np.float32)[:, [1,2]]*[width,height]).astype(np.int16)
                            distances_pixels = pairwise_distances(centers_pixels)
                            in_same_cell = np.where(distances_pixels < max_cell_size, 1, 0)
                            nb_in_same_cell = in_same_cell.sum(axis=1) - 1
                        else:
                            nb_in_same_cell = np.full(nb_boxes, 0)

                        for box_index, box in enumerate(boxes):
                            if len(box) == 5:
                                box = np.array(box, dtype=np.float32)
                                infos['boxes'].append([index, label_path, int(box[0]), box[1], box[2], box[3], box[4], box[3]*box[4], nb_in_same_cell[box_index], step, 1])
                            else:
                                infos['boxes'].append([index, label_path, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, step, 0])

                    infos['images'].append([index, image_path, width, height, math.ceil(width*1000/height)/1000, mode, max(nb_in_same_cell), step, 1 if nb_invalid == 0 else 0])

                print(f'{total_boxes} boxes dont {total_invalid_boxes} invalides')

            self.df_images = pd.DataFrame(infos['images'], columns=['index', 'path', 'width', 'height', 'ratio', 'mode', 'box_in_same_cell', 'step', 'valid'])
            self.df_boxes = pd.DataFrame(infos['boxes'], columns=['index', 'path', 'class', 'x', 'y', 'width', 'height', 'surface', 'max_box_in_same_cell', 'step', 'valid'])
            self.df_images.to_parquet(self.df_images_path)
            self.df_boxes.to_parquet(self.df_boxes_path)

        df_images = self.df_images
        df_boxes = self.df_boxes

        if filter_valid:
            df_images = df_images[df_images['valid']==1]
            df_boxes = df_boxes[df_boxes['valid']==1]

        if filter_min_size is not None:
            df_images = df_images[(df_images['width']>=filter_min_size[0]) & (df_images['height']>=filter_min_size[1])]

        if step is not None:
            df_images = df_images[df_images['step'] == step]
            df_boxes = df_boxes[df_boxes['step'] == step]

        return {'images': df_images, 'boxes': df_boxes}

    def read_label(self, label):
        boxes = []
        with open(label, 'r') as f:
            for line in f:
                boxes.append(line.strip().split())

        f.close()
        return boxes

    def plot_hist_by_step(self, df_name: str, x: str, bins=5, title = '', x_label = '', y_label='', steps=Steps.ALL):
        dfs = self.get_dfs(filter_valid=True)
        fig, axis = plt.subplots(1, len(steps), layout='constrained', sharey=True)
        fig.set_figwidth(12)
        fig.suptitle(title)
        for index, step in enumerate(steps):
            current_axis = axis[index] if type(axis) is np.ndarray else axis
            df = dfs[df_name][dfs[df_name]['step'] == step]
            sns.histplot(df, x=x, bins=bins, ax=current_axis)
            current_axis.set_title(step)
            current_axis.set_xlabel(None)
            current_axis.set_ylabel(None)
        fig.supxlabel(x_label)
        fig.supylabel(y_label)

        if self.mode == 'notebook':
            plt.show()
        else:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            return base64.b64encode(buf.getbuffer()).decode("ascii")

    def plot_values_by_step(self, df_name: str, x: str, title = '', x_label = '', y_label='', steps=Steps.ALL):
        dfs = self.get_dfs(filter_valid=True)
        fig, axis = plt.subplots(1, len(steps), layout='constrained', sharey=True)
        fig.set_figwidth(12)
        fig.suptitle(title)
        for index, step in enumerate(steps):
            current_axis = axis[index] if type(axis) is np.ndarray else axis
            df = dfs[df_name][dfs[df_name]['step'] == step]
            df[x].sort_values(ascending=True).value_counts(sort=False).plot(kind='bar', ax=current_axis)
            current_axis.set_title(step)
            current_axis.set_xlabel(None)
            current_axis.set_ylabel(None)
        fig.supxlabel(x_label)
        fig.supylabel(y_label)
        
        if self.mode == 'notebook':
            plt.show()
        else:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            return base64.b64encode(buf.getbuffer()).decode("ascii")

    def plot_pie_valid_boxes_by_step(self, steps=Steps.ALL):
        dfs = self.get_dfs()
        fig, axis = plt.subplots(1, len(steps), layout='constrained', sharey=True)
        fig.set_figwidth(12)
        fig.suptitle('Nombre de boites valides')
        for index, step in enumerate(steps):
            current_axis = axis[index] if type(axis) is np.ndarray else axis
            plot = dfs['boxes'][dfs['boxes']['step'] == step]['valid'].value_counts().plot.pie(ax=current_axis, autopct='%1.1f%%', explode=(0, 0.2), shadow=False, startangle=0)
            current_axis.set_title(step)
            current_axis.set_xlabel(None)
            current_axis.set_ylabel(None)
        fig.supxlabel(None)
        fig.supylabel(None)
        
        if self.mode == 'notebook':
            plt.show()
        else:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            return base64.b64encode(buf.getbuffer()).decode("ascii")

    def plot_count_boxes_per_images_by_step(self, steps=Steps.ALL):
        dfs = self.get_dfs(filter_valid=True)
        fig, axis = plt.subplots(1, len(steps), layout='constrained', sharey=True)
        fig.set_figwidth(12)
        fig.suptitle('Nombre de boites par image')
        for index, step in enumerate(steps):
            current_axis = axis[index] if type(axis) is np.ndarray else axis
            df = dfs['boxes'][dfs['boxes']['step'] == step]
            df['index'].sort_values(ascending=True).value_counts(sort=False).plot.hist(ax=current_axis, logy=True)
            current_axis.set_title(step)
            current_axis.set_xlabel(None)
            current_axis.set_ylabel(None)
        fig.supxlabel('Nombre de boites')
        fig.supylabel('Nombre d\'images')
        
        if self.mode == 'notebook':
            plt.show()
        else:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            return base64.b64encode(buf.getbuffer()).decode("ascii")

    def plot_box_in_same_cell(self, steps=Steps.ALL):
        dfs = self.get_dfs(filter_valid=True, filter_min_size=Config.BATCH_IMAGE_SIZE)
        fig, axis = plt.subplots(1, len(steps), layout='constrained', sharey=True)
        fig.set_figwidth(12)
        fig.suptitle(f'Dépassement du nombre de boites dans une même cellule (max={len(Config.ANCHORS)})')
        for index, step in enumerate(steps):
            current_axis = axis[index] if type(axis) is np.ndarray else axis
            df = dfs['images'][dfs['images']['step'] == step]
            df[df['box_in_same_cell'] > len(Config.ANCHORS)]['box_in_same_cell'].sort_values().value_counts(sort=False).plot(ax=current_axis)
            df[df['box_in_same_cell'] > len(Config.ANCHORS)]['box_in_same_cell'].sort_values().value_counts(sort=False).cumsum().plot(ax=current_axis)
            current_axis.set_title(step)
            current_axis.set_xlabel(None)
            current_axis.set_ylabel(None)
            current_axis.legend(['Nombre d\'images', 'Nombre d\'images cumulées'])
        fig.supxlabel('Nombre de boites')
        fig.supylabel('Nombre d\'images')
        
        if self.mode == 'notebook':
            plt.show()
        else:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            return base64.b64encode(buf.getbuffer()).decode("ascii")

    def get_indexes(self, step, min_size = Config.BATCH_IMAGE_SIZE):
        dfs = dfs = self.get_dfs(filter_valid=True, filter_min_size=min_size, step=step)
        return dfs['images']['index'].to_list()

    def get(self, index, crop=Config.BATCH_IMAGE_SIZE, seed = None):
        dfs = self.get_dfs()

        # Image
        dfs_image_row = dfs['images'][dfs['images']['index'] == index].iloc[0]
        image_path = dfs_image_row['path']
        image = np.asarray(Image.open(image_path))

        # boxes
        df_boxes = dfs['boxes'][dfs['boxes']['index'] == index]
        yolo_boxes = [[box.x, box.y, box.width, box.height] for _, box in df_boxes.iterrows()]

        # Transform
        if seed is not None:
            random.seed(seed)
            imgaug.random.seed(seed)

        format_transform = A.Compose([
            A.RandomCrop(
                height=crop[1],
                width=crop[0],
                p=1.0,
            )
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=[], clip=True))

        result = format_transform(image=image, bboxes=yolo_boxes)
        target, table = self.yolo_boxes_to_target(result['bboxes'])

        random.seed(None)
        return [result['image'], result['bboxes'], target, table]

    def yolo_box_to_rectangle(self, yolo_box, image) -> Rectangle:
        x_min = int((yolo_box[0] - yolo_box[2]/2)*image.shape[0])
        y_min = int((yolo_box[1] - yolo_box[3]/2)*image.shape[1])
        width = int(yolo_box[2]*image.shape[0])
        height = int(yolo_box[3]*image.shape[1])
        x_max = x_min+width
        y_max = y_min+height
        return {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'width': width,
                'height': height,
                'center_x': x_min + int(width/2),
                'center_y': y_min + int(height/2),
                'area': width*height
            }

    def display_image(self, index, with_boxes=True, crop=Config.BATCH_IMAGE_SIZE, ax = None, print_target=False, seed=None):
        [image, yolo_boxes, target, table] = self.get(index=index, crop=crop, seed=seed)
        if print_target:
            print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        if with_boxes:
            self.draw_image(image=image, yolo_boxes=yolo_boxes, ax=ax)
        else:
            self.draw_image(image=image, ax=ax)

    def draw_image(self, image, yolo_boxes = [], ax = None, grid=True):
        if ax is None:
            fig = plt.figure(linewidth=0)
            ax = plt.gca()

        ax.grid(False)
        ax.imshow(image)

        for yolo_box in yolo_boxes:
            rectangle = self.yolo_box_to_rectangle(yolo_box, image)
            rect = patches.Rectangle((rectangle['x_min'],
                                      rectangle['y_min']),
                                      rectangle['width'],
                                      rectangle['height'],
                                      linewidth=1,
                                      edgecolor='w',
                                      facecolor="none")
            center = patches.Circle((rectangle['center_x'], rectangle['center_y']), radius=2)

            ax.add_patch(rect)
            ax.add_patch(center)

        if grid:
            for i in range(Config.GRID_X):
                ax.vlines(image.shape[0]/Config.GRID_X*i, 1, image.shape[1] - 1, colors=(1,1,1,0.15))
            for i in range(Config.GRID_Y):
                ax.hlines(image.shape[1]/Config.GRID_Y*i, 1, image.shape[0] - 1, colors=(1,1,1,0.15))

    def display_sample(self, indexes = None, with_boxes=True, nb_images=1, print_target=False):
        if indexes is None:
            dfs = self.get_dfs(filter_valid=True, filter_min_size=Config.BATCH_IMAGE_SIZE)
            sample = dfs['images'].sample(nb_images)
            indexes = sample['index'].to_list()
        else:
            nb_images = len(indexes)
        print(indexes)

        cols = 1
        rows = nb_images

        fig, axis = plt.subplots(rows, cols, figsize=(14,10), linewidth=0)

        for i in range(nb_images):
            if type(axis) is np.ndarray:
                if rows > 1:
                    current_axis = axis[i]
                else:
                    current_axis = axis[0]
            else:
                current_axis = axis

            self.display_image(index=indexes[i], with_boxes=with_boxes, ax=current_axis, print_target=print_target)
            if self.mode == 'notebook':
                plt.show()
            else:
                buf = BytesIO()
                fig.savefig(buf, format="png")
                return base64.b64encode(buf.getbuffer()).decode("ascii")

    def yolo_boxes_to_target(self, yolo_boxes):
        target = np.zeros((Config.GRID_X, Config.GRID_Y, len(Config.ANCHORS), 5 + len(Config.CLASSES)), dtype=np.float32)
        class_id=0
        table = [['#', 'cell X', 'cell Y', 'anchor', 'iou', 'largeur', 'hauteur', 'centre x', 'centre y']]
        for box_number, yolo_box in enumerate(yolo_boxes):
            cx = Config.GRID_X * yolo_box[0]
            cy = Config.GRID_Y * yolo_box[1]
            width = Config.GRID_X * yolo_box[2]
            height = Config.GRID_Y * yolo_box[3]

            cell_x = int(cx)
            cell_y = int(cy)

            center_x = cx - cell_x
            center_y = cy - cell_y

            t_x_min = cx - width/2
            t_x_max = cx + width/2
            t_y_min = cy - height/2
            t_y_max = cy + height/2

            id_a=0
            best_iou=0
            for index, anchor in enumerate(Config.ANCHORS):
                a_x_min = cx - anchor[0]/2
                a_x_max = cx + anchor[0]/2
                a_y_min = cy - anchor[1]/2
                a_y_max = cy + anchor[1]/2
                iou=self.intersection_over_union([t_x_min, t_y_min, t_x_max, t_y_max], [a_x_min, a_y_min, a_x_max, a_y_max])
                if iou>best_iou:
                    best_iou=iou
                    id_a=index

            table.append([box_number+1, cell_x, cell_y, id_a, best_iou, width, height, center_x, center_y])

            target[cell_x, cell_y, id_a, 0]=center_x
            target[cell_x, cell_y, id_a, 1]=center_y
            target[cell_x, cell_y, id_a, 2]=width
            target[cell_x, cell_y, id_a, 3]=height
            target[cell_x, cell_y, id_a, 4]=1.
            target[cell_x, cell_y, id_a, 5+class_id]=1.

        return target, table

    def target_to_yolo_boxes(self, target):
        yolo_boxes = []
        for cell_x in range(Config.GRID_X):
            for cell_y in range(Config.GRID_Y):
                for box in range(len(Config.ANCHORS)):
                    if (target[cell_x, cell_y, box, 4] > 0):
                        class_id = ids=np.argmax(target[cell_x, cell_y, box, 5:])
                        center_x = target[cell_x, cell_y, box, 0]
                        center_y = target[cell_x, cell_y, box, 1]
                        width = target[cell_x, cell_y, box, 2]
                        height = target[cell_x, cell_y, box, 3]

                        yolo_box = np.zeros(4)
                        yolo_box[0] = (cell_x + center_x) / Config.GRID_X
                        yolo_box[1] = (cell_y + center_y) / Config.GRID_Y
                        yolo_box[2] = width / Config.GRID_X
                        yolo_box[3] = height / Config.GRID_Y
                        yolo_boxes.append(yolo_box)

        return yolo_boxes

    def intersection_over_union(self, boxA, boxB):
        xA=np.maximum(boxA[0], boxB[0])
        yA=np.maximum(boxA[1], boxB[1])
        xB=np.minimum(boxA[2], boxB[2])
        yB=np.minimum(boxA[3], boxB[3])
        interArea=np.maximum(0, xB-xA)*np.maximum(0, yB-yA)
        boxAArea=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return interArea/(boxAArea+boxBArea-interArea)