import numpy as np

class Config():
    DATA_ZIP_URL = 'https://codeheurespublic.s3.eu-west-3.amazonaws.com/ships.zip'
    ORIGINAL_DATA_DIR = 'ships-aerial-images'
    PREPARED_DATA_DIR = 'datasets'
    BATCH_SIZE = 16
    BATCH_IMAGE_SIZE = (384,384)
    GRID_X = 13
    GRID_Y = 13
    ANCHORS = np.array([[2,1], [1,1], [1,1.5]])
    CLASSES = ['ship']

    LARGEUR_CELLULE = BATCH_IMAGE_SIZE[0]/GRID_X
    HAUTEUR_CELLULE = BATCH_IMAGE_SIZE[1]/GRID_Y