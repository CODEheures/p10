import comet_ml
from ultralytics import YOLO
from ultralytics import settings
from app.data import Data
from app.config import Config
import os
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
#import tempfile  
import logging

class Models():

    def __init__(self, data: Data, prepare_for_train = True):
        self.loaded = {}
        self.data = data
        
        if prepare_for_train:
            self.data.prepare_data()

    def train_yolo(self, model_name='yolov3n', epochs=10, fraction=1, pretrained=False):
        experiment = 'yolo'
        comet_ml.init(project_name=experiment)

        # Create a new YOLO model from scratch
        if pretrained:
            model = YOLO(f"{model_name}.pt")
        else:
            model = YOLO(f"{model_name}.yaml")

        # Train the model using the 'coco8.yaml' dataset for 3 epochs
        results = model.train(data=self.data.dataset_yaml, epochs=epochs, batch=Config.BATCH_SIZE, imgsz=Config.BATCH_IMAGE_SIZE, pretrained=pretrained, fraction=fraction)

    def predict(self, index, model='yolov3n', version='1.0.0'):
        [image, bboxes, _, _] = self.data.get(index=index, seed=0)

        # temp_file_path = tempfile.gettempdir()  
        local_model_dir = os.path.join(model, version)
        local_model_path = os.path.join(local_model_dir, 'best.pt')
        logging.info(local_model_path)
        if not os.path.exists(local_model_path):
            comet_ml.init(project_name=model)
            api = comet_ml.API()
            model = api.get_model(
                workspace=api.get_default_workspace(), model_name=model,
            )
            Path(local_model_dir).mkdir(parents=True, exist_ok=True)
            model.download(version, output_folder=local_model_dir, expand=True)
            self.loaded[f'{model}_{version}'] = True

        model = YOLO(model=local_model_path)
        predict = model.predict(source=image)

        fig, axis = plt.subplots(1, 3, layout='constrained', sharey=True)
        fig.set_figwidth(12)
        axis[0].set_title('Image Test')
        axis[1].set_title('Image Test + Labels')
        axis[2].set_title('Prédiction')
        self.data.draw_image(image=image, ax=axis[0])
        self.data.draw_image(image=image, yolo_boxes=bboxes, ax=axis[1])
        self.data.draw_image(image=image, yolo_boxes=predict[0].boxes.xywhn.cpu().numpy(), ax=axis[2])

        if self.data.mode == 'notebook':
            fig.set_figwidth(12)
            plt.show()
        else:
            fig.set_figwidth(15)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            return buf.getbuffer().tobytes()