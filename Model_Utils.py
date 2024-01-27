import os
import pandas as pd
import PIL
from fastai.vision.all import *
from functools import partial
from sklearn.model_selection import train_test_split
from Custom_Splitter import CustomSplitter

# class CustomSplitter:
#     def __init__(self, df, valid_pct=0.2, seed=None):
#         self.train_idx, self.valid_idx = train_test_split(
#             range(len(df)), test_size=valid_pct, random_state=seed
#         )

#     def __call__(self, _):
#         # The splitter function expects a callable, so we implement the __call__ method.
#         return (list(self.train_idx), list(self.valid_idx))


# Utility class for model operations

def get_x(row, train_images='train_images'):
    return os.path.join(train_images, row['image_id'])


class ModelUtils:
    def __init__(self, model_name):
        self.model_name = model_name
        self.learn = self.load_model()

    def get_y(self, row): 
        return row['has_oilpalm']

    def create_dataloader(self, batch_size=64, train_images='train_images'):
        train_labels = 'train_labels_filtered.csv'
        # splitter = CustomSplitter(df, valid_pct=0.2, seed=42)
        palm_data_block = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=partial(get_x, train_images=train_images),
            get_y=self.get_y,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),  # Use the class-based custom splitter
            item_tfms=Resize(256),
            batch_tfms=aug_transforms()
        )
        return palm_data_block.dataloaders(pd.read_csv(train_labels), bs=batch_size)

    def load_model(self):
        learn = vision_learner(self.create_dataloader(), self.model_name, metrics=accuracy)
        path_to_model = f'{self.model_name}.pkl'
        return load_learner(path_to_model)
    
    def predict_image(self, image):
        img = PILImage.create(image)
        img = img.resize((256, 256))  # Resize to 256x256
        pred_class, _, _ = self.learn.predict(img)
        return pred_class

    def predict_image_from_path(self, image_path):
        image = PILImage.create(image_path)
        return self.predict_image(image)


