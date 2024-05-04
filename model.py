!pip install ultralytics
!yolo checks
!pip install supervision

import os
import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import yaml

import cv2
import plotly.express as px
from plotly import subplots
from tqdm.notebook import tqdm
import torch
from ultralytics import YOLO

DATASET_PATH = '/kaggle/input/animals-detection-images-dataset'  # Path to source dataset
MASTER_PATH = '/kaggle/working'  # Path where all outputs are stored (intermediate and final)
DEBUG = False # Activete to run notebook faster
CPU = False

if not CPU:
    assert torch.cuda.is_available(), 'CUDA not found!'

class LookupTable:
    """Vocabulary - Label lookup table (token <-> index)."""
    def __init__(
        self,
        token_to_index: Optional[Dict[str, int]] = None,
        unknown_token: str = '<unk>',
        add_unknown_token: bool = True
    ):
        """
        Args:
            token_to_index: Predefine token to index mappings.
            unknown_token: Unknown token value.
            add_unknown_token: Use unknown token.
        """
        self._token_to_index = token_to_index
        self._unknown_token = unknown_token
        self._add_unknown_token = add_unknown_token

        if self._token_to_index is None:
            self._token_to_index = {}

        if unknown_token not in self._token_to_index and add_unknown_token:
            self._token_to_index[unknown_token] = len(self._token_to_index)

        self._index_to_token = {v: k for k, v in self._token_to_index.items()}
        self._next_index = len(self._token_to_index)

    def add(self, token: str) -> int:
        """
        Adds token to the lookup table if it does not already exist.
        
        Args:
            token: Label (token)
            
        Returns:
            label (token) index
        """
        if token in self._token_to_index:
            return self._token_to_index[token]

        new_index = self._next_index
        self._next_index += 1
        self._token_to_index[token] = new_index
        self._index_to_token[new_index] = token
        return new_index

    def lookup(self, token: str) -> int:
        """
        Acquires token index if it exists in the table.
        In case the token does not exist in the lookup table:
            - and unknown token is used then unkown token index is returned;
            - otherwise KeyError is raised
            
        Raises:
            KeyError: Unknown token
            
        Returns:
            label (token) index
        """
        if token not in self._token_to_index and self._add_unknown_token:
            return self._token_to_index[self._unknown_token]

        return self._token_to_index[token]

    def inverse_lookup(self, index: int) -> str:
        """
        Inverse to `lookup`. Acquire token by index.
        
        Raises:
            KeyError: Unknown index
            
        Returns:
            label (token)
        """
        return self._index_to_token[index]
    
    def __iter__(self) -> Iterable[Tuple[str, int]]:
        for token, index in self._token_to_index.items():
            yield token, index

    def __getitem__(self, token: str) -> int:
        return self.lookup(token)  # Alias for `lookup`

    def __len__(self):
        return self._next_index
DatasetIndex = Dict[str, Dict[str, List[str]]]
DatasetStats = Dict[str, int]


class AnimalToYOLODatasetAdapter:
    """Adapts custom animal dataset to YOLO format."""
    def __init__(self, path: str, label_filter: Optional[List[str]] = None):
        """
        Args:
            path: Path where dataset is stored
            label_filter: Use specific set of labels (remove others from dataset)
        """
        self._path = path
        
        self._index, self.label_stats, self.split_stats, self.label_lookup, self._size = \
            self._index_dataset(path, label_filter)
        
    @staticmethod
    def _index_dataset(path: str, label_filter: Optional[List[str]] = None) \
        -> Tuple[DatasetIndex, DatasetStats, DatasetStats, LookupTable, int]:
        """
        Creates datast index. Index is mapping (split -> label -> sample_id). 
        Input dataset format is given in previosly defined structure.

        Args:
            path: Dataset path
            label_filter: Filter used labels

        Returns:
            Dataset index, Label stats, Split stats, dataset size
        """
        index: DatasetIndex = defaultdict(dict)
        label_stats: DatasetStats = Counter()
        split_stats: DatasetStats = Counter()
        lookup = LookupTable(add_unknown_token=False)
        size = 0

        splits = os.listdir(path)
        for split in splits:        
            split_path = os.path.join(path, split)
            labels = os.listdir(split_path)
            for label in tqdm(labels, desc=f'Indexing {split}', unit='sample'):
                if label_filter is not None and label not in label_filter:
                    continue
                
                label_path = os.path.join(split_path, label)
                sample_ids = [Path(filename).stem for filename in os.listdir(label_path) 
                              if filename != 'Label' and filename.endswith('.jpg')]
                annotations_path = os.path.join(label_path, 'Label')
                annot_sample_ids = [Path(filename).stem for filename in os.listdir(annotations_path)
                                    if filename.endswith('.txt')]
                assert set(sample_ids) == set(annot_sample_ids), 'Image sample ids and annotation sample ids do not match'

                # Update index, stats and lookup
                index[split][label] = sample_ids
                
                n_samples = len(sample_ids)
                label_stats[label] += n_samples
                split_stats[split] += n_samples
                size += n_samples
                
                lookup.add(label)

        return dict(index), dict(label_stats), dict(split_stats), lookup, size
    
    def __len__(self) -> int:
        return self._size
    
    @property
    def labels(self) -> List[str]:
        """
        Returns:
            List of labels (classes) in lookup table
        """
        return list(self.label_lookup)

    @property
    def n_labels(self) -> int:
        """
        Returns:
            Number of labels (classes) in lookup table
        """
        return len(self.label_lookup)
    
    def get_random_samples(self, n: int, split: str = 'train') -> List[Tuple[str, str, str]]:
        """
        Fetchen `n` random samples from dataset for chosen split.
        
        Args:
            n: Number of samples
            split: chosen split
            
        Returns:
            List of tuples (split, label, sample_id)
        """
        split_index = self._index[split]
        label_names, _ = zip(*self.labels)
        
        result: List[Tuple[str, str, str]] = []
        for i in range(n):
            label = random.choice(label_names)
            sample_ids = split_index[label]
            sample_id = random.choice(sample_ids)
            result.append((split, label, sample_id))
            
        return result
    
    def get_split_size(self, split: str) -> int:
        """
        Returns:
            Number of samples in split
        """
        return self.split_stats[split]

    def get_image_path(self, split: str, label: str, sample_id: str) -> str:
        """
        Animal dataset image path convention.
        
        Args:
            split: Split
            label: Label (token)
            sample_id: Sample id
        
        Returns:
            Image path
        """
        return os.path.join(self._path, split, label, f'{sample_id}.jpg')

    def load_image(self, split: str, label: str, sample_id: str) -> str:
        """        
        Args:
            split: Split
            label: Label (token)
            sample_id: Sample id
        
        Returns:
            Loaded image
        """
        image_path = self.get_image_path(split, label, sample_id)
        if not os.path.exists(image_path):
            raise FileNotFound(f'Image "{image_path}" not found!')
        return cv2.imread(image_path)

    def get_annot_path(self, split: str, label: str, sample_id: str) -> str:
        """
        Animal dataset annotation path convention.
        
        Args:
            split: Split
            label: Label (token)
            sample_id: Sample id
        
        Returns:
            Annotation path
        """
        return os.path.join(self._path, split, label, 'Label', f'{sample_id}.txt')

    def parse_annot(self, split: str, label: str, sample_id: str) \
        -> List[Tuple[str, float, float, float, float]]:
        """        
        Parses annotation (ground truth) file.
        
        Args:
            split: Split
            label: Label (token)
            sample_id: Sample id
        
        Returns:
            Parsed annotations
        """
        annot_path = self.get_annot_path(split, label, sample_id)
        with open(annot_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        annots: List[Tuple[str, float, float, float, float]] = []
        for l in lines:
            items = l.split()
            label_name = ' '.join(items[:-4])
            coords = [float(v) for v in items[-4:]]
            annots.append([label_name, *coords])
        return annots
    
    def convert(self, path: str) -> None:
        """
        Converts dataset tp YOLO format.
        
        Args:
            path: Output path
        """
        for split in self._index:
            split_path = os.path.join(path, split)
            images_path = os.path.join(split_path, 'images')
            labels_path = os.path.join(split_path, 'labels')
            Path(images_path).mkdir(parents=True, exist_ok=True)
            Path(labels_path).mkdir(parents=True, exist_ok=True)
            
            for label, sample_ids in tqdm(self._index[split].items(), desc='Converting to Yolo format', total=len(self._index[split])):
                assert len(sample_ids) == len(set(sample_ids))
                for sample_id in sample_ids:
                    image_path = self.get_image_path(split, label, sample_id)
                    new_image_path = os.path.join(images_path, f'{sample_id}.jpg')
                    annots = self.parse_annot(split, label, sample_id)
                    new_annot_path = os.path.join(labels_path, f'{sample_id}.txt')
                    
                    # Image needs to be loaded in order to read width and height
                    # which are required for coordinate normalization
                    image = self.load_image(split, label, sample_id)
                    h, w, _ = image.shape
                    
                    # Conversion
                    converted_annot: List[Tuple[int, float, float, float, float]] = []
                    for label, x_min, y_min, x_max, y_max in annots:
                        label_index = self.label_lookup[label]
                        x_center = (x_min + x_max) / (2 * w)
                        y_center = (y_min + y_max) / (2 * h)
                        width = (x_max - x_min) / w
                        height = (y_max - y_min) / h
                        
                        converted_annot.append((label_index, x_center, y_center, width, height))
                        
                    # Save data
                    with open(new_annot_path, 'a', encoding='utf-8') as f:
                        converted_annot_lines = [' '.join([str(v) for v in row]) for row in converted_annot]
                        f.write('\n'.join(converted_annot_lines))
                        f.write('\n')
                        
                    if not os.path.exists(new_image_path):  
                        shutil.copy(image_path, new_image_path)


adapter = AnimalToYOLODatasetAdapter(
    path=DATASET_PATH, 
    label_filter=['Horse'] if DEBUG else None
)

print(f'Total number of samples in the dataset is {len(adapter)}.')
print(f'Total number of classes in the dataset is {adapter.n_labels}.')
print(f'Train dataset size is {adapter.get_split_size("train")} (images). Test dataset size is {adapter.get_split_size("test")} (images)')

fig = px.histogram(x=list(adapter.label_stats.keys()), y=list(adapter.label_stats.values())) \
        .update_layout(xaxis_title="Class", yaxis_title="Class size", xaxis={'categoryorder':'total descending'})
fig.show()


def visualize_samples(
    adapter: AnimalToYOLODatasetAdapter,
    n_rows: int,
    n_cols: int,
    bbox_color: Tuple[int, int, int] = (255, 0, 0),  # RBG - RED
    model: Optional[YOLO] = None
) -> None:
    """
    Visualizes image sample with ground truths and (optionally) model predictions.
    Number of images is equal to product of `n_rows` and `n_cols`
    
    Args:
        adapter: Animal dataset to YOLO adapter
        n_rows: Number of rows in image matrix
        n_cols: Number of cols in image matrix
        bbox_color: Ground truth bbox color
        model: Model to generate prediction for given images
    """
    n: int = n_rows * n_cols
    
    viz_samples = adapter.get_random_samples(n)
    fig = subplots.make_subplots(rows=n_rows, cols=n_cols)
    for plot_index, (split, label, sample_id) in enumerate(viz_samples):
        row = plot_index // n_cols + 1
        col = plot_index % n_cols + 1
        image = adapter.load_image(split, label, sample_id)
        label_index = adapter.label_lookup.lookup(label)
        
        # Visualize ground truth
        for _, x_min, y_min, x_max, y_max in adapter.parse_annot(split, label, sample_id):
            x_min, y_min, x_max, y_max = [int(v) for v in [x_min, y_min, x_max, y_max]]
            labek_text = f'{label} ({label_index})'
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=bbox_color, thickness=4)
            image = cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, bbox_color, 3)
            
        if model is not None:
            # Visualize model predictions
            prediction = model.predict([image], imgsz=512, conf=0.3)
            for p in prediction:
                image = p.plot()
            
        subfig = px.imshow(image)
        fig.add_trace(subfig.data[0], row=row, col=col)

    fig = fig.update_xaxes(showticklabels=False)
    fig = fig.update_yaxes(showticklabels=False)
    fig = fig.update_layout(
        autosize=False,
        width=1200,
        height=600
    )
    fig.show()
    
visualize_samples(adapter, 1, 3)

# Identify the specific classes to use
specific_classes = ['Cattle','Pig','Horse','Sheep','Goat','Chicken','Mule','Duck','Fox','Raccoon','Bear','Owl']

# Filter the class_names list
filtered_class_names = [name for name in adapter.label_lookup if name in specific_classes]

# Update the config dictionary
config = {
    'path': MASTER_PATH,
    'train': 'train/images',
    'val': 'test/images',
    'nc': len(filtered_class_names),  # number of remaining classes
    'names': filtered_class_names
}

# Regenerate the configuration file
config_path = os.path.join(MASTER_PATH, 'config.yaml')
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f)

# class_names = [name for name, _ in adapter.label_lookup]
# config = {
#     'path': MASTER_PATH,
#     'train': 'train/images',
#     'val': 'test/images',
#     'nc': len(adapter.label_lookup),  # number of classes
#     'names': class_names
# }

# config_path = os.path.join(MASTER_PATH, 'config.yaml')
# with open(config_path, 'w', encoding='utf-8') as f:
#     yaml.dump(config, f)

# print(yaml.dump(config))

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the processed dataset
results = model.train(
    data='config.yaml', 
    epochs=5 if not DEBUG else 1,
    optimizer='Adam',
    val=True,
    batch=64,
    imgsz=640,
    device=[0] if not CPU else 'cpu',
    lr0=0.001,
    lrf=0.0005
    metrics=["accuracy"]
)

# Evaluate the model's performance on the validation set
results = model.val()

print(results)




# Convert the DetMetrics object to a list of dictionaries
val_results_list = results.process_results()

# Calculate the average precision (AP) for each class
classwise_ap = []
for val_result in val_results_list:
    precision = val_result['precision']
    recall = val_result['recall']
    ap = calculate_ap(precision, recall)
    classwise_ap.append(ap)

# Calculate the mean average precision (mAP)
mAP = np.mean(classwise_ap)

# Calculate the overall accuracy
correct = 0
total = 0
for val_result in val_results_list:
    gt_labels = val_result['gt_labels']
    pred_labels = val_result['pred_labels']
    correct += np.sum(gt_labels == pred_labels)
    total += len(gt_labels)

accuracy = correct / total * 100

print('Mean average precision (mAP): {:.4f}'.format(mAP))
print('Overall accuracy: {:.2f}%'.format(accuracy))
import supervision as sv
from ultralytics import YOLO
import numpy as np
def callback(image: np.ndarray) -> sv.Detections:
    result = model(image)[0]
    return sv.Detections.from_ultralytics(result)

confusion_matrix = sv.ConfusionMatrix.benchmark(
   dataset = DATASET_PATH,
   callback = callback
)

confusion_matrix.plot()

import torch
import numpy as np

def calculate_map(preds, targets):
    """
    Calculates mean average precision (mAP) for object detection.

    Args:
        preds (list): List of predicted bounding boxes and corresponding class labels.
        targets (list): List of ground truth bounding boxes and corresponding class labels.

    Returns:
        float: Mean average precision (mAP).
    """

    # Calculate AP for each class
    ap_list = []
    for i in range(len(preds)):
        pred_boxes = preds[i][0]
        pred_labels = preds[i][1]
        gt_boxes = targets[i][0]
        gt_labels = targets[i][1]

        # Calculate intersection over union (IoU) between predicted and ground truth boxes
        ious = calculate_ious(pred_boxes, gt_boxes)

        # Sort IoU in descending order
        sort_idx = np.argsort(ious)[:-1]
        ious = ious[sort_idx]
        pred_labels = pred_labels[sort_idx]

        # Calculate precision and recall
        precision = np.zeros(len(ious))
        recall = np.zeros(len(ious))

        for j in range(len(ious)):
            if pred_labels[j] == 1:
                precision[j] = (j + 1) / (j + 1)
                recall[j] = j / (len(ious) - 1)
            else:
                precision[j] = 0
                recall[j] = 0

        # Calculate average precision (AP)
        ap = calculate_ap(precision, recall)
        ap_list.append(ap)

    # Calculate mean average precision (mAP)
    mAP = sum(ap_list) / len(ap_list)
    return mAP
visualize_samples(adapter, 2, 3, model=model)
import supervision as sv
from ultralytics import YOLO

dataset = sv.DetectionDataset.from_yolo('/kaggle/input/animals-detection-images-dataset')

model = YOLO(`YOLOv8-n-baseline`)
def callback(image: np.ndarray) -> sv.Detections:
    result = model(image)[0]
    return sv.Detections.from_ultralytics(result)

confusion_matrix = sv.ConfusionMatrix.benchmark(
   dataset = dataset,
   callback = callback
)

confusion_matrix.plot()

model.export(format='engine', half=True)

from ultralytics.yolo.utils.benchmarks import benchmark

benchmark(model=model, imgsz=640, half=True, device=0 if not CPU else 'cpu')