import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

def preprocess_mammogram(image, size=512):
    """Deterministic preprocessing: resize/pad, Wiener filtering, CLAHE,
    breast-region thresholding and per-image z-score normalization."""
    image = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = image.shape
    scale = min(size / h, size / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y+nh, x:x+nw] = image
    denoised = cv2.medianBlur(canvas, 3)  # stable default; replace with Wiener if matching exact pipeline
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    # Keep the largest non-background component as a coarse breast field.
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        field = (labels == idx)
        enhanced = enhanced * field.astype(np.uint8)
    arr = enhanced.astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return np.clip(arr, -5, 5)

class MammographyCSVDataset(Dataset):
    """
    CSV columns:
      image, mask, label, case_id
    mask may be blank for classification-only examples; label must be 0/1.
    Provide patient/case-level splits in separate CSVs before training.
    """
    def __init__(self, csv_file, image_size=512, augment=False):
        self.df = pd.read_csv(csv_file).fillna("")
        required = {"image", "label", "case_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        self.image_size, self.augment = image_size, augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = preprocess_mammogram(Image.open(row.image).convert("L"), self.image_size)
        mask_path = row.get("mask", "")
        if mask_path and str(mask_path).strip():
            mask = np.asarray(Image.open(mask_path).convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST))
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros_like(image, dtype=np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if np.random.rand() < 0.5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()

        image = torch.from_numpy(image[None]).float().repeat(3, 1, 1)
        mask = torch.from_numpy(mask[None]).float()
        label = torch.tensor(int(row.label), dtype=torch.long)
        return {"image": image, "mask": mask, "label": label, "case_id": str(row.case_id)}
