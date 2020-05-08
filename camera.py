import os
import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, sampler
from transform import get_train_transforms, get_test_transforms, CLAHE_GRAY
from PIL import Image
from skimage import io
from skimage.transform import resize
import cv2

def preprocess(path):
    if not os.path.exists(f"{path}/train_gray.p"):
        for dataset in ['train', 'valid', 'test']:
            with open(f"{path}/{dataset}.p", mode='rb') as f:
                data = pickle.load(f)
                X = data['features']
                y = data['labels']

            clahe = CLAHE_GRAY()
            for i in tqdm(range(len(X)), desc=f"Processing {dataset} dataset"):
                X[i] = clahe(X[i])

            print("before preprocess X shape and size")
            print(X.shape)
            print(len(X))
            X = X[:, :, :, 0]
            print("after")
            print(X.shape)
            print(len(X))
            with open(f"{path}/{dataset}_gray.p", "wb") as f:
                pickle.dump({"features": X.reshape(
                    X.shape + (1,)), "labels": y}, f)

for i in range(5): 
  path = '/root/final/final_8M/' + str(i) + '.jpg'
  a = Image.open(path)
  a = a.resize((32,32))
  a = np.asarray(a)
  print(a.size,a.shape)
  clahe = CLAHE_GRAY()
  a = clahe(a)
  path1 = '/root/final/final_8M/' + str(i) + '_.jpg'
  cv2.imwrite(path1, a)
  device = torch.device('cuda')
  model = torch.load('model_after_weight_sharing.ptmodel').to(device)
  c = []
  c.append(a)
  c = np.array(c)
  a_tensor = torch.from_numpy(np.transpose(c,(0,3,1,2))).float().to(device)
  pred = torch.argmax(model(a_tensor),dim = 1)
  print(pred)
  print(model(a_tensor))

