Python 3.6.2 (v3.6.2:5fd33b5, Jul  8 2017, 04:57:36) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "42de181a-f187-461c-8ebf-7e9898a0de7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm import tqdm\n",
    "import joblib\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "import cv2\n",
    "import seaborn as sns\n",
    "import time\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "144ffb61-d631-4827-b8f4-16ce7e6973db",
   "metadata": {},
   "outputs": [],
   "source": [
    "folder_path = f\"Dataset/\"\n",
    "os.makedirs(folder_path, exist_ok=True)\n",
    "\n",
    "confusion_image_path = os.path.join(folder_path, 'confusion matrix.png')\n",
    "classification_file_path = os.path.join(folder_path, 'classification_report.txt')\n",
    "model_file_path = os.path.join(folder_path, \"svm_model.pkl\")\n",
    "\n",
    "dataset_dir = \"Dataset/\"\n",
    "train_dir = os.path.join(dataset_dir, \"train\")\n",
    "test_dir = os.path.join(dataset_dir, \"test1\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "234b4192-2201-4039-9877-2d8b5db04215",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Processing Train Images: 100%|████████████| 25000/25000 [05:46<00:00, 72.11it/s]\n"
     ]
    }
   ],
   "source": [
    "train_images = os.listdir(train_dir)\n",
    "features = []\n",
    "labels = []\n",
    "image_size = (50, 50)\n",
    "\n",
    "for image in tqdm(train_images, desc=\"Processing Train Images\"):\n",
    "    if image[0:3] == 'cat' :\n",
    "        label = 0\n",
    "    else :\n",
    "        label = 1\n",
    "    image_read = cv2.imread(train_dir+\"/\"+image)\n",
    "    image_resized = cv2.resize(image_read, image_size)\n",
    "    image_normalized = image_resized / 255.0\n",
    "    image_flatten = image_normalized.flatten()\n",
    "    features.append(image_flatten)\n",
    "    labels.append(label)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "52000794-e32b-4dd3-8362-4fe656a74a71",
   "metadata": {},
   "outputs": [],
   "source": [
    "del train_images\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6d6657bf-5fad-456a-801f-ad2571ad1b2b",
   "metadata": {},
   "outputs": [],
   "source": [
    "features = np.asarray(features)\n",
    "labels = np.asarray(labels)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7ab27514-5d58-4bb2-b72c-9058385480fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "del features\n",
    "del labels\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "aca96a63-dfc6-4400-b252-fa23441d75cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "n_components = 0.8\n",
    "pca = PCA(n_components=n_components)\n",
    "svm = SVC()\n",
    "pca = PCA(n_components=n_components, random_state=42)\n",
    "pipeline = Pipeline([\n",
    "    ('pca', pca),\n",
    "    ('svm', svm)\n",
    "])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "de467cd6-6b6a-45e6-8442-27e55c67da0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "param_grid = {\n",
    "    'pca__n_components': [2, 1, 0.9, 0.8],\n",
    "    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a16065ea-8ea1-49c5-9959-2fdb8d99a672",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 16 candidates, totalling 48 fits\n",
      "[CV 1/3] END pca__n_components=2, svm__kernel=linear;, score=0.529 total time=  15.0s\n",
      "[CV 2/3] END pca__n_components=2, svm__kernel=linear;, score=0.522 total time=  13.9s\n",
      "[CV 3/3] END pca__n_components=2, svm__kernel=linear;, score=0.539 total time=  14.5s\n",
      "[CV 1/3] END pca__n_components=2, svm__kernel=rbf;, score=0.572 total time=  10.0s\n",
      "[CV 2/3] END pca__n_components=2, svm__kernel=rbf;, score=0.560 total time=  10.0s\n",
      "[CV 3/3] END pca__n_components=2, svm__kernel=rbf;, score=0.564 total time=   9.8s\n",
      "[CV 1/3] END pca__n_components=2, svm__kernel=poly;, score=0.494 total time=   8.9s\n",
      "[CV 2/3] END pca__n_components=2, svm__kernel=poly;, score=0.492 total time=   8.8s\n",
      "[CV 3/3] END pca__n_components=2, svm__kernel=poly;, score=0.498 total time=   8.6s\n",
      "[CV 1/3] END pca__n_components=2, svm__kernel=sigmoid;, score=0.502 total time=   9.5s\n",
      "[CV 2/3] END pca__n_components=2, svm__kernel=sigmoid;, score=0.508 total time=   9.7s\n",
      "[CV 3/3] END pca__n_components=2, svm__kernel=sigmoid;, score=0.510 total time=  10.2s\n",
      "[CV 1/3] END pca__n_components=1, svm__kernel=linear;, score=0.521 total time=   9.9s\n",
      "[CV 2/3] END pca__n_components=1, svm__kernel=linear;, score=0.517 total time=   9.4s\n",
      "[CV 3/3] END pca__n_components=1, svm__kernel=linear;, score=0.518 total time=   9.7s\n",
      "[CV 1/3] END pca__n_components=1, svm__kernel=rbf;, score=0.527 total time=  10.0s\n",
      "[CV 2/3] END pca__n_components=1, svm__kernel=rbf;, score=0.527 total time=  10.5s\n",
      "[CV 3/3] END pca__n_components=1, svm__kernel=rbf;, score=0.527 total time=  10.3s\n",
      "[CV 1/3] END pca__n_components=1, svm__kernel=poly;, score=0.501 total time=   8.2s\n",
      "[CV 2/3] END pca__n_components=1, svm__kernel=poly;, score=0.496 total time=   8.2s\n",
      "[CV 3/3] END pca__n_components=1, svm__kernel=poly;, score=0.500 total time=   8.2s\n",
      "[CV 1/3] END pca__n_components=1, svm__kernel=sigmoid;, score=0.502 total time=   8.8s\n",
      "[CV 2/3] END pca__n_components=1, svm__kernel=sigmoid;, score=0.508 total time=   8.8s\n",
      "[CV 3/3] END pca__n_components=1, svm__kernel=sigmoid;, score=0.504 total time=   8.9s\n",
      "[CV 1/3] END pca__n_components=0.9, svm__kernel=linear;, score=0.602 total time=15.2min\n",
      "[CV 2/3] END pca__n_components=0.9, svm__kernel=linear;, score=0.613 total time=15.5min\n",
      "[CV 3/3] END pca__n_components=0.9, svm__kernel=linear;, score=0.610 total time=15.5min\n",
      "[CV 1/3] END pca__n_components=0.9, svm__kernel=rbf;, score=0.675 total time= 6.4min\n",
      "[CV 2/3] END pca__n_components=0.9, svm__kernel=rbf;, score=0.674 total time= 6.4min\n",
      "[CV 3/3] END pca__n_components=0.9, svm__kernel=rbf;, score=0.673 total time= 6.5min\n",
      "[CV 1/3] END pca__n_components=0.9, svm__kernel=poly;, score=0.612 total time= 6.4min\n",
      "[CV 2/3] END pca__n_components=0.9, svm__kernel=poly;, score=0.607 total time= 6.5min\n",
      "[CV 3/3] END pca__n_components=0.9, svm__kernel=poly;, score=0.600 total time= 6.5min\n",
      "[CV 1/3] END pca__n_components=0.9, svm__kernel=sigmoid;, score=0.518 total time= 6.1min\n",
      "[CV 2/3] END pca__n_components=0.9, svm__kernel=sigmoid;, score=0.530 total time= 6.2min\n",
      "[CV 3/3] END pca__n_components=0.9, svm__kernel=sigmoid;, score=0.524 total time= 6.1min\n",
      "[CV 1/3] END pca__n_components=0.8, svm__kernel=linear;, score=0.585 total time= 8.9min\n",
      "[CV 2/3] END pca__n_components=0.8, svm__kernel=linear;, score=0.591 total time= 8.8min\n",
      "[CV 3/3] END pca__n_components=0.8, svm__kernel=linear;, score=0.594 total time= 9.0min\n",
      "[CV 1/3] END pca__n_components=0.8, svm__kernel=rbf;, score=0.665 total time= 5.9min\n",
      "[CV 2/3] END pca__n_components=0.8, svm__kernel=rbf;, score=0.662 total time= 5.9min\n",
      "[CV 3/3] END pca__n_components=0.8, svm__kernel=rbf;, score=0.667 total time= 5.9min\n",
      "[CV 1/3] END pca__n_components=0.8, svm__kernel=poly;, score=0.595 total time= 5.8min\n",
      "[CV 2/3] END pca__n_components=0.8, svm__kernel=poly;, score=0.600 total time= 5.9min\n",
      "[CV 3/3] END pca__n_components=0.8, svm__kernel=poly;, score=0.593 total time= 5.9min\n",
      "[CV 1/3] END pca__n_components=0.8, svm__kernel=sigmoid;, score=0.516 total time= 5.8min\n",
      "[CV 2/3] END pca__n_components=0.8, svm__kernel=sigmoid;, score=0.525 total time= 5.8min\n",
      "[CV 3/3] END pca__n_components=0.8, svm__kernel=sigmoid;, score=0.521 total time= 5.8min\n"
     ]
    }
   ],
   "source": [
    "start_time = time.time()\n",
    "\n",
    "grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=4)\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "end_time = time.time()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4dd18f82-8cc8-4b95-8ab2-371a18506135",
   "metadata": {},
   "outputs": [],
   "source": [
    "del X_train\n",
    "del y_train\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "06e92bd5-fafc-46b0-be1c-a273a39259ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Parameters:  {'pca__n_components': 0.9, 'svm__kernel': 'rbf'}\n",
      "Best Score:  0.673899930866043\n"
     ]
    }
   ],
   "source": [
    "best_pipeline = grid_search.best_estimator_\n",
    "best_params = grid_search.best_params_\n",
    "best_score = grid_search.best_score_\n",
    "\n",
    "print(\"Best Parameters: \", best_params)\n",
    "print(\"Best Score: \", best_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b3dac191-3086-4bdd-ab86-8de0b4726983",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.6816\n"
     ]
    }
   ],
   "source": [
    "accuracy = best_pipeline.score(X_test, y_test)\n",
    "print(\"Accuracy:\", accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "572c48f2-6315-44bf-aa0c-bcec85acafb6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "         Cat       0.69      0.68      0.68      2514\n",
      "         Dog       0.68      0.69      0.68      2486\n",
      "\n",
      "    accuracy                           0.68      5000\n",
      "   macro avg       0.68      0.68      0.68      5000\n",
      "weighted avg       0.68      0.68      0.68      5000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = best_pipeline.predict(X_test)\n",
    "target_names = ['Cat', 'Dog']\n",
    "classification_rep = classification_report(y_test, y_pred, target_names=target_names)\n",
    "print(\"Classification Report:\\n\", classification_rep)\n",
    "\n",
    "with open(classification_file_path, 'w') as file:\n",
    "    file.write(classification_rep)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4ecdf5c9-b60a-4fed-a349-67031ac898a3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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s6a8vb2VnZ3tcj47O1uhoaEWVQUAKM8IJPA4X19fdejQQevXr3ees9vtWr9+vSIiIiysDABQXvlYXQAqpgkTJigmJkYdO3bUTTfdpBdffFH5+fkaPny41aUBlsnLy9O+ffucrw8cOKD09HQFBwerfv36FlYGWI9tvzDNggULNHv2bGVlZaldu3ZKSEhQeHi41WUBltm4caO6d+9e4nxMTIySkpKufEFAOUIgAQAAlmMNCQAAsByBBAAAWI5AAgAALEcgAQAAliOQAAAAyxFIAACA5QgkAADAcgQSAABgOQIJYKH77rtPAwYMcL7u1q2bHn300Stex8aNG2Wz2XTy5MmLjrHZbFq9enWZ7zl9+nS1a9fuD9X1008/yWazKT09/Q/dB0D5RyABfue+++6TzWaTzWaTr6+vGjdurKeeekrnzp0z/bNXrlypmTNnlmlsWUIEAFwt+HI9oBS9evXSG2+8ocLCQn388ccaM2aMKlWqpKlTp5YYe/bsWfn6+nrkc4ODgz1yHwC42tAhAUrh5+en0NBQNWjQQA899JAiIyP14YcfSvrvNMszzzyjsLAwNWvWTJJ0+PBhDRkyREFBQQoODlb//v31008/Oe9ZXFysCRMmKCgoSDVq1NDjjz+u33+V1O+nbAoLCzV58mTVq1dPfn5+aty4sV5//XX99NNPzi9pq169umw2m+677z5Jkt1uV3x8vBo1aiR/f3+1bdtW7733nsvnfPzxx2ratKn8/f3VvXt3lzrLavLkyWratKmqVKmia6+9VnFxcSoqKiox7p///Kfq1aunKlWqaMiQITp16pTL9ddee00tWrRQ5cqV1bx5c7388ssX/cwTJ04oOjpatWrVkr+/v5o0aaI33njD7doBlD90SIAy8Pf312+//eZ8vX79egUEBCg5OVmSVFRUpKioKEVERGjTpk3y8fHR008/rV69emnXrl3y9fXVCy+8oKSkJC1evFgtWrTQCy+8oFWrVumvf/3rRT/33nvvVWpqqhISEtS2bVsdOHBAv/76q+rVq6f3339fgwYNUmZmpgICAuTv7y9Jio+P11tvvaXExEQ1adJEKSkpuvvuu1WrVi117dpVhw8f1sCBAzVmzBiNHj1a27Zt02OPPeb276RatWpKSkpSWFiYdu/erVGjRqlatWp6/PHHnWP27dund955R2vWrFFubq5Gjhyphx9+WMuWLZMkLVu2TNOmTdOCBQvUvn177dixQ6NGjVLVqlUVExNT4jPj4uL03Xff6ZNPPlHNmjW1b98+nTlzxu3aAZRDDgAuYmJiHP3793c4HA6H3W53JCcnO/z8/BwTJ050Xg8JCXEUFhY63/Pmm286mjVr5rDb7c5zhYWFDn9/f8enn37qcDgcjjp16jhmzZrlvF5UVOT4y1/+4vwsh8Ph6Nq1q+ORRx5xOBwOR2ZmpkOSIzk5udQ6P//8c4ckx4kTJ5znCgoKHFWqVHFs3rzZZezIkSMdd911l8PhcDimTp3qaNmypcv1yZMnl7jX70lyrFq16qLXZ8+e7ejQoYPz9ZNPPunw9vZ2/Pzzz85zn3zyicPLy8tx7Ngxh8PhcFx33XWO5cuXu9xn5syZjoiICIfD4XAcOHDAIcmxY8cOh8PhcPTt29cxfPjwi9YA4OpFhwQoxdq1a3XNNdeoqKhIdrtdw4YN0/Tp053XW7du7bJuZOfOndq3b5+qVavmcp+CggLt379fp06d0rFjxxQeHu685uPjo44dO5aYtrkgPT1d3t7e6tq1a5nr3rdvn06fPq2//e1vLufPnj2r9u3bS5IyMjJc6pCkiIiIMn/GBStWrFBCQoL279+vvLw8nTt3TgEBAS5j6tevr7p167p8jt1uV2ZmpqpVq6b9+/dr5MiRGjVqlHPMuXPnFBgYWOpnPvTQQxo0aJC2b9+unj17asCAAbrlllvcrh1A+UMgAUrRvXt3LVq0SL6+vgoLC5OPj+sflapVq7q8zsvLU4cOHZxTEf+rVq1al1XDhSkYd+Tl5UmSPvroI5cgIJ1fF+Mpqampio6O1owZMxQVFaXAwEC9/fbbeuGFF9yu9dVXXy0RkLy9vUt9T+/evXXw4EF9/PHHSk5OVo8ePTRmzBjNmTPn8n8YAOUCgQQoRdWqVdW4ceMyj7/hhhu0YsUK1a5du0SX4II6depoy5Yt6tKli6TznYC0tDTdcMMNpY5v3bq17Ha7vvjiC0VGRpa4fqFDU1xc7DzXsmVL+fn56dChQxftrLRo0cK5QPeCr7/+2viH/B+bN29WgwYN9MQTTzjPHTx4sMS4Q4cO6ejRowoLC3N+jpeXl5o1a6aQkBCFhYXpxx9/VHR0dJk/u1atWoqJiVFMTIw6d+6sSZMmEUiACoBdNoAHREdHq2bNmurfv782bdqkAwcOaOPGjRo3bpx+/vlnSdIjjzyi5557TqtXr9b333+vhx9++JLPEGnYsKFiYmI0YsQIrV692nnPd955R5LUoEED2Ww2rV27Vr/88ovy8vJUrVo1TZw4UePHj9eSJUu0f/9+bd++XfPnz9eSJUskSQ8++KB++OEHTZo0SZmZmVq+fLmSkpLc+nmbNGmiQ4cO6e2339b+/fuVkJCgVatWlRhXuXJlxcTEaOfOndq0aZPGjRunIUOGKDQ0VJI0Y8YMxcfHKyEhQXv37tXu3bv1xhtvaO7cuaV+7rRp0/TBBx9o37592rNnj9auXasWLVq4VTuA8olAAnhAlSpVlJKSovr162vgwIFq0aKFRo4cqYKCAmfH5LHHHtM999yjmJgYRUREqFq1arrjjjsued9FixZp8ODBevjhh9W8eXONGjVK+fn5kqS6detqxowZmjJlikJCQhQbGytJmjlzpuLi4hQfH68WLVqoV69e+uijj9SoUSNJ59d1vP/++1q9erXatm2rxMREPfvss279vP369dP48eMVGxurdu3aafPmzYqLiysxrnHjxho4cKD69Omjnj17qk2bNi7beu+//3699tpreuONN9S6dWt17dpVSUlJzlp/z9fXV1OnTlWbNm3UpUsXeXt76+2333ardgDlk81xsRV1AAAAVwgdEgAAYDkCCQAAsByBBAAAWI5AAgAALEcgAQAAliOQAAAAyxFIAACA5QgkAADAcgQSAABgOQIJAACwHIEEAABY7v8D+4+n3fiZ67QAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test, y_pred)\n",
    "sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\")\n",
    "plt.xlabel('Predicted labels')\n",
    "plt.ylabel('True labels')\n",
    "plt.savefig(confusion_image_path)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b63c2c8a-2a56-49f5-8ae2-562c5a78e0b3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
