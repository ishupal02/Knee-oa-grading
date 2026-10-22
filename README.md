# Simple Knee OA Classifier

One file, one job: classify knee X-rays into KL grades 0-4.

## Setup

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity

   (Click "Download" on that page -- you'll need a free Kaggle account.
   No API/token setup required if you just download the zip manually.)

2. Unzip it, and make sure you end up with this structure next to `train.py`:

   ```
   dataset/
       train/
           0/ 1/ 2/ 3/ 4/
       val/
           0/ 1/ 2/ 3/ 4/
       test/
           0/ 1/ 2/ 3/ 4/
   ```

   (The zip sometimes nests folders one level deeper -- if so, move the
   train/val/test folders up so they sit directly inside `dataset/`.)

3. Install dependencies:
   ```
   pip install torch torchvision
   ```

4. Run:
   ```
   python train.py
   ```

That's it. It fine-tunes a pretrained ResNet-18 for 5 epochs and prints
train loss + validation accuracy each epoch, then final test accuracy at
the end. Takes a few minutes on a GPU (Colab's free tier works fine), longer
on CPU.

## Want to tweak it?

Everything adjustable is at the top of `train.py`:
- `NUM_EPOCHS` -- train longer for better accuracy
- `BATCH_SIZE` -- lower this if you run out of memory
- `LEARNING_RATE` -- lower if loss is unstable/NaN
