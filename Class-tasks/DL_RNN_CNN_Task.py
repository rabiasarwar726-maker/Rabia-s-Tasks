"""Deep Learning Task: RNN (IMDB) and CNN (CIFAR-10)

Run in VS Code: open the file and either click "Run Cell" above each
`# %%` marker (needs the Python + Jupyter extensions), or run the whole
file with `python DL_RNN_CNN_Task.py`.

Requirements: pip install tensorflow numpy matplotlib
Note: when run as a normal script, close each plot window to continue.
"""

# %% [markdown]
# # Deep Learning Task Notebook: RNN and CNN
#
# **Goal:** Practice two deep-learning topics by importing real datasets and completing the TODO tasks.
#
# - **Part A — RNN:** Movie-review sentiment classification with the IMDB dataset.
# - **Part B — CNN:** Image classification with the CIFAR-10 dataset.
#
# **Requirements:** Python, TensorFlow/Keras, NumPy, Matplotlib. If needed: `pip install tensorflow`.

# %% [markdown]
# ## Part A — RNN for Sentiment Analysis
#
# A movie review is text. Build a model that predicts whether a review is **positive (1)** or **negative (0)** using the IMDB dataset included with Keras (reviews are already integer word IDs).

# %%
# TODO A1: Import the required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
# TODO A2: Load the IMDB dataset (10,000 most frequent words)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# %%
# TODO A3: Inspect the dataset
print("Training reviews:", len(x_train))
print("Test reviews:", len(x_test))
print("Label of first training review:", y_train[0])
print("Length of first training review:", len(x_train[0]))
print("First 20 IDs:", x_train[0][:20])

# %% [markdown]
# ### Why do we pad sequences?
# Reviews have different lengths, but a neural network processes batches of equal-sized sequences. Padding makes every review the same length. With `maxlen=200`, longer reviews are truncated and shorter ones are padded.
#
# ### RNN model plan
# `Integer word IDs → Embedding → SimpleRNN → Dense sigmoid → sentiment probability`

# %%
# TODO A4: Pad the reviews to a length of 200
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# %%
# TODO A5: Build the RNN model
rnn_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=32, mask_zero=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
rnn_model.build(input_shape=(None, 200))
rnn_model.summary()

# %%
# TODO A6: Compile the model
rnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# %%
# TODO A7: Train the model
rnn_history = rnn_model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# %%
# TODO A8: Evaluate on the test data
test_loss, test_acc = rnn_model.evaluate(x_test, y_test)
print("RNN test loss:", test_loss)
print("RNN test accuracy:", test_acc)

# %%
# TODO A9: Plot training vs validation accuracy
plt.plot(rnn_history.history["accuracy"], label="Training accuracy")
plt.plot(rnn_history.history["val_accuracy"], label="Validation accuracy")
plt.title("RNN (IMDB): Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% [markdown]
# ### RNN reflection
#
# 1. **Why an Embedding layer instead of raw word IDs?** Word IDs are just arbitrary labels (ID 500 is not "bigger" or "more positive" than ID 20). Treating them as numeric measurements would imply false order and distance. The Embedding layer maps each ID to a learnable dense vector, so words with similar meaning or sentiment can end up with similar vectors.
# 2. **Why pad reviews?** Batches must be rectangular arrays. Padding (and truncating) makes all reviews the same length (200) so they can be stacked into one tensor.
# 3. **What does `mask_zero=True` do?** It tells downstream layers that ID 0 is padding, so the SimpleRNN skips those time steps instead of treating padding as real words.
# 4. **Why sigmoid?** Sigmoid squashes the output to a value between 0 and 1, which can be read as the probability of a positive review. It pairs naturally with `binary_crossentropy`.
# 5. **Training accuracy much higher than validation accuracy?** This indicates overfitting: the model memorizes training reviews instead of learning patterns that generalize. Remedies include dropout, fewer epochs or early stopping, a smaller model, or more data.

# %% [markdown]
# ---
#
# ## Part B — CNN for Image Classification
#
# Classify a small color image (32 × 32 × 3) into one of 10 categories using **CIFAR-10**.

# %%
# TODO B1: Import the required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# %%
# TODO B2: Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

# %%
# TODO B3: Display one training image
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

plt.imshow(x_train[0])
plt.title(class_names[y_train[0][0]])
plt.axis("off")
plt.show()
print("Label:", y_train[0][0], "->", class_names[y_train[0][0]])

# %%
# TODO B4: Normalize image pixels (0-255 -> 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# %% [markdown]
# ### CNN model plan
# `Image → Conv2D + ReLU → MaxPooling2D → Conv2D + ReLU → MaxPooling2D → Flatten → Dense → 10-class output`

# %%
# TODO B5: Build the CNN
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
cnn_model.summary()

# %%
# TODO B6: Compile the CNN
cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# %%
# TODO B7: Train the CNN
cnn_history = cnn_model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# %%
# TODO B8: Evaluate the CNN
cnn_loss, cnn_acc = cnn_model.evaluate(x_test, y_test)
print("CNN test loss:", cnn_loss)
print("CNN test accuracy:", cnn_acc)

# %%
# TODO B9: Predict one test image
pred = cnn_model.predict(x_test[:1])
pred_class = np.argmax(pred, axis=1)[0]
actual_class = y_test[0][0]
print("Predicted:", class_names[pred_class])
print("Actual:   ", class_names[actual_class])

plt.imshow(x_test[0])
plt.title(f"Predicted: {class_names[pred_class]} | Actual: {class_names[actual_class]}")
plt.axis("off")
plt.show()

# %%
# TODO B10: Plot training vs validation accuracy
plt.plot(cnn_history.history["accuracy"], label="Training accuracy")
plt.plot(cnn_history.history["val_accuracy"], label="Validation accuracy")
plt.title("CNN (CIFAR-10): Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% [markdown]
# ### CNN reflection
#
# 1. **Why divide pixels by 255?** Raw pixels range from 0 to 255. Scaling to 0–1 keeps inputs small and consistent, which makes gradient-based training (Adam) faster and more stable.
# 2. **What does a convolution filter learn?** A small weight matrix (e.g. 3×3) that slides over the image and responds to a local pattern. Early filters tend to detect edges and colors, deeper ones textures and shapes. The same filter is reused across the whole image (weight sharing).
# 3. **Purpose of max pooling?** It downsamples each feature map by keeping the strongest activation in each window. This reduces size and computation, adds some tolerance to small shifts, and helps limit overfitting.
# 4. **Why Flatten before Dense?** Dense layers expect a 1-D vector per sample, while Conv/Pool layers output 3-D feature maps (height × width × channels). Flatten reshapes them.
# 5. **Why 10 units with softmax?** There are 10 classes, so one output per class. Softmax turns the 10 scores into probabilities that sum to 1; the largest is the predicted class.
# 6. **binary vs sparse categorical crossentropy?** `binary_crossentropy` is for two classes (one sigmoid output, labels 0/1). `sparse_categorical_crossentropy` is for multi-class problems with softmax where labels are integer class IDs (not one-hot).

# %% [markdown]
# ---
#
# ## Final comparison task
#
# | Question | RNN — IMDB | CNN — CIFAR-10 |
# |---|---|---|
# | Input data type | Text (sequences of integer word IDs) | Images (RGB pixels) |
# | Input shape after preprocessing | (25000, 200) | (50000, 32, 32, 3) |
# | Main feature-learning layer | Embedding + SimpleRNN | Conv2D |
# | Output layer | Dense(1, sigmoid) | Dense(10, softmax) |
# | Loss function | binary_crossentropy | sparse_categorical_crossentropy |
# | Test accuracy | *(write your result from A8)* | *(write your result from B8)* |

# %% [markdown]
# ---
#
# ## Challenge tasks
#
# Each challenge reloads or rebuilds what it needs, so the cells can be run independently (Part B overwrote `x_train`, so the RNN challenges reload IMDB). Results are collected in a `results` dictionary and compared at the end.

# %%
import time
results = {}

def load_imdb(maxlen):
    (xtr, ytr), (xte, yte) = imdb.load_data(num_words=10000)
    xtr = pad_sequences(xtr, maxlen=maxlen)
    xte = pad_sequences(xte, maxlen=maxlen)
    return xtr, ytr, xte, yte

def run_rnn(name, layer, maxlen, epochs=3):
    xtr, ytr, xte, yte = load_imdb(maxlen)
    m = tf.keras.Sequential([
        tf.keras.layers.Embedding(10000, 32, mask_zero=True),
        layer,
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    m.build(input_shape=(None, maxlen))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    start = time.time()
    h = m.fit(xtr, ytr, epochs=epochs, batch_size=64, validation_split=0.2, verbose=1)
    secs = time.time() - start
    loss, acc = m.evaluate(xte, yte, verbose=0)
    results[name] = {"train_time_s": round(secs, 1), "val_acc": round(h.history["val_accuracy"][-1], 4), "test_acc": round(acc, 4)}
    print(name, results[name])
    return h

# %%
# Challenge 1 (RNN): maxlen = 100 vs baseline maxlen = 200
h_base = run_rnn("RNN baseline (maxlen=200)", tf.keras.layers.SimpleRNN(32), maxlen=200)
h_100 = run_rnn("RNN maxlen=100", tf.keras.layers.SimpleRNN(32), maxlen=100)

# %%
# Challenge 2 (RNN): SimpleRNN(32) -> LSTM(32), maxlen = 200
h_lstm = run_rnn("LSTM(32) maxlen=200", tf.keras.layers.LSTM(32), maxlen=200)

# %%
# Challenge 3 & 4 (CNN): helper using the normalized CIFAR-10 data from Part B
def run_cnn(name, filters2=64, dropout=None, epochs=5):
    layers = [
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(filters2, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
    ]
    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(10, activation="softmax"))
    m = tf.keras.Sequential(layers)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start = time.time()
    h = m.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_split=0.1, verbose=1)
    secs = time.time() - start
    loss, acc = m.evaluate(x_test, y_test, verbose=0)
    results[name] = {"train_time_s": round(secs, 1),
                     "train_acc": round(h.history["accuracy"][-1], 4),
                     "val_acc": round(h.history["val_accuracy"][-1], 4),
                     "test_acc": round(acc, 4)}
    print(name, results[name])
    return h

h_cnn_base = run_cnn("CNN baseline")

# %%
# Challenge 3 (CNN): add Dropout(0.3) before the final output layer
h_drop = run_cnn("CNN + Dropout(0.3)", dropout=0.3)

for h, label in [(h_cnn_base, "Baseline"), (h_drop, "Dropout 0.3")]:
    plt.plot(h.history["accuracy"], label=label + " train")
    plt.plot(h.history["val_accuracy"], "--", label=label + " val")
plt.title("CNN: effect of Dropout")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.show()

# %%
# Challenge 4 (CNN): second Conv2D filters 64 -> 128
h_128 = run_cnn("CNN 128 filters (conv2)", filters2=128)

# %%
# Summary of all experiments
for name, r in results.items():
    print(f"{name:32s}", r)

# %% [markdown]
# ### Challenge observations — fill in from your own `results` output
#
# | Experiment | What to compare | What usually happens |
# |---|---|---|
# | RNN `maxlen` 200 → 100 | Training time, test accuracy | Faster per epoch; accuracy often changes little or drops slightly because less of each review is seen |
# | `SimpleRNN` → `LSTM` | Test accuracy, stability | LSTM usually learns long-range sentiment better and trains more stably (it handles vanishing gradients), but is slower |
# | CNN + `Dropout(0.3)` | Gap between train and val accuracy | The gap usually shrinks (less overfitting); training accuracy may be a bit lower |
# | CNN 64 → 128 filters | Test accuracy, time, parameters | Slightly higher capacity and often a small accuracy gain, but slower and more parameters (and more overfitting risk) |
#
# Write your actual observations here after running: *(your results)*

# %% [markdown]
# ### Submission checklist
# - [x] All TODO cells completed.
# - [ ] Both models trained successfully (run the notebook).
# - [ ] Test loss and accuracy recorded.
# - [ ] Accuracy plots included.
# - [x] Reflection questions answered.
# - [ ] Results compared in the final table.
# - [ ] Challenge tasks run and observations written.
