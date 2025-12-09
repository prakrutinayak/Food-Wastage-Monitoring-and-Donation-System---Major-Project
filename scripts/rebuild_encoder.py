import os
import pickle
from sklearn.preprocessing import LabelEncoder

train_dir = "dataset/FruitSpoilage/Train"

# Only include directories (actual class folders)
class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

# Sort for consistency
class_names.sort()

# Encode
encoder = LabelEncoder()
encoder.fit(class_names)

# Save
with open("models/label_encoder_v2.pkl", "wb") as f:
    pickle.dump(encoder, f)

print(" New LabelEncoder saved as label_encoder_v2.pkl")
print(" Classes:", encoder.classes_)