from data_load import X, y
from sklearn.model_selection import train_test_split

# =========================
# TRAIN / VAL / TEST SPLIT
# =========================

# First split: Train + Temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,        # 80% train, 20% temp
    random_state=42,
    stratify=y            # keeps class distribution balanced
)

# Second split: Validation + Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,        # 10% val, 10% test
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)
