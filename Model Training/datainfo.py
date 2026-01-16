import os

DATASET_DIR = "C:\Rishika\MajorProject_1\Indian"

number_classes = {}
alphabet_classes = {}

total_images = 0

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    # Count only JPG files
    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(".jpg")
    ]

    count = len(images)
    total_images += count

    # Separate numbers and alphabets
    if class_name.isdigit():
        number_classes[class_name] = count
    elif class_name.isalpha() and len(class_name) == 1:
        alphabet_classes[class_name] = count

print("\n📊 ISL DATASET SUMMARY")
print("=" * 55)

print("\n🔢 NUMBERS (0–9)")
for cls, cnt in number_classes.items():
    print(f"Class {cls} : {cnt} images")

print(f"➡ Total Number Images: {sum(number_classes.values())}")

print("\n🔤 ALPHABETS (A–Z)")
for cls, cnt in alphabet_classes.items():
    print(f"Class {cls} : {cnt} images")

print(f"➡ Total Alphabet Images: {sum(alphabet_classes.values())}")

print("\n📦 GRAND TOTAL")
print(f"Total Images in Dataset: {total_images}")
