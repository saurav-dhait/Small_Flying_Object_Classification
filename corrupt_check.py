import os
from PIL import Image


def check_corrupted_images(images_path):
    """Check for corrupted images in a directory."""
    corrupted_images = []
    for image_file in os.listdir(images_path):
        image_path = os.path.join(images_path, image_file)
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            corrupted_images.append(image_path)
    return corrupted_images


def check_empty_labels(labels_path):
    """Check for truly empty label files."""
    empty_labels = []
    for label_file in os.listdir(labels_path):
        label_path = os.path.join(labels_path, label_file)
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                if not any(line.strip() for line in f.readlines()):  # Check for non-empty content
                    empty_labels.append(label_path)
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
            empty_labels.append(label_path)  # Add in case file is inaccessible
    return empty_labels


def delete_empty_labels_and_images(empty_labels, images_path):
    """Delete empty label files and their corresponding image files."""
    deleted_files = {"labels": [], "images": []}
    for label_path in empty_labels:
        # Delete label file
        os.remove(label_path)
        deleted_files["labels"].append(label_path)

        # Find corresponding image file
        label_name = os.path.splitext(os.path.basename(label_path))[0]
        image_extensions = ['.jpg', '.png', '.jpeg']
        for ext in image_extensions:
            image_path = os.path.join(images_path, label_name + ext)
            if os.path.exists(image_path):
                os.remove(image_path)
                deleted_files["images"].append(image_path)
                break  # Stop checking other extensions
    return deleted_files


def check_unmatched_files(images_path, labels_path):
    """Check for unmatched image and label files."""
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_path) if f.endswith('.txt')}

    unmatched_images = [os.path.join(images_path, img + ".jpg") for img in
                        image_files - label_files]  # Assuming images are JPG
    unmatched_labels = [os.path.join(labels_path, lbl + ".txt") for lbl in label_files - image_files]

    return unmatched_images, unmatched_labels


def check_yolo_dataset(dataset_path):
    """Check the YOLO dataset for issues."""
    issues = {"corrupted_images": [], "empty_labels": [], "unmatched_files": []}

    for split in ["train", "test", "val"]:
        split_path = os.path.join(dataset_path, split)
        images_path = os.path.join(split_path, "images")
        labels_path = os.path.join(split_path, "labels")

        # Perform checks
        issues["corrupted_images"].extend(check_corrupted_images(images_path))
        issues["empty_labels"].extend(check_empty_labels(labels_path))
        unmatched_images, unmatched_labels = check_unmatched_files(images_path, labels_path)
        issues["unmatched_files"].extend(unmatched_images + unmatched_labels)

    return issues


def print_issues(issues):
    """Print the issues found in the dataset."""
    if issues["corrupted_images"]:
        print("Corrupted Images:")
        print("\n".join(issues["corrupted_images"]))

    if issues["empty_labels"]:
        print("\nEmpty Label Files:")
        print("\n".join(issues["empty_labels"]))

    if issues["unmatched_files"]:
        print("\nUnmatched Files:")
        print("\n".join(issues["unmatched_files"]))

    if not any(issues.values()):
        print("Dataset is clean and well-formatted!")
    else:
        print(f"Empty label files : {len(issues['empty_labels'])}")
        print(f"Corrupted images : {len(issues['corrupted_images'])}")
        print(f"Unmatched files : {len(issues['unmatched_files'])}")


# Usage
if __name__ == "__main__":
    dataset_path = r"dataset/FLYING_OBJECT_DATASET_SMALL"  # Replace with the path to your YOLO dataset

    # Step 1: Check for issues
    issues = check_yolo_dataset(dataset_path)
    print_issues(issues)

    # Step 2: Delete empty labels and their corresponding images
    for split in ["train", "test", "val"]:
        split_path = os.path.join(dataset_path, split)
        images_path = os.path.join(split_path, "images")
        labels_path = os.path.join(split_path, "labels")

        empty_labels = check_empty_labels(labels_path)
        deleted_files = delete_empty_labels_and_images(empty_labels, images_path)

        if deleted_files["labels"] or deleted_files["images"]:
            print(
                f"\nDeleted {len(deleted_files['labels'])} label files and {len(deleted_files['images'])} image files in '{split}' split.")
