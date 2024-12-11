import os


def generate_csv(yolo_dir, output_csv_path):
    splits = ['train', 'test', 'val']
    data = []

    for split in splits:
        images_dir = os.path.join(yolo_dir, split, 'images')
        labels_dir = os.path.join(yolo_dir, split, 'labels')

        for img_file in os.listdir(images_dir):
            if img_file.endswith('.jpg'):  # Adjust for your image extension
                img_path = os.path.join(images_dir, img_file)
                label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))  # Match YOLO label file

                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    if lines:  # Check if label file is not empty
                        line = lines[0].strip()  # Read the first line
                        parts = line.split()

                        if len(parts) >= 1:  # Ensure there's at least a class ID
                            class_label = int(parts[0])  # Extract class ID
                            data.append([split, img_path, class_label])
                        else:
                            print(f"Skipping {label_path}: Label format invalid. Deleting files.")
                            os.remove(img_path)  # Delete the image file
                            os.remove(label_path)  # Delete the label file
                    else:
                        print(f"Skipping {label_path}: Label file is empty. Deleting files.")
                        os.remove(img_path)  # Delete the image file
                        os.remove(label_path)  # Delete the label file
                else:
                    print(f"Label file not found for image {img_file}. Skipping.")

    # Write to CSV
    with open(output_csv_path, 'w') as f:
        f.write('split,image_path,class_label\n')
        for row in data:
            f.write(','.join(map(str, row)) + '\n')


# Paths
yolo_dataset_path = 'dataset/FLYING_OBJECT_DATASET_SMALL'
output_csv_path = 'dataset/dataset_labels.csv'

# Generate CSV
generate_csv(yolo_dataset_path, output_csv_path)
