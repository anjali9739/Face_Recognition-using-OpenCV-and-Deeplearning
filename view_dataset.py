import os
import cv2

# Path to the dataset folder
DATASET_DIR = "dataset"

# Loop through each folder in the dataset directory
for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    
    # Continue only if it's a directory (each directory corresponds to a person)
    if not os.path.isdir(person_dir):
        continue

    # Loop through each image in the person's folder
    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        
        # Read the image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            print(f"Unable to load image: {img_path}")
            continue
        
        # Display the image in a window
        window_title = f"{person} - {filename}"
        cv2.imshow(window_title, image)
        
        # Wait until any key is pressed
        cv2.waitKey(0)
        
        # Close the current image window
        cv2.destroyWindow(window_title)

# Optional: Close all windows if any are still open
cv2.destroyAllWindows()
