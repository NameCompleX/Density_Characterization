import cv2

# Read the grayscale image
image_path = "Your/Image/Path/Here.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the slice dimensions
WIDTH = 4
LENGTH = 4

# Calculate the average density for each slice
with open("Your save path.txt", "w") as file:
    for i in range(0, image.shape[0], LENGTH):
        for j in range(0, image.shape[1], WIDTH):
            x_start = j
            x_end = j + WIDTH
            y_start = i
            y_end = i + LENGTH
            slice = image[y_start:y_end, x_start:x_end]
            average_density = slice.mean()
            file.write(f"{x_start}  {x_end} {y_start} {y_end} {average_density}\n")



