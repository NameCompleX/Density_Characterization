import numpy as np
import cv2
import os

# File dir
IMAGE_DIR = "Your/Path/To/Save/Images"
TXT_DIR = "Your/Path/To/Save/Txts"
WIDTH = 4
LENGTH = 4
IMAGE_SIZE = 512

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)


import numpy as np
import cv2

def generate_image(img_size, WIDTH):
    image = np.zeros(img_size, dtype=np.float32)  
    pixel_info = []  

    grayscale_range = (120, 200)

    num_blocks = img_size[1] // WIDTH  

    line_count = 0  

    for block in range(num_blocks):
        x_start_block = block * WIDTH
        x_end_block = (block + 1) * WIDTH

        # randomly determine the number of lines in this block
        num_lines_in_block = np.random.randint(5 * WIDTH, 100 * WIDTH)
        for _ in range(num_lines_in_block):
            # randomly determine line properties
            intensity_value = np.random.randint(grayscale_range[0], grayscale_range[1])
            x_start = np.random.randint(x_start_block, x_end_block)  # begin x coordinate of the line
            y_start = np.random.randint(-300, 600)  # begin y coordinate of the line
            line_length = np.random.randint(50, 201)  # length of the line
            y_end = y_start + line_length  

            for y in range(y_start, y_end + 1):
                if 0 <= y < img_size[0]:  
                    image[y, x_start] += intensity_value  

            for y in range(y_start, y_end + 1):
                if 0 <= y < img_size[0]:  
                    pixel_info.append([x_start, y, int(intensity_value)])

    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

    max = 200
    min = 50

    image_normalized = image_normalized * (max - min) + min


    image_normalized = np.clip(image_normalized, min, max)

    return image_normalized.astype(np.uint8), pixel_info



def save_image_and_data(img, pixel_data, image_path, txt_path, block_size=(WIDTH, LENGTH)):

    cv2.imwrite(image_path, img)


    height, width = img.shape
    block_data = {}  


    for (x_start, y_start, intensity) in pixel_data:
        if intensity > 0:  
            
            x_block_start = (x_start // block_size[0]) * block_size[0]
            y_block_start = (y_start // block_size[1]) * block_size[1]
            x_block_end = x_block_start + block_size[0] - 1
            y_block_end = y_block_start + block_size[1] - 1

            block_key = (x_block_start, y_block_start)

            if block_key not in block_data:
                block_data[block_key] = {"count": 0, "line_count": 0}

            block_data[block_key]["count"] += 1
            block_data[block_key]["line_count"] += 1

    with open(txt_path, 'w') as f:
        for (x_start, y_start), data in block_data.items():
            x_end = x_start + block_size[0] 
            y_end = y_start + block_size[1] 

            line_count = data["count"]
            pixel_count = block_size[0] 
            density = line_count / pixel_count  

            f.write(f"{x_start} {x_end} {y_start} {y_end} {density}\n")  

    print(f"Data saved to {txt_path}")

def generate_images_in_batches(total_images, img_size=(IMAGE_SIZE, IMAGE_SIZE)):
    for idx in range(total_images):
        image, pixel_data = generate_image(img_size, WIDTH)

        image_path = os.path.join(IMAGE_DIR, f"img{idx+1}.png")
        txt_path = os.path.join(TXT_DIR, f"img{idx+1}.txt") 
        save_image_and_data(image, pixel_data, image_path, txt_path)

        print(f"Image {idx + 1}/{total_images} saved.")

def main():
    total_images = 1   #generate number of images

    print(f"Generating {total_images} images...")
    generate_images_in_batches(total_images)
    print("Image generation completed.")

if __name__ == "__main__":
    main()
