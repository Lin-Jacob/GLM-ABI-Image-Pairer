from PIL import Image
import os
import cv2

def split_image(image_path, sub_image_size, output_dir = None):
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = Image.open(image_path).convert("L")
    image_width, image_height = image.size
    sub_width, sub_height = sub_image_size

    sub_images = []
    for top in range(0, image_height, sub_height):
        for left in range(0, image_width, sub_width):
            right = min(left + sub_width, image_width)
            bottom = min(top + sub_height, image_height)
            sub_image = image.crop((left, top, right, bottom))
            #sub_image_path = os.path.join(output_dir, f'sub_image_{count}.png')
            #sub_image.save(sub_image_path)
            sub_images.append(sub_image)
    return sub_images

  
def reassemble_image(sub_image_paths, original_size, sub_image_size, output_image_path = None):
    original_width, original_height = original_size
    sub_width, sub_height = sub_image_size

    reassembled_image = Image.new('L', (original_width, original_height))
    index = 0
    for top in range(0, original_height, sub_height):
        for left in range(0, original_width, sub_width):
            sub_image = Image.open(sub_image_paths[index])
            reassembled_image.paste(sub_image, (left, top))
            index += 1
            
    reassembled_image.save(output_image_path) if output_image_path is not None else None
    return reassembled_image

def extract_center_image(image_path: str):
    img = cv2.imread(image_path)
    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
    subset_y, subset_x = img.shape[0] // 4, img.shape[1] // 4
    subset_img = img[center_y - subset_y: center_y +
                         subset_y, center_x - subset_x: center_x + subset_x]
    output = f'middle_subset_images/subset_{image_path}'
    cv2.imwrite(output,subset_img)
    return output