import sys
import os
import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageOps

def centre_of_mass(image):
    if image.mode != 'L':
        image = image.convert('L')
    
    width, height = image.size

    sum_weighted_x = 0.0
    sum_weighted_y = 0.0
    total_weight = 0.0

    for y in range(height):
        for x in range(width):
            color = image.getpixel((x, y))

            weighted_x = x * color
            weighted_y = y * color

            sum_weighted_x += weighted_x
            sum_weighted_y += weighted_y
            total_weight += color

    if total_weight != 0:
        centre_of_mass_x = round(sum_weighted_x / total_weight)
        centre_of_mass_y = round(sum_weighted_y / total_weight)
    else:
        centre_of_mass_x = 0
        centre_of_mass_y = 0

    return (centre_of_mass_x, centre_of_mass_y)

def reshape_image(image, size):
    centre = centre_of_mass(image)

    new_centre_x = round(size[0] / 2)
    new_centre_y = round(size[1] / 2)

    offset_x = new_centre_x - centre[0]
    offset_y = new_centre_y - centre[1]

    offset = (offset_x, offset_y)

    new_image = Image.new('L', size, color=255)

    new_image.paste(image, offset)

    new_image.thumbnail(size)

    return new_image

def reshape_images(images):
    max_x = 0
    max_y = 0

    for name, image in images.items():
        max_x = max(max_x, image.size[0])
        max_y = max(max_y, image.size[1])

    new_size = (max_x, max_y)
    
    for name, image in images.items():
        images[name] = reshape_image(image, new_size)
        
    return images

def metric(image1, image2):
    array1 = np.array(ImageOps.invert(image1))
    array2 = np.array(ImageOps.invert(image2))

    if array1.shape != array2.shape:
        raise ValueError("Images must have the same size")
    
    #calculating how similar the images are using something close to intersection over union, if they are the same the function returns ~1.0
    return np.sum(np.minimum(array1, array2)) / np.sum(np.maximum(array1, array2))

def group_images(images, sample_images):
    grouping = {}
    min_similarity = 1.0
    for i, image in enumerate(sample_images):
        grouping[i] = list()

    for name, image in images.items():
        max_similarity = 0.0
        best_group = 0
        for i, s_image in enumerate(sample_images):
            similarity = metric(image, s_image)
            if similarity >= max_similarity:
                best_group = i
                max_similarity = similarity

        if max_similarity < min_similarity:
            min_similarity = min(min_similarity, max_similarity)
            worst_name = name

        grouping[best_group].append(name)
    return grouping, min_similarity, worst_name

    
def resample(images, grouping):
    sample_images = []
    for key in grouping.keys():
        groups_images = [np.array(images[name]) for name in grouping[key]]
        if len(groups_images) > 0:
            average_array = np.mean(groups_images, axis=0).astype(np.uint8)
            sample_images.append(Image.fromarray(average_array))
    return sample_images

def save_images(imgs):
    for key, val in enumerate(imgs):
        val.save("test_out/" + str(key) + ".png")
    
def read_input(file_path):

    images = {}

    with open(file_path, 'r') as file:
        file_paths = file.readlines()

    for path in file_paths:
        path = path.strip()
        filename = os.path.basename(path)
        try:
            image = Image.open(path).convert('L')
            images[filename] = image
        except IOError:
            print(f"Error: Unable to open image file '{path}'")

    return images

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        sys.exit(1)

    file_path = sys.argv[1]

    images = read_input(file_path)
    
    images = reshape_images(images)

    best_grouping = {}
    best_sample = []
    k = 1
    best_eval = 0
    new_worst = ''
    while k <= len(images) and best_eval < 0.45:
        i_max = 3
        previ_eval = best_eval

        to_be_removed = []
        for j in range(len(best_sample)):
            for l in range(j+1, len(best_sample)):
                if metric(best_sample[j], best_sample[l]) > 0.47:
                    to_be_removed.append(l)

        best_sample = [best_sample[j] for j in range(len(best_sample)) if j not in to_be_removed]
        prev_sample = best_sample

        
        print()
        print("k: ", k)

        for i in range(i_max):
           

            sample_names = np.random.choice(np.array(list(images.keys())), size=k-len(prev_sample), replace=False)
            sample_images = [images[name] for name in sample_names]

            if k > 1 and len(sample_images) > 0 and i == i_max - 1:
                sample_images.pop()
                sample_images.append(images[new_worst])
            
            sample_images = sample_images + prev_sample

            if i < 2:
                sample_names = np.random.choice(np.array(list(images.keys())), size=k, replace=False)
                sample_images = [images[name] for name in sample_names]

            grouping, prev_eval, _ = group_images(images, sample_images)

            if k == len(images):
                break

            sample_images = resample(images, grouping)

            grouping, new_eval, new_worst = group_images(images, sample_images)

            optimizations = 1

            while new_eval - prev_eval > 0.01:
                optimizations = optimizations + 1

                prev_eval = new_eval

                sample_images = resample(images, grouping)

                grouping, new_eval, new_worst = group_images(images, sample_images)

            if new_eval > best_eval:
                best_eval = new_eval
                best_sample = sample_images
                best_grouping = grouping
            print("optimizations: ", optimizations)

        print("best: ", best_eval, "prev: ", previ_eval)

        save_images(best_sample)

        if best_eval - previ_eval < 0.01 and best_eval < 0.5:
            k = k + 2
        k = k + 1
    print(grouping)

