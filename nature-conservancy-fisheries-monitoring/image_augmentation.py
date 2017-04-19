from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
import os

train_folder = 'train'
fish_folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')



for folder in fish_folders:
    # get all image names in that folder
    folder_path = os.path.join(train_folder, folder) + os.sep
    print("Generating images for: {}".format(folder_path))
    images = glob.glob(folder_path + "*.jpg")

    image_count = len(images)
    current_image = 1


    for image in images:
        current_image += 1
        print("Processing image {} from {}".format(current_image, image_count))
        image_name = os.path.basename(image)
        image_name_without_extension = os.path.splitext(image_name)[0]
        img = load_img(image)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=folder_path, save_prefix=image_name_without_extension + '_augmented_' + str(i), save_format='jpg'):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely
