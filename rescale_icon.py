import cv2
import numpy as np

def rescale_image(image_path, output_prefix, num_scales):
    # Load the image from the specified path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at path: {image_path}")
        return

    # Determine the scale factors from 0.2 to 2.0
    scale_factors = np.linspace(0.2, 2.0, num_scales)

    # Process each scale factor
    for i, scale in enumerate(scale_factors, start=1):
        # Calculate the new dimensions
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Save the resized image
        output_filename = f'{output_prefix}{i}.png'
        cv2.imwrite(output_filename, resized_image)
        print(f"Image has been resized to scale {scale:.2f} and saved as '{output_filename}'")

# Specify the path to your image and the desired output prefix
image_path = 'icon.png'
output_prefix = 'icon'  # Prefix for the output files

# Number of scales and output files to generate
num_scales = 10

# Call the function
rescale_image(image_path, output_prefix, num_scales)
