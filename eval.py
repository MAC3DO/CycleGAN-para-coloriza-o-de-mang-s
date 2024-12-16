import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import config

def calculate_psnr(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    mse = np.sum((image1 - image2) ** 2) / float(image1.shape[0] * image1.shape[1])
   
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0  
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def evaluate_psnr(directory1, directory2):
    psnr_list = []
    
    images = sorted(os.listdir(directory1))
    
    for img_name in images:
        if img_name.startswith("color_out") and img_name.endswith(".png"):
            reference_name = img_name.replace("color_out", "color_ref")
            reference_path = os.path.join(directory2, reference_name)
            if os.path.exists(reference_path):
                generated_image = cv2.imread(os.path.join(directory1, img_name))
                reference_image = cv2.imread(reference_path)
                
                if generated_image is None or reference_image is None:
                    print(f"Error loading image pair: {img_name} and {reference_name}")
                    continue
                
                psnr_value = calculate_psnr(generated_image, reference_image)
                psnr_list.append(psnr_value)
                print(f"PSNR for {img_name} and {reference_name}: {psnr_value:.2f} dB")
    
    if psnr_list:
        average_psnr = np.mean(psnr_list)
        return average_psnr
    else:
        print("No valid image pairs found.")
        return 0


def calculate_ssim(image1, image2):
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    ssim_value, _ = ssim(image1, image2, full=True)
    return ssim_value

def evaluate_ssim(directory1, directory2):
    ssim_list = []
    
    images = sorted(os.listdir(directory1))
    
    for img_name in images:
        if img_name.startswith("color_out") and img_name.endswith(".png"):
            reference_name = img_name.replace("color_out", "color_ref")
            reference_path = os.path.join(directory2, reference_name)
            if os.path.exists(reference_path):
                generated_image = cv2.imread(os.path.join(directory1, img_name))
                reference_image = cv2.imread(reference_path)
                
                if generated_image is None or reference_image is None:
                    print(f"Error loading image pair: {img_name} and {reference_name}")
                    continue
                
                ssim_value = calculate_ssim(generated_image, reference_image)
                ssim_list.append(ssim_value)
                print(f"SSIM for {img_name} and {reference_name}: {ssim_value}")
    
    if ssim_list:
        average_ssim = np.mean(ssim_list)
        return average_ssim
    else:
        print("No valid image pairs found.")
        return 0
    
if __name__ == "__main__":
    directory = config.OUTPUT_DIR
    
    eval_psnr = evaluate_psnr(directory, directory)
    eval_ssim = evaluate_ssim(directory, directory)
    
    print(f"PSNR = {eval_psnr}\nSSIM = {eval_ssim}")
