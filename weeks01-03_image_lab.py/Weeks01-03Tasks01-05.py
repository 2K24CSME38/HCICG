import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def inspect_image(image_path: str)-> dict:
    """
    Task 1: Load image and measure properties like dimensions,channels,and size.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image Not Found: {image_path}")

    height, width, channels = img.shape
    pixel_count = height * width
    shape = [height, width, channels] 

    estimated_bytes = pixel_count * channels
    color_order = "BGR"

    print("---Task 1: Image Properties---")
    print(f"Dimensions: {width}x{height}")
    print(f"Channels: {channels}")
    print(f"Total Pixels: {pixel_count}")
    print(f"Estimated Size (Bytes): {estimated_bytes}")
    print(f"Color Format: {color_order}")

    return {
        "width":width,
        "height":height,
        "channels":channels,
        "shape":shape,
        "pixel_count":pixel_count,
        "estimated_bytes":estimated_bytes,
        "color_order":color_order
    }

def create_pixel_views(image_path: str, outputs_dir: str)-> dict:
    """
    Task 2: Color channels separation,grayscale,conversion,and downsampling.
    """
    os.makedirs(outputs_dir,exist_ok=True)
    img_bgr = cv2.imread(image_path)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    red_ch = img_rgb[:, :, 0]
    green_ch = img_rgb[:, :, 1]
    blue_ch = img_rgb[:, :, 2]

    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray_img.shape
    new_w, new_h = w // 2, h // 2
    downsampled = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)


    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(red_ch, cmap='Reds')
    plt.title("Red Channels")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(green_ch, cmap='Greens')
    plt.title("Green Channels")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(blue_ch, cmap='Blues')
    plt.title("Blue Channels")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(gray_img, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(downsampled, cmap='gray')
    plt.title(f"Half-Size ({new_w}x{new_h})")
    plt.axis('off')

    plt.tight_layout()
    output_path = os.path.join(outputs_dir, "pixel_views.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()

    return {
        "original_size" :[w, h],
        "downsampled_size": [new_w, new_h,],
        "output_path": output_path
    }

def create_adjustments(
        image_path: str,
        output_dir: str,
        brightness_delta: int = 40,
        contrast_factor : float = 1.5,
        threshold: int = 127
)-> dict:
    """
    Task 3: Adjust brightness, contrast, and apply binary thresholding.
    """
    os.makedirs(output_dir,exist_ok=True)
    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bright_img = np.clip(gray.astype(np.int16) + brightness_delta, 0,255).astype(np.uint8)
    contrast_img = np.clip(gray.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)
    
        
    _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')
    
        
    plt.subplot(2, 2, 2)
    plt.imshow(bright_img, cmap='gray')
    plt.title(f"Brighter (+{brightness_delta})")
    plt.axis('off')
    
        
    plt.subplot(2, 2, 3)
    plt.imshow(contrast_img, cmap='gray')
    plt.title(f"High Contrast (x{contrast_factor})")
    plt.axis('off')
    
        
    plt.subplot(2, 2, 4)
    plt.imshow(thresh_img, cmap='gray')
    plt.title(f"Binary Threshold (T={threshold})")
    plt.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "adjustments.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()
    
    return {
            "brightness_delta": brightness_delta,
            "contrast_factor": contrast_factor,
            "threshold": threshold,
            "output_path": output_path
        }
    
def create_blur_and_edges(
            image_path: str,
            output_dir: str,
            kernel_size: int = 5
    ) -> dict:
        """
        Task 4: Apply Mean Blur and perform Sobel Edge Detection.
        """
        os.makedirs(output_dir, exist_ok=True)
        img_bgr = cv2.imread(image_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
        blurred = cv2.blur(gray, (kernel_size, kernel_size))
    
        sobelx_orig = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely_orig = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_orig = cv2.magnitude(sobelx_orig, sobely_orig)
        edges_orig = np.clip(edges_orig, 0, 255).astype(np.uint8)
    
        sobelx_blur = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely_blur = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges_blur = cv2.magnitude(sobelx_blur, sobely_blur)
        edges_blur = np.clip(edges_blur, 0, 255).astype(np.uint8)
    
        plt.figure(figsize=(10, 8))
    
        plt.subplot(2, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Original Grayscale")
        plt.axis('off')
    
        plt.subplot(2, 2, 2)
        plt.imshow(blurred, cmap='gray')
        plt.title(f"Mean Blur (Kernel {kernel_size}x{kernel_size})")
        plt.axis('off')
    
        plt.subplot(2, 2, 3)
        plt.imshow(edges_orig, cmap='gray')
        plt.title("Sobel Edges (Original)")
        plt.axis('off')
    
        plt.subplot(2, 2, 4)
        plt.imshow(edges_blur, cmap='gray')
        plt.title("Sobel Edges (Blurred)")
        plt.axis('off')
    
        plt.tight_layout()
        output_path = os.path.join(output_dir, "blur_and_edges.png")
        plt.savefig(output_path)
        plt.show()
        plt.close()
    
        return {
            "kernel_size": kernel_size,
            "output_path": output_path
        }
    
def run_lab(image_path: str, output_dir: str) -> dict:
        """
        Task 5: Execute all lab tasks sequentially.
        """
        t1 = inspect_image(image_path)
        t2 = create_pixel_views(image_path, output_dir)
        t3 = create_adjustments(image_path, output_dir)
        t4 = create_blur_and_edges(image_path, output_dir)
    
        return {
            "task1": t1,
            "task2": t2,
            "task3": t3,
            "task4": t4
        }
    
import tkinter as tk
from tkinter import filedialog
    
if __name__ == "__main__":
        
        root = tk.Tk()
        root.withdraw()
    
        
        print("Please select your image file...")
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
    
        if not image_path:
            print("No image selected. Exiting...")
        else:
            output_dir = "outputs"
            run_lab(image_path, output_dir)