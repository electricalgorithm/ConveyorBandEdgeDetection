"""
@file: detect_edges.py
@brief: This script finds you the edges of the conveyor band in an image.
@author: Gökhan Koçmarlı (github.com/electricalgorithm)
@company: YongaTek (yongatek.com)
"""
import cv2
import numpy
import math
import argparse


def apply_gaussian_filter(image: numpy.ndarray, window_size=51, gaussian_sigma=5) -> numpy.ndarray:
    """
    Function returns the cv2 Image instance that
    has Gaussian blur with predefined window size
    and gaussian sigma ratio.
    :param image: Original image to perform blur onto.
    :param window_size: Default is 5.
    :param gaussian_sigma: Default is 5.
    :return: The blurred image as a numpy.ndarray.
    """
    return cv2.GaussianBlur(image, (window_size, window_size), gaussian_sigma)


def apply_binarization(image: numpy.ndarray) -> numpy.ndarray:
    """
    The function gets an image input, and apply Otsu's Method
    to find its binary representation.
    :param image: Original 3-channel image to perform Otsu's method.
    :return: The binary image as a numpy.ndarray.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def apply_dilation_erosion(image: numpy.ndarray, kernel_size=35) -> numpy.ndarray:
    """
    The function removes unnecassary objects in the picture with using
    dilation firstly with very-high kernel size, and erosion to save
    real sizes of the object considered.
    :param image: A binary image that has more than one object.
    :param kernel_size: Default is 35 (experimentally set).
    :return: The image's numpy.ndarray that has less objects.
    """
    element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (2 * kernel_size + 1, 2 * kernel_size + 1),
                                        (kernel_size, kernel_size))
    dilated_image = cv2.dilate(image, element)
    return cv2.erode(dilated_image, element)


def apply_canny_detection(image: numpy.ndarray, min_threshold=100, max_threshold=250) -> numpy.ndarray:
    """
    This function applies the Canny algorithm to distinguish a object border.
    :param image: A binary image.
    :param min_threshold: Default is 100.
    :param max_threshold: Default is 250.
    :return: A numpy.ndarray that has only borders of the image.
    """
    return cv2.Canny(image, min_threshold, max_threshold)


def apply_hough_lines(original_image: numpy.ndarray, dilated_image: numpy.ndarray, threshold=175) -> None:
    """
    This method firstly finds the border lines using Hough's transformation method with experimentally
    predetermined threshold. Afterwards, it marks the original image with that border line.
    :param original_image: A image to put found lines into.
    :param dilated_image: The image that will be looked for lines.
    :param threshold: Default is 175.
    :return: None.
    """
    # Find the line.
    lines = cv2.HoughLines(dilated_image, 1, numpy.pi / 180, threshold, None, 0, 0)

    # Mark the line into the original image.
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(original_image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)


def get_image_location_from_argparser() -> str:
    """
    This function creates the argument parser and returns the image file location.
    :return: File location as string
    """
    parser = argparse.ArgumentParser(description="This script finds you the edges of the conveyor band in an image.")
    parser.add_argument("-i", "--inputImage", nargs="+", required=True, help="Provide the image file location.")
    args = parser.parse_args()
    return args.inputImage[0]


if __name__ == "__main__":
    # Get the file location, and check it.
    file_location = get_image_location_from_argparser()
    if not file_location:
        raise Exception("You have to give a file location using --inputImage flag.")

    # Read the image as ndarray.
    input_img = cv2.imread(file_location)
    if input_img.size == 0:
        raise Exception("The file you provided couldn't find.")

    # Make the operations.
    blurred_img = apply_gaussian_filter(input_img)
    binary_img = apply_binarization(blurred_img)
    dilated_img = apply_dilation_erosion(binary_img)
    cannyed_img = apply_canny_detection(dilated_img)

    # Create a copy of original to show line in that photo.
    output_img = numpy.copy(input_img)
    apply_hough_lines(output_img, cannyed_img)

    # Show the results.
    cv2.imshow("Conveyor Edge Detection Results | YongaTek", output_img)
    cv2.waitKey(0)
