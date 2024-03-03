# import cv2
# import numpy as np
#
# # Load the shredded image
# image_path = 'Riddles/cv_easy_example/actual.jpg'
# shredded_image = cv2.imread(image_path)
#
# # Convert the image to grayscale
# gray_image = cv2.cvtColor(shredded_image, cv2.COLOR_BGR2GRAY)
#
# # Apply edge detection
# edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
#
# # Find contours in the edge-detected image
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# print(contours)
# # Sort contours by their x-coordinates
# contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
#
# # Reassemble the shredded pieces
# shred_indices = []
# for contour in contours:
#     # Get the bounding box of the contour
#     x, y, w, h = cv2.boundingRect(contour)
#     # Calculate the index of the shred based on its leftmost position
#     shred_index = x // 64
#     shred_indices.append(shred_index)
#
#
# print(shred_indices)
import math

import cv2
import numpy as np

from LSBSteg import encode
from Solvers.riddle_solvers import riddle_solvers

# Read the concatenated image parts
image_path = 'Riddles/cv_easy_example/shredded.jpg'
concatenated_image = cv2.imread(image_path)

# Assuming you know the dimensions of each shredded part
part_width = 100
part_height = 100

# Assuming each strip has a width of 64 pixels
strip_width = 64

# Determine the number of strips
num_strips = concatenated_image.shape[1] // strip_width

# # Create an empty image to store the reconstructed image
# reconstructed_image = np.zeros_like(concatenated_image)
#
# # Reassemble the image
# for i in range(num_strips):
#     strip = concatenated_image[:, i * strip_width: (i + 1) * strip_width]
#     reconstructed_image[:, i * strip_width: (i + 1) * strip_width] = strip

# cv2.imshow('Reconstructed Image', reconstructed_image)
# cv2.waitKey(0)


# # Calculate average colors for each strip
# average_colors = []
# for i in range(num_strips):
#     strip = concatenated_image[:, i * strip_width: (i + 1) * strip_width]
#     average_color = np.mean(strip, axis=(0, 1))  # Calculate average color
#     average_colors.append((average_color, i))
#
# # Sort strips based on average color
# sorted_strips = sorted(average_colors, key=lambda x: np.sum(x[0]))
#
# # Reassemble the image in the correct order
# reconstructed_image = np.zeros_like(concatenated_image)
# for i, (_, index) in enumerate(sorted_strips):
#     strip = concatenated_image[:, index * strip_width: (index + 1) * strip_width]
#     reconstructed_image[:, i * strip_width: (i + 1) * strip_width] = strip


# TODO:Part 2
# # Calculate edge information for each strip
# edges = []
# for i in range(num_strips):
#     strip = concatenated_image[:, i * strip_width: (i + 1) * strip_width]
#     gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
#     edges.append((cv2.Canny(gray_strip, 50, 150), i))
#
# # Set print options to print the entire array
# np.set_printoptions(threshold=np.inf)
#
# # Print the entire array
# print(edges[0])
#
# # Sort strips based on edge information
# sorted_strips = sorted(edges, key=lambda x: np.sum(x[0]))
#
# # Reassemble the image in the correct order
# reconstructed_image = np.zeros_like(concatenated_image)
# for i, (_, index) in enumerate(sorted_strips):
#     strip = concatenated_image[:, index * strip_width: (index + 1) * strip_width]
#     reconstructed_image[:, i * strip_width: (i + 1) * strip_width] = strip


# histograms = []
# for i in range(num_strips):
#     strip = concatenated_image[:, i * strip_width: (i + 1) * strip_width]
#     hist = cv2.calcHist([strip], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
#     histograms.append((hist, i))
#
# # Sort strips based on histogram comparison
# sorted_strips = sorted(histograms, key=lambda x: cv2.compareHist(x[0], histograms[0][0], cv2.HISTCMP_CHISQR))
#
# # Reassemble the image in the correct order
# reconstructed_image = np.zeros_like(concatenated_image)
# for i, (_, index) in enumerate(sorted_strips):
#     strip = concatenated_image[:, index * strip_width: (index + 1) * strip_width]
#     reconstructed_image[:, i * strip_width: (i + 1) * strip_width] = strip
#


# # Initialize ORB detector
# orb = cv2.ORB_create()
#
# # Find keypoints and descriptors for the first strip (assuming it's correctly placed)
# first_strip = concatenated_image[:, :strip_width]
# kp1, des1 = orb.detectAndCompute(first_strip, None)
#
# # Initialize a list to store the ordered keypoints and descriptors
# ordered_kp_des_pairs = [(kp1, des1)]
#
# # Match features for subsequent strips and align them with the first strip
# for i in range(1, num_strips):
#     # Extract features for the current strip
#     strip = concatenated_image[:, i * strip_width: (i + 1) * strip_width]
#     kp2, des2 = orb.detectAndCompute(strip, None)
#
#     # Find keypoint matches between the two strips
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     if des2 is None:
#         print(f"Descriptors couldn't be computed for strip {i+1}. Skipping...")
#         continue
#     matches = bf.match(des1, des2)
#
#     # Check if there are enough matched keypoints to compute homography
#     if len(matches) < 4:
#         print(f"Not enough matched keypoints for strip {i+1}. Skipping alignment...")
#         continue
#
#     # Sort matches based on distance
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     # Extract matched keypoints
#     matched_kp1 = np.array([kp1[match.queryIdx].pt for match in matches])
#     matched_kp2 = np.array([kp2[match.trainIdx].pt for match in matches])
#
#     # Find the transformation matrix using RANSAC
#     M, _ = cv2.findHomography(matched_kp2, matched_kp1, cv2.RANSAC)
#
#     # Warp the current strip to align it with the first strip
#     warped_strip = cv2.warpPerspective(strip, M, (strip_width, strip.shape[0]))
#
#     # Update keypoints and descriptors for the next iteration
#     kp1, des1 = orb.detectAndCompute(warped_strip, None)
#
#     # Store ordered keypoints and descriptors
#     ordered_kp_des_pairs.append((kp1, des1))
#
# # Reassemble the image using the ordered keypoints and descriptors
# reconstructed_image = np.zeros_like(concatenated_image)
# for i, (kp, des) in enumerate(ordered_kp_des_pairs):
#     strip = concatenated_image[:, i * strip_width: (i + 1) * strip_width]
#     reconstructed_image[:, i * strip_width: (i + 1) * strip_width] = strip
#
#
# # Initialize BRISK detector
# brisk = cv2.BRISK_create()
#
# # Find keypoints and descriptors for the first strip (assuming it's correctly placed)
# first_strip = concatenated_image[:, :strip_width]
# kp1, des1 = brisk.detectAndCompute(first_strip, None)
#
# # Initialize a list to store the ordered keypoints and descriptors
# ordered_kp_des_pairs = [(kp1, des1)]
#
# # Match features for subsequent strips and align them with the first strip
# for i in range(1, num_strips):
#     # Extract features for the current strip
#     strip = concatenated_image[:, i * strip_width: (i + 1) * strip_width]
#     kp2, des2 = brisk.detectAndCompute(strip, None)
#
#     # If descriptors couldn't be computed for the current strip, skip it
#     if des2 is None:
#         print(f"Descriptors couldn't be computed for strip {i + 1}. Skipping...")
#         continue
#
#     # Find keypoint matches between the two strips
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#
#     # Apply ratio test to filter good matches
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)
#
#     # Check if there are enough good matches to compute homography
#     if len(good_matches) < 4:
#         print(f"Not enough good matches for strip {i + 1}. Skipping alignment...")
#         continue
#
#     # Extract matched keypoints
#     matched_kp1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     matched_kp2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#
#     # Find the transformation matrix using RANSAC
#     M, _ = cv2.findHomography(matched_kp2, matched_kp1, cv2.RANSAC)
#
#     # Warp the current strip to align it with the first strip
#     warped_strip = cv2.warpPerspective(strip, M, (strip_width, strip.shape[0]))
#
#     # Update keypoints and descriptors for the next iteration
#     kp1, des1 = brisk.detectAndCompute(warped_strip, None)
#
#     # Store ordered keypoints and descriptors
#     ordered_kp_des_pairs.append((kp1, des1))
#
# # Reassemble the image using the ordered keypoints and descriptors
# reconstructed_image = np.zeros_like(concatenated_image)
# for i, (kp, des) in enumerate(ordered_kp_des_pairs):
#     strip = concatenated_image[:, i * strip_width: (i + 1) * strip_width]
#     reconstructed_image[:, i * strip_width: (i + 1) * strip_width] = strip
#
# # Display or save the reconstructed image
# cv2.imshow('Reconstructed Image', reconstructed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

message = "hellohellohellohello"
# Split the message into chunks
num_of_chunks = 3
chunks_size = math.ceil(len(message) / num_of_chunks)
message_chunks = [message[i:min(len(message), i + chunks_size)] for i in range(0, len(message), chunks_size)]
print(message_chunks)


#
# riddles_to_solve = ['cv_medium', 'cv_hard', 'sec_medium_stegano', 'sec_hard', 'problem_solving_easy',
#                     'problem_solving_medium', 'problem_solving_hard']
# for i in range(len(riddles_to_solve)):
#     print(riddles_to_solve[i])
def generate_message_array(message, image_carrier):
    '''
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier
    '''
    # Split the message into chunks
    num_of_chunks = 3
    chunks_size = math.ceil(len(message) / num_of_chunks)
    message_chunks = [message[i:min(len(message), i + chunks_size)] for i in range(0, len(message), chunks_size)]
    fake_message = "HelloFromTheOtherSid"
    fake_message_chunks = [fake_message[i:min(len(fake_message), i + chunks_size)] for i in
                           range(0, len(fake_message), chunks_size)]

    result_array = [0] * 9  # Create an array of size 9 filled with zeros

    # Insert elements from array1 into specific positions
    result_array[0] = (encode(image_carrier, message_chunks[0])).tolist()
    result_array[4] = (encode(image_carrier, message_chunks[1])).tolist()
    result_array[8] = (encode(image_carrier, message_chunks[2])).tolist()
    #
    # # Insert elements from array2 into remaining positions
    for i in range(1, 5):
        result_array[i] = (encode(image_carrier, fake_message_chunks[i % 3])).tolist()
        result_array[i + 4] = (encode(image_carrier, fake_message_chunks[(i + 4) % 3])).tolist()
    return result_array

image = cv2.imread('C:\\Users\\bodyo\\OneDrive\\Documents\\GitHub\\HackTrick24\\SteganoGAN\\sample_example\\encoded.png')

# Convert to NumPy array
image_array = np.array(image)

message_array = [0,1,2,3,4,5,6,7,8]
for i in range(3):
    print(message_array[i * 3:i * 3 + 3])

solution = riddle_solvers['ml_easy']([1])
