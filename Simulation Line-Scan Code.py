import numpy as np
import cv2
from skimage.io import imshow, imread
import matplotlib.pyplot as plt
import os  #functions for creating/removing a directory(folder)
import fnmatch
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
import matplotlib.pylab as plt
import matplotlib.image as mpimg 

# =============================================================================
# Define all paths to save files in
# =============================================================================
# this is for the raw scanned images
path = 'C:/Users/RaphaelShiQi/anaconda3/Linescanphotos' 
# this is for the concatenated images
path1 = 'C:/Users/RaphaelShiQi/anaconda3/Linescanphotos/Concatenated'
# this is for the image processing codes
path2 = 'C:/Users/RaphaelShiQi/anaconda3/CTmetrix image processing codes'
# this is for the grayscaled converted images
path3 = 'C:/Users/RaphaelShiQi/anaconda3/Linescanphotos/Concatenated/Grayscale'
# for radon transformed images
path4 = 'C:/Users/RaphaelShiQi/anaconda3/Linescanphotos/Concatenated/Grayscale/RadonTransform'

cam = cv2.VideoCapture(0)  #select webcam 
cv2.namedWindow("CTmetrix test")  #set window name

#ret is a boolean variable that returns true if the frame is available.
#frame is an image array vector captured based on the default frames per second defined explicitly or implicitly

# =============================================================================
# Parameters
# =============================================================================
max_number_of_images = 100
img_counter = 0

# # =============================================================================
# # Capture up to the given maximum number of images
# # =============================================================================
while img_counter < max_number_of_images:
    ret, frame = cam.read()
    # if the camera fails to take a photo i.e. ret value is false
    if ret == False:
        print("failed to grab frame")
        break

    cv2.imshow("CTmetrixWindows", frame)  #showing the big window 
 
    k = cv2.waitKey(1)
    if k%256 == 27:     # press ESCAPE button
        print("Escape hit, closing...")
        break
    
    # taking a screenshot
    elif k%256 == 32:     # press SPACE button   
        img_name = "{}CTmetrix.png".format('%02d' %img_counter)  #'%02d' format to 2 digits
        cv2.imwrite(img_name, frame)    #save it in the directory the code is in
        print("{} written!".format(img_name))
        img_counter += 1
        
        # cropping image to slice
        width = frame.shape[0]    #480
        height = frame.shape[1]      #640
        mid_w = round(width/2)   #240
        mid_h = round(height/2)   #320   # print(".shape data type: ", type(mid_y)) #.shape data type:  <class 'int'>
        halflinewidth = 20     # print(width , height, mid_w, mid_h)
        cropped_frame = frame[:height, mid_h - halflinewidth:mid_h + halflinewidth] #take at the center
        print("Cropped Dimensions: ", cropped_frame.shape)
        cv2.imshow("cropped", cropped_frame)  # Display cropped image on a window
        
        # Saves the photo immediately without an extra spacebar
        path = 'C:/Users/RaphaelShiQi/anaconda3/Linescanphotos'
        cv2.imwrite(os.path.join(path , str('%02d' % img_counter)+ ' linescan_'+ '.png'), cropped_frame)     

# =============================================================================
# Concatenate linescans and save
# =============================================================================
ls_img = fnmatch.filter(os.listdir(path), '*.png') # image [list]
# print out the original list of images
print(ls_img)
# create an empty lists of images
ls3_img = [] 

# counter
i = 0
# number of photos to concatenate
number = 10
# calculate the number of loops of concatenation
number_of_concatenated_images = round(max_number_of_images/number)

# Concatenation loop
for i in range(number_of_concatenated_images):
    ls2_img = []
    # obtain a set of images e.g. 10 images
    # every ith set will start from i*number inclusive to (i+1)*number exclusive
    # e.g. the second set (i=1) starts from image 10 to image 19
    for img in ls_img[i*number: (i+1)*number]:      
        ls2_img.append(cv2.imread(os.path.join(path, img))) #imread returns image file
        image_combined = cv2.hconcat(ls2_img)  #hconcat is for horizontal concatination while vconhat is for vertical concatination
        cv2.imwrite(os.path.join(path1 , 'Combined'+str(i)+'.png'), image_combined)
        
        # convert image to greyscale and add into another folder
        img_gray = cv2.cvtColor(image_combined, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(path3 , 'grayscale'+str(i)+'.png'), img_gray )
        
    # plot images to show
    # first convert bgr used in cv2 to rgb used in matplotlib to plot correct colors
    image_combined_flipped = cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB)
    img_gray_flipped = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
    
    
    fig , (ax3, ax4) = plt.subplots(1,2 , figsize = (9,5))
    ax3.set_title("concatenated image with colour")
    ax3.imshow(image_combined_flipped)
    
    ax4.set_title("concatenated grayscale")
    ax4.imshow(img_gray_flipped)
    
#============================ Reconstruction ===========================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

    ax1.set_title("concatenated grayscale")
    ax1.imshow(img_gray, cmap=plt.cm.Greys_r)

    theta = np.linspace(0., 180., max(img_gray.shape), endpoint=False)
    sinogram = radon(img_gray , theta=theta)
    dx, dy = 0.5 * 180.0 / max(img_gray .shape), 0.5 / sinogram.shape[0]
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
                extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
                aspect='auto')
    
    cv2.imwrite(os.path.join(path4 , 'Reconstructed'+str(i)+'.png'), sinogram)
    
    fig.tight_layout()
    plt.show()

#close camera and stops code 
cam.release()
cv2.destroyAllWindows()