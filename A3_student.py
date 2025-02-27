import numpy as np
import math
import random
from matplotlib import pyplot as plt
from skimage import io, color, img_as_float, transform
from sklearn.cluster import KMeans
from scipy import spatial
from skimage.color import gray2rgb, rgb2gray
from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage.transform import SimilarityTransform
from skimage.color import rgb2gray
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
#from skimage.feature import match_descriptors, ORB, plot_matches
from skimage import feature


#TODO: part 1 here
def part1():
    filename_Grayimage = 'PeppersBayerGray.bmp'
    filename_gridB = 'gridB.bmp'
    filename_gridR = 'gridR.bmp'
    filename_gridG = 'gridG.bmp'
    img = io.imread(filename_Grayimage, as_gray =True)

    h,w = img.shape

    # our final image will be a 3 dimentional image with 3 channels
    rgb = np.zeros((h,w,3),np.uint8)

    # reconstruction of the IG channel IG
    IG = np.copy(img) # copy the image into each channel
    IR = np.copy(img)
    IB = np.copy(img)

    print("original ",img[:4])
    
    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.
            # TODO: compute pixel value for each location where mask is unshaded (0)
            # interpolate each pixel using its every valid (shaded) neighbour
            IG[row,col+1]= (int(img[row,col])+int(img[row,col+2])+int(img[row+1,col+1]))/3  # B (recommendation: add this kinf of inline comments to each line within for loop)
            IG[row,col+3] = (int(img[row,col+2]) + int(img[row+1,col+3]))/2         #D
            IG[row+1,col] = (int(img[row,col]) + int(img[row+1,col+1]) + int(img[row+2,col]))/3       #E
            IG[row+1,col+2] = (int(img[row,col+2]) + int(img[row+1,col+1]) + int(img[row+1,col+3]) + int(img[row+2,col+2]))/4       #G
            IG[row+2,col+1] = (int(img[row+2,col]) + int(img[row+2,col+2]) + int(img[row+1,col+1]) + int(img[row+1,col+3]))/4         #J
            IG[row+2,col+3] = (int(img[row+2,col+2]) + int(img[row+1,col+3]) + int(img[row+3,col+3]))/3       #L
            IG[row+3,col]= (int(img[row+2,col])+int(img[row+3,col+1]))/2                    # M
            IG[row+3,col+2]= (int(img[row+3,col+1])+int(img[row+2,col+2])+int(img[row+3,col+3]))/3      #O
            
            #print(IG)
            # TODO: reconstruction of the red channel IR (similar to loops above),

            IR[row,col] = img[row,col+1]        #A
            IR[row,col+2] = (int(img[row,col+1]) + int(img[row,col+3]))/2       #C
            IR[row+1,col+1] = (int(img[row,col+1]) + int(img[row+2,col+1]))/2       #F
            IR[row+1,col] = IR[row+1,col+1]         #E
            IR[row+1,col+2] = (int(img[row,col+1]) + int(img[row,col+3]) + int(img[row+2,col+1]) + int(img[row+2,col+3]))/4         #G
            IR[row+1,col+3] = (int(img[row,col+3]) + int(img[row+2,col+3]))/2       #H
            IR[row+2,col+2] = (int(img[row+2,col+1]) + int(img[row+2,col+3]))/2
            IR[row+2,col],IR[row+3,col],IR[row+3,col+1] = img[row+2,col+1],img[row+2,col+1],img[row+2,col+1]        #I,M,N
            IR[row+3,col+2] = IR[row+2,col+2]       #O
            IR[row+3,col+3] = img[row+2,col+3]      #P


            # TODO: reconstruction of the blue channel IB (similar to loops above),
            
            IB[row+1,col+1] = (int(img[row+1,col]) + int(img[row+1,col+2]))/2       #F
            IB[row+2,col] = (int(img[row+1,col]) + int(img[row+3,col]))/2       #I
            IB[row+2,col+2] = (int(img[row+1,col+2]) + int(img[row+3,col+2]))/2         #K
            IB[row+3,col+1] = (int(img[row+3,col]) + int(img[row+3,col+2]))/2       #N
            IB[row+2,col+1] = (int(img[row+1,col]) + int(img[row+3,col]) + int(img[row+1,col+2]) + int(img[row+3,col+2]))/4
            IB[row,col] = img[row+1,col]        #A
            IB[row,col+1] = IB[row+1,col+1]         #B
            IB[row,col+2],IB[row,col+3],IB[row+1,col+3] = img[row+1,col+2],img[row+1,col+2],img[row+1,col+2]        #C,D,H
            IB[row+2,col+3] = IB[row+2,col+2]       #L
            IB[row+3,col+3] = img[row+3,col+2]      #P

    print("green",IG[:4][:4])
    print("red",IR[:4][:4])
    print("blue",IB[:4][:4])
    # TODO: merge the three channels IG, IB, IR in the correct order
    rgb[:,:,1]=IG
    rgb[:,:,0]=IR
    rgb[:,:,2]=IB
    #...

    # plotting code
    plt.figure(figsize=(10,8))
    plt.subplot(221), plt.imshow(IG, cmap='gray'),plt.title('IG')
    plt.subplot(222), plt.imshow(IR, cmap='gray'),plt.title('IR')
    plt.subplot(223), plt.imshow(IB, cmap='gray'),plt.title('IB')

    plt.subplot(224), plt.imshow(rgb),plt.title('rgb')
    plt.show()


#TODO: part 2 here
def part2():

    # Finds the closest colour in the palette using kd-tree.
    def nearest(palette, colour):
        dist, i = palette.query(colour)
        return palette.data[i]


    # Make a kd-tree palette from the provided list of colours
    def makePalette(colours):
        #print(colours)
        return spatial.KDTree(colours)


    # Dynamically calculates and N-colour palette for the given image
    # Uses the KMeans clustering algorithm to determine the best colours
    # Returns a kd-tree palette with those colours
    def findPalette(image, nColours):
        # TODO: perform KMeans clustering to get 'colours' --  the computed k means
        print(image.shape)
        (h,w,c)=image.shape
        k_means = KMeans(n_clusters=nColours,max_iter=200).fit(image.reshape(h*w,c))     #referenve - https://hackernoon.com/learn-k-means-clustering-by-quantizing-color-images-in-python
        colours= (k_means.cluster_centers_)
        #print(img_as_float([colours.astype(np.ubyte)])[0])
        return makePalette(colours)


    def ModifiedFloydSteinbergDitherColor(image, palette):
        """
        The following pseudo-code for a grayscale image is grabbed from Wikipedia:
        https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering.
        The error distribution has been modified according to the instructions on eClass.

        total_abs_err := 0
        for each y from top to bottom ==> (height)
            for each x from left to right ==> (width)
                oldpixel  := image[x][y]
                newpixel  := nearest(oldpixel) # Determine the new colour for the current pixel from palette
                image[x][y]  := newpixel
                quant_error  := oldpixel - newpixel

                total_abs_err := total_abs_err + abs(quant_error)

                image[x + 1][y    ] := image[x + 1][y    ] + quant_error * 11 / 26
                image[x - 1][y + 1] := image[x - 1][y + 1] + quant_error * 5 / 26
                image[x    ][y + 1] := image[x    ][y + 1] + quant_error * 7 / 26
                image[x + 1][y + 1] := image[x + 1][y + 1] + quant_error * 3 / 26

        avg_abs_err := total_abs_err / image.size
        """

        # TODO: implement agorithm for RGB image (hint: you need to handle error in each channel separately)

        #print(image,image.shape)
        
        total_abs_error = 0   #RGB
        height, width =image.shape[0],image.shape[1]

        
        for y in range(1,height-1):
            for x in range(1,width-1):
                old = image[x,y].copy()
                #print(old)
                new = nearest(palette, old)
                #print(new)
                image[x,y] = new
                quant_err = old - new   
                #print(new, old, quant_err)
                total_abs_error += np.abs(quant_err)

                image[x+1,y] = image[x+1,y] + (quant_err * (11/26))
                image[x-1,y+1] = image[x-1,y+1] + (quant_err * (5/26))
                image[x,y+1] = image[x,y+1] + (quant_err * (7/26))
                image[x+1,y+1] = image[x+1,y+1] + (quant_err * (3/26))

        print(total_abs_error)
        avg_abs_error =(np.array(total_abs_error)/image.size)
        print(avg_abs_error)
        return image


    nColours = 7 # The number colours: change to generate a dynamic palette
    imfile = 'mandrill.png'
    image = io.imread(imfile)
    orig = image.copy()

    # Strip the alpha channel if it exists
    image = image[:,:,:3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    image = img_as_float(image)

    #print(image)

    # Dynamically generate an N colour palette for the given image
    palette = findPalette(image, nColours)
    colours = palette.data
    print("pallette= ",palette.data)
    colours = img_as_float([colours.astype(np.ubyte)])[0]
    print(colours)

    img = ModifiedFloydSteinbergDitherColor(image, palette)

    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(orig),plt.title('Original Image')
    plt.subplot(122),plt.imshow(img),plt.title(f'Dithered Image (nColours = {nColours})')
    plt.show()


#TODO: part 3 here
def part3():

    def read_image():
      original_img = io.imread('bird.jpeg')
      return original_img


    def calculate_trans_mat(image):
        """
        return translation matrix that shifts center of image to the origin and its inverse
        """
        h = image.shape[0]//2
        w = image.shape[1]//2
        
        #print(h, w)
        
        trans_mat = np.array([[1,0,-w],
                             [0,1,-h],
                             [0,0,1]])
        
        
        trans_mat_inv = np.linalg.inv(trans_mat)
        
        #print(trans_mat_inv)

        # TODO: implement this function (overwrite the two lines above)
        # ...

        return trans_mat, trans_mat_inv


    def rotate_image(image):
        ''' rotate and return image '''
        h, w = image.shape[:2]
        trans_mat, trans_mat_inv = calculate_trans_mat(image)

        angle = 75
        angle_rad = np.radians(angle)
        Tr = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                       [np.sin(angle_rad), np.cos(angle_rad), 0],
                       [0,0,1]])
        
        Tr_inv = np.linalg.inv(Tr)
        
        #to save on computation in each loop, we can multiple both rotation and translation beforehand and then take the inverse
        # reference = https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html#:~:text=A%20rotation%20matrix%20and%20a,rotations%20followed%20by%20a%20translation.
        combined_Tr_trans = np.matmul(Tr, trans_mat_inv)    
        combined_inverses = np.matmul(trans_mat, Tr_inv)
        
        
        out_img = np.zeros_like(image)
        for out_y in range(h):
            for out_x in range(w):
                # TODO: find input pixel location from output pixel ocation and inverse transform matrix, copy over value from input location to output location
                x_y_new = np.matmul(combined_Tr_trans, np.array([[out_x],[out_y],[1]]))
                print("x_y_new = ",x_y_new)
                out = np.matmul(combined_inverses, x_y_new)
                print("out = ",out)
                in_x, in_y = int(round(out[0, 0])), int(round(out[1, 0]))
                #print("x= ",in_x)
                #print("y= ",in_y)
                out_img[out_y, out_x] = image[in_y, in_x]
                

        return out_img, Tr


    def scale_image(image):
        ''' scale image and return '''
        # TODO: implement this function, similar to above
        out_img = np.zeros_like(image)
        Ts = np.array([])

        return out_img, Ts


    def skew_image(image):
        ''' Skew image and return '''
        # TODO: implement this function like above
        out_img = np.zeros_like(image)
        Tskew = np.array([])

        return out_img, Tskew


    def combined_warp(image):
        ''' implement your own code to perform the combined warp of rotate, scale, skew and return image + transformation matrix  '''
        # TODO: implement combined warp on your own.
        # You need to combine the transformation matrices before performing the warp
        # (you may want to use the above functions to get the transformation matrices)
        out_img = np.zeros_like(image)
        Tc = np.array([])

        return out_img, Tc


    def combined_warp_biinear(image):
        ''' perform the combined warp with bilinear interpolation (just show image) '''
        # TODO: implement combined warp -- you can use skimage.trasnform functions for this part (import if needed)
        # (you may want to use the above functions (above combined) to get the combined transformation matrix)
        out_img = np.zeros_like(image)

        return out_img


    # Plotting code here
    image = read_image()
    plt.imshow(image), plt.title("Oiginal Image"), plt.show()

    rotated_img, _ = rotate_image(image)
    plt.figure(figsize=(15,5))
    plt.subplot(131),plt.imshow(rotated_img), plt.title("Rotated Image")

    scaled_img, _ = scale_image(image)
    plt.subplot(132),plt.imshow(scaled_img), plt.title("Scaled Image")

    skewed_img, _ = skew_image(image)
    plt.subplot(133),plt.imshow(skewed_img), plt.title("Skewed Image"), plt.show()

    combined_warp_img, _ = combined_warp(image)
    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(combined_warp_img), plt.title("Combined Warp Image")

    combined_warp_biliear_img = combined_warp_biinear(image)
    plt.subplot(122),plt.imshow(combined_warp_biliear_img.astype(np.uint8)), plt.title("Combined Warp Image with Bilinear Interpolation"),plt.show()


# TODO: part 4 here
def part4():
    from skimage.color import rgb2gray
    from skimage import transform
    from skimage import io
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(100)

    filename1 = 'im1.jpg'
    filename2 = 'im2.jpg'

    image0 = io.imread(filename1, as_gray=True)
    image1 = io.imread(filename2, as_gray=True)

    plt.figure(figsize=(8,10))
    plt.subplot(321),plt.imshow(image0,cmap='gray'),plt.title("First Image")
    plt.subplot(322),plt.imshow(image1,cmap='gray'),plt.title("Second Image")

    # -------- Feature detection and matching -----

    # TODO: Initiate ORB detector
    from skimage.feature import ORB, match_descriptors
    # ...

    # TODO: Find the keypoints and descriptors
    # ...

    # TODO: initialize Brute-Force matcher and exclude outliers. See match descriptor function.
    # ...

    # -------- Transform estimation -------

    # TODO: Compute homography matrix using ransac and ProjectiveTransform
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    # ...
    # model_robust, inliers = ransac ...

    # ------------- Warping ----------------
    #Next, we produce the panorama itself. The first step is to find the shape of the output image by considering the extents of all warped images.

    r, c = image1.shape[:2]

    # Note that transformations take coordinates in
    # (x, y) format, not (row, column), in order to be
    # consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)

    # Find the extents of both the reference image and
    # the warped target image.
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1])

    #The images are now warped according to the estimated transformation model.
    #A shift is added to ensure that both images are visible in their entirety. Note that warp takes the inverse mapping as input.

    from skimage.color import gray2rgb
    from skimage.exposure import rescale_intensity
    from skimage.transform import warp
    from skimage.transform import SimilarityTransform

    offset = SimilarityTransform(translation=-corner_min)

    image0_ = warp(image0, offset.inverse,
                  output_shape=output_shape)

    image1_ = warp(image1, (model_robust + offset).inverse,
                  output_shape=output_shape)

    #An alpha channel is added to the warped images before merging them into a single image:

    def add_alpha(image, background=-1):
        """Add an alpha layer to the image.

        The alpha layer is set to 1 for foreground
        and 0 for background.
        """
        rgb = gray2rgb(image)
        alpha = (image != background)
        return np.dstack((rgb, alpha))

    # TODO: add alpha to the image0 and image1
    # ...

    # TODO: merge the alpha added image (only change the next line)
    # merged = ...
    alpha = merged[..., 3]
    merged /= np.maximum(alpha, 1)[..., np.newaxis]
    # The summed alpha layers give us an indication of
    # how many images were combined to make up each
    # pixel.  Divide by the number of images to get
    # an average.

    # Plotting code down here
    plt.subplot(323),plt.imshow(image0_, cmap="gray"),plt.title("Warped first image")
    plt.subplot(324),plt.imshow(image1_, cmap="gray"),plt.title("Warped second image")
    plt.show()

    plt.figure(figsize=(8,10))
    plt.imshow(merged, cmap="gray")
    plt.imsave('imgOut.png', merged)
    plt.show()

    from skimage.feature import match_descriptors, ORB, plot_matches
    import random
    #fig, ax = plt.subplots(nrows=1, ncols=1)
    fig = plt.figure(figsize=[10,10])
    plot_matches(
        plt.gca(),
        io.imread(filename1, as_gray=False),
        io.imread(filename2, as_gray=False),
        keypoints1,
        keypoints2,
        matches12[random.sample(range(matches12.shape[0]), k=10), :],
        only_matches = True
      )

    plt.show()


if __name__ == "__main__":
    # You can comment those lines to run only part of the code you are working on but remember to uncomment before submission
    #print("***************Output for part 1:")
    #part1()
    
    #print("***************Output for part 2:")
    #part2()
    
    print("***************Output for part 3:")
    part3()
    '''
    print("***************Output for part 4:")
    part4()'''
