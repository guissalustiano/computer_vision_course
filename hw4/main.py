import numpy as np
import cv2 as cv
import scipy
import skimage
from math import ceil
import matplotlib.pyplot as plt


cap = cv.VideoCapture("original.mp4")
fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# from https://stackoverflow.com/a/46892763
def gkern(sigma=5, kernlen=None):
    """Returns a 2D Gaussian kernel array."""
    kernlen = kernlen or (2*ceil(4*sigma)+1) # pdf = 0.95
    gkern1d = scipy.signal.windows.gaussian(kernlen, std=sigma).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

# Based on from https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
def fft2(img):
  dft = np.fft.fft2(img)
  dft_shift = np.fft.fftshift(dft)
  magnitude_spectrum = 20*np.log(np.abs(dft_shift))

  return dft_shift, magnitude_spectrum

def ifft2(dft_shift):
    dft = np.fft.ifftshift(dft_shift)
    idft = np.fft.ifft2(dft)
    img_back = np.abs(idft)

    return img_back

def get_circle_mask(shape, radius):
    mask = np.zeros(frame.shape, dtype=np.bool_)
    rr,cc = skimage.draw.disk(shape, radius)
    mask[rr,cc] = True
    return mask

def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv.CAP_PROP_POS_MSEC)) < upper

def write_text(frame, text: str):
    cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

old_gray = None
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        if cv.waitKey(28) & 0xFF == ord('q'):
            break

        timestamp = int(cap.get(cv.CAP_PROP_POS_MSEC))//1000
        if timestamp < 79:
            continue

        if timestamp < 5:
            pass
            write_text(frame, "Original")
        elif timestamp < 10:
            # Gaussian filter
            filter = gkern(5)
            filter /= filter.sum()
            frame = cv.filter2D(frame, -1, filter)
            write_text(frame, "Gaussian filter")
        elif timestamp < 15:
            # Sharpen filter
            filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            frame = cv.filter2D(frame, -1, filter)
            write_text(frame, "Sharpen filter")
        elif timestamp < 20:
            # Sobel X
            frame = np.float32(frame)/255.
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
            frame = cv.Sobel(frame,cv.CV_32F,1,0,ksize=5)
            frame = cv.cvtColor(frame,cv.COLOR_GRAY2RGB)
            frame = cv.normalize(frame, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
 
            write_text(frame, "Sobel X")
        elif timestamp < 25:
            # Sobel Y
            frame = np.float32(frame)/255.
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
            frame = cv.Sobel(frame,cv.CV_32F,0,1,ksize=5)
            frame = cv.cvtColor(frame,cv.COLOR_GRAY2RGB)
            frame = cv.normalize(frame, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

            write_text(frame, "Sobel Y")
        elif timestamp < 30:
            # gradient magnitude
            frame = np.float32(frame)/255.
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 

            sobelx = cv.Sobel(frame,cv.CV_32F,1,0,ksize=5)
            sobely = cv.Sobel(frame,cv.CV_32F,0,1,ksize=5)
            frame = np.sqrt(sobelx**2 + sobely**2)

            frame = cv.cvtColor(frame,cv.COLOR_GRAY2RGB)
            frame = cv.normalize(frame, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

            write_text(frame, "Gradient magnitude")

        elif timestamp < 35:
            # Canny
            tresholds = (100, 200)
            frame = cv.Canny(frame, *tresholds)
            frame = cv.cvtColor(frame,cv.COLOR_GRAY2RGB)
            write_text(frame, f"Canny with tresholds {tresholds}")

        elif timestamp < 40:
            # fft
            frame = np.float32(frame)/255.

            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, frame = fft2(frame)

            frame = cv.cvtColor(frame.astype(np.float32), cv.COLOR_GRAY2RGB)
            frame = cv.normalize(frame, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
            write_text(frame, "FFT log magnitude")

        elif timestamp < 50:
            # fft low pass filter
            frame = np.float32(frame)/255.

            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_fft, frame_magnitude = fft2(frame)

            circle_mask = get_circle_mask((frame.shape[0]//2, frame.shape[1]//2), max(frame.shape)//4)

            frame_fft[~circle_mask] = 0
            frame_magnitude[~circle_mask] = 0

            if timestamp < 45:
                frame = frame_magnitude
            else:
                frame = ifft2(frame_fft)

            frame = cv.cvtColor(frame.astype(np.float32), cv.COLOR_GRAY2RGB)
            frame = cv.normalize(frame, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

            if timestamp < 45:
                write_text(frame, "FFT low pass filter log magnitude")
            else:
                write_text(frame, "FFT low pass filter spatial domain")

        elif timestamp < 60:
            # fft high pass filter
            frame = np.float32(frame)/255.

            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_fft, frame_magnitude = fft2(frame)
            
            circle_mask = get_circle_mask((frame.shape[0]//2, frame.shape[1]//2), max(frame.shape)//4)

            frame_fft[circle_mask] = 0
            frame_magnitude[circle_mask] = 0

            if timestamp < 55:
                frame = frame_magnitude
            else:
                frame = ifft2(frame_fft)

            frame = cv.cvtColor(frame.astype(np.float32), cv.COLOR_GRAY2RGB)
            frame = cv.normalize(frame, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

            if timestamp < 50:
                write_text(frame, "FFT high pass filter log magnitude")
            else:
                write_text(frame, "FFT high pass filter spatial domain")
        elif timestamp < 70:
            # fft high pass filter
            frame = np.float32(frame)/255.

            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_fft, frame_magnitude = fft2(frame)

            mask = np.zeros(frame.shape, dtype=np.bool_)
            height, width = frame.shape
            mask[:,(width//5):(2*width//5)] = True
            mask |= mask[::,::-1]


            frame_fft[mask] = 0
            frame_magnitude[mask] = 0

            if timestamp < 65:
                frame = frame_magnitude
            else:
                frame = ifft2(frame_fft)

            frame = cv.cvtColor(frame.astype(np.float32), cv.COLOR_GRAY2RGB)
            frame = cv.normalize(frame, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

            if timestamp < 65:
                write_text(frame, "FFT high pass filter log magnitude")
            else:
                write_text(frame, "FFT high pass filter spatial domain")

        elif timestamp < 75:
            # Template matching
            template = cv.imread("template.png", cv.IMREAD_GRAYSCALE)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.matchTemplate(frame, template, cv.TM_CCOEFF_NORMED)


            frame = cv.cvtColor(frame.astype(np.float32), cv.COLOR_GRAY2RGB)
            frame = cv.normalize(frame, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

            height,width = template.shape
            template = cv.cvtColor(template.astype(np.float32), cv.COLOR_GRAY2RGB)
            frame[:height, -width:] = template

            write_text(frame, "Template matching")
        elif timestamp < 80:
            template = cv.imread('template.png', cv.IMREAD_GRAYSCALE)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            height,width = template.shape
            res = cv.matchTemplate(frame_gray,template,cv.TM_CCOEFF_NORMED)
            threshold = 0.5
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                cv.rectangle(frame, pt, (pt[0] + width, pt[1] + height), (0,0,255), 2)

            template = cv.cvtColor(template.astype(np.float32), cv.COLOR_GRAY2RGB)
            frame[:height, -width:] = template
            write_text(frame, "Template matching rectangles")
        elif timestamp < 90:
            # Optical flow
            if old_gray is None:
                # flow parameters
                feature_params = dict( maxCorners = 500,
                 qualityLevel = 0.5,
                 minDistance = 20,
                 blockSize = 7 )

                # Parameters for lucas kanade optical flow
                lk_params = dict( winSize = (15, 15),
                 maxLevel = 2,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

                # Create some random colors
            color = (0, 255, 0)

            old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
            p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
            # Calculate Optical Flow
            p1, st, err = cv.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )
            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
         
            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)
                frame = cv.arrowedLine(frame, (a, b), (c, d), color, 2)
         
            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            break


        out.write(frame)
        cv.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        # if cv.waitKey(25) & 0xFF == ord('q'):
        #     break

    else:
        break
    
    old_frame = frame

# When everything done, release the video capture and writing object
cap.release()
out.release()
# Closes all the frames
cv.destroyAllWindows()
