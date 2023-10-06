from math import floor, ceil
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# From lecture notes
def psf2otf(psf, shape):
  M, N = shape
  # get size of psf
  sz = np.shape(psf)
  # calculate needed paddings
  n = N - sz[1]
  m = M - sz[0]
  n1 = int(np.floor(n / 2) + np.mod(n, 2))
  n2 = int(np.floor(n / 2))
  m1 = int(np.floor(m / 2) + np.mod(m, 2))
  m2 = int(np.floor(m / 2))
  # pad array with zeros
  psf = np.pad(psf, ((m1, m2), (n1, n2)))
  # shift origin
  psf = np.fft.ifftshift(psf)
  # apply DFT
  otf = np.fft.fft2(psf)
  return otf

def show_image(img, title, filename = None, show_windows = True, centrylized = False):
  [img_size_y, img_size_x] = img.shape
  extent =  [-img_size_x/2, img_size_x/2, -img_size_y/2, img_size_y/2] if centrylized else None

  plt.imshow(img, cmap = 'gray', extent=extent)
  plt.title(title)

  if filename:
    plt.savefig(filename)
  plt.show()

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

def main():
    # read image
    img = cv.imread('car_dis.png', cv.IMREAD_GRAYSCALE)

    # convert to float64
    img = np.float64(img)/255.

    # usefull variables
    [img_size_y, img_size_x] = img.shape
    img_center_x, img_center_y = (img_size_x//2, img_size_y//2)

    # show original image
    show_image(img, 'Original image', filename = 'original.png')

    # Calculate FFT of original image
    dft, magnitude_spectrum = fft2(img)
    show_image(magnitude_spectrum, 'Magnitude Spectrum', 'Imlogmag.png', centrylized=True)

    # Plot one line to easy count lambda
    plt.plot(np.arange(img_size_x), img[0])
    plt.title('First line of original image')
    plt.show()
    lambda_ = 9 # Based on previous plot

    # Calculate noise position
    noise_fft_y = img_center_y
    noise_fft_x = img_center_x + img_size_x//lambda_
    print('Noise position in FFT:', noise_fft_x, noise_fft_y)
    print('Noise value in FFT:', magnitude_spectrum[noise_fft_y, noise_fft_x])

    # Apply average filter
    average_filter = np.ones((1, lambda_))
    average_filter /=  average_filter.sum()

    imgfil = cv.filter2D(src=img, ddepth=-1, kernel=average_filter, borderType=cv.BORDER_REPLICATE)
    show_image(imgfil, 'Image using average filter', filename='ImFil.png')

    # Calculate FFT of filtered image
    dft_fil, magnitude_spectrum_fil = fft2(imgfil)
    show_image(magnitude_spectrum_fil, 'Magnitude Spectrum After average filter', filename='SpectrumFil.png', centrylized=True)
    print('Noise value in FFT after average filter:', magnitude_spectrum_fil[noise_fft_y, noise_fft_x])

    # Calculate OTF
    otf = psf2otf(average_filter, img.shape)
    mtf = np.abs(otf)

    show_image(mtf, 'MTF average filter', 'MtfFil.png', centrylized=True)
    print("sum of otf's imaginary part:", np.sum(np.abs(otf.imag)))

    # Calculate atenuation factor
    atenuation_factor = magnitude_spectrum_fil[noise_fft_y, noise_fft_x] - magnitude_spectrum[noise_fft_y, noise_fft_x]
    print('Atenuation factor:', atenuation_factor)

    # Create FFT filter
    s = 24
    rstart = noise_fft_y - floor(s/2)
    rend =  noise_fft_y + ceil(s/2)
    cstart = noise_fft_x - floor(s/2)
    cend = noise_fft_x + ceil(s/2)
    H = np.ones_like(img)
    H[rstart:rend, cstart:cend] = 0
    H *= H[:,::-1]
    show_image(H, 'Mask "H', filename="H_ftt.png", centrylized=True)

    # Apply FFT filter
    dft_for = dft.copy() * H
    magnitude_spectrum_for = 20*np.log(np.abs(dft_for))
    show_image(magnitude_spectrum_for, 'Magnitude Spectrum after FFT filter', filename="fft_mag.png", centrylized=True)

    # Apply inverse FFT
    img_back_for = ifft2(dft_for)
    show_image(img_back_for, 'Image after FFT filter', filename='forrier.png')
    print("sum of img_back_for's imaginary part:", np.sum(np.abs(otf.imag)))


if __name__ == '__main__':
    main()
