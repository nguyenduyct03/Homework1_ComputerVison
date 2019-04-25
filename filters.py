import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE
    ph = Hk//2
    pw = Wk//2
    for y in range(Hi):
        for x in range(Wi):
            sum = 0.0
            for i in range(Hk-1,-1,-1):
                for j in range(Wk-1,-1,-1):
                    if (y-i+ph < 0 or y-i+ph > Hi-1 or x-j+pw < 0 or x-j+pw > Wi-1):
                        continue
                    else:
                        sum += kernel[i,j]*image[y-i+ph,x-j+pw]
            out[y,x] = sum
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = np.zeros((H+2*pad_height, W+2*pad_width))

    ### YOUR CODE HERE
    for y in range(0,H):
        for x in range(0,W):
            out[y+pad_height,x+pad_width] = image[y,x]
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    img = zero_pad(image,Hk//2,Wk//2)
    kernel = np.flip(np.flip(kernel,0),1)
    for y in range(Hi):
        for x in range(Wi):
            out[y,x] = np.sum(kernel*img[y:y+Hk,x:x+Wk])
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    # Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g1 = np.flip(np.flip(g,1),0)
    out = conv_fast(f,g1)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    # Padding f, out with zeros
    fd = zero_pad(f,g.shape[0]//2,g.shape[1]//2)
    out = np.zeros(f.shape)
    # Calculating mean value of both image
    mean_f = np.mean(f)
    mean_g = np.mean(g)
    # Calculating sd of both image
    std_f = np.std(f)
    std_g = np.std(g)
    # Template
    kg = g-mean_g
    # ZNCC
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            kf = fd[y:y+g.shape[0],x:x+g.shape[1]]-mean_f
            out[y,x] = np.sum(kg*kf)/(std_f*std_g)
    ### END YOUR CODE
    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    # Padding f, out with zeros
    fd = zero_pad(f,g.shape[0]//2,g.shape[1]//2)
    out = np.zeros(f.shape)
    # Calculating mean value of template
    mean_g = np.mean(g)
    # Calculating sd of template
    std_g = np.std(g)
    # Template
    kg = (g-mean_g)/std_g
    # NCC
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            # Calculating mean and sd of sub image
            mean_f = np.mean(fd[y:y+g.shape[0],x:x+g.shape[1]])
            std_f = np.std(fd[y:y+g.shape[0],x:x+g.shape[1]])
            # NCC
            kf = (fd[y:y+g.shape[0],x:x+g.shape[1]]-mean_f)/std_f
            out[y,x] = np.sum(kg*kf)
    ### END YOUR CODE
    return out