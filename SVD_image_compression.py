## SVD image compression example, 
## based on https://inst.eecs.berkeley.edu/~ee127a/book/login/l_svd_apps_image.html
## Eli, 201510
import numpy as np
from numpy import linalg
import scipy as scipy
import matplotlib.pyplot as plt
import matplotlib
from skimage.color import rgb2gray;


print('')
print('QUESTION 4B')
print(' 60*(N+M+1)/(N*M)')

print('')
print('QUESTION 4C')
print('!!!!')

## read data:
A = plt.imread('Baboon_or_Mandrill.bmp'); # loads image in integer format as a 3d-array
A=rgb2gray(A);
A = np.double(A); # transform to real values
N=np.size(A[:,1]);

## show original image:
ifig=0;
ifig=ifig+1; plt.figure(ifig); plt.clf(); plt.imshow(A); plt.set_cmap('gray');
plt.title('Original image, size='+repr(N)+'x'+repr(N));
plt.pause(0.05)
wait=input("Enter to continue...")

## the SVD:
U,S,V = scipy.linalg.svd(A);

## plot singular values:
ifig=ifig+1; plt.figure(ifig); plt.clf()
plt.subplot(1,2,1)
plt.plot(S)
plt.title('singular values')
plt.xlabel('n')
plt.ylabel('$\sigma_n$')

plt.subplot(1,2,2)
plt.semilogy(S)
plt.title('log singular values')
plt.xlabel('n')
plt.ylabel('$\log(\sigma_n)$')
plt.pause(0.05)
wait=input("Enter to continue...")


## reconstruct using different low-rank approximations:
ifig=ifig+1; 
for k in [1,3,5,10,20,30,50,80,100,120,300,N]:
    fig=plt.figure(ifig); plt.clf();
  
    ## low-rank SVD reconstruction using k degrees:
    # note that in python A=U*S*V rather than V^T
    Ak = U[:,0:k]@np.diag(S[0:k])@V[0:k,:];
    compression_ratio=100*N*k*2/N**2;

    ## Two methods to calculate explained variance; I verified they
    ## are the same during debugging:
    explained_variance=100*sum(S[0:k]**2)/sum(S[:]**2);
    explained_variance2=100*sum(Ak**2)/sum(A*A);
    
    ## show reconstructed image:
    plt.imshow(Ak); plt.set_cmap('gray');
    plt.title('k='+repr(k)+', compression='+repr(compression_ratio)+', explained var='+repr(explained_variance));
    plt.text(0.1,0.1,'SVD reconstruction')
    fig.canvas.flush_events()
    plt.pause(0.05)
    wait=input("Enter to continue...")
