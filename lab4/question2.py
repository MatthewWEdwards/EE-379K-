import scipy.misc

img_mat = scipy.misc.imread('mona_lisa.png', flatten=True) #read the image in grayscale
u, s, v = np.linalg.svd(img_mat, full_matrices=False) #use svd

#create three copies of s and zeros pad them after k = 2, k = 5, and k = 10 respectively
s2 = s.copy()
s2 = np.concatenate([np.array(s2[:2]), np.zeros(len(s2[2:]))], axis=0) 
s5 = s.copy()
s5 = np.concatenate([np.array(s5[:5]), np.zeros(len(s5[5:]))], axis=0)
s10 = s.copy()
s10 = np.concatenate([np.array(s10[:10]), np.zeros(len(s10[10:]))], axis=0)

#build the image again using s2, s5, and s10
m2 = u.dot(np.diag(s2)).dot(v)
m5 = u.dot(np.diag(s5)).dot(v)
m10 = u.dot(np.diag(s10)).dot(v)

# k = 2
plt.imshow(m2, cmap = plt.get_cmap('gray'))
plt.show()

# k = 5
plt.imshow(m5, cmap = plt.get_cmap('gray'))
plt.show()

# k = 10
plt.imshow(m10, cmap = plt.get_cmap('gray'))
plt.show()

print u.shape
print v.shape

#size of image for k = 2:
print (2 * 603 + 2 + 2 * 400) * 64

#size of image for k = 5:
print (5 * 603 + 5 + 5 * 400) * 64

#size of image for k = 10:
print (10 * 603 + 10 + 10 * 400) * 64

#original:
print 16 * 603 * 400