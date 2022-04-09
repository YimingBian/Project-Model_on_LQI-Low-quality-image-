import matplotlib.pyplot as plt
import matplotlib.image as im

image_set = (
    'D:/Academic/2022Spring/575/Project/original/dog.jpg',
    "D:/Academic/2022Spring/575/Project/original/dog.jpg_SNP_0.1.JPEG",
    "D:/Academic/2022Spring/575/Project/original/dog.jpg_SNP_0.2.JPEG",
    "D:/Academic/2022Spring/575/Project/original/dog.jpg_SNP_0.3.JPEG",
    "D:/Academic/2022Spring/575/Project/original/dog.jpg_SNP_0.4.JPEG"
)

fig, ax = plt.subplots(1,5)
ax[ 0].imshow(image_set[0])
ax[ 1].imshow(image_set[1])
ax[ 2].imshow(image_set[2])
ax[ 3].imshow(image_set[3])
ax[ 4].imshow(image_set[4])


plt.imshow(im1)