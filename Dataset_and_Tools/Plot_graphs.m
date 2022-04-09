im1 = imread('./dog.jpg');
im2 = imread('./dog.jpg_SNP_0.1.JPEG');
im3 = imread('./dog.jpg_SNP_0.2.JPEG');
im4 = imread('./dog.jpg_SNP_0.3.JPEG');
im5 = imread('./dog.jpg_SNP_0.4.JPEG');

subplot(1,5,1), imshow(im1);
title('Original, 88.46%');
subplot(1,5,2), imshow(im2);
title('lvl 1, 95.17%');
subplot(1,5,3), imshow(im3);
title('lvl 2, 89.27%');
subplot(1,5,4), imshow(im4);
title('lvl 3, 69.59%');
subplot(1,5,5), imshow(im5);
title('lvl 4, 46.41%');


