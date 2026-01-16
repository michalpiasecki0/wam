iou is calculated as follows:
i took wavelets = ['haar', 'db4', 'sym4', 'sym8']
i took all images
for single image i calculate mean iou by calulating mean iou across all wavelet pairs -> let's call it `MEAN_IOU_IMG`
then i calculate mean across all imgs, `MEAN(sum(MEAN_IOU_IMG))`