# forgery image
file1 = open('./ADE20K/test2.txt')####################################
image1 = file1.read().split('\n')
image1.pop()
f = open('./test2.txt', 'w', encoding='UTF-8')
for i in image1:
    f.write('ADE20K/forgery_image/' + i + '\n')

file2 = open('./Microsoft_COCO/test2.txt')
image2 = file2.read().split('\n')
image2.pop()
for i in image2:
    f.write('Microsoft_COCO/forgery_image/' + i + '\n')

f.close()


# mask
# file1 = open('./ADE20K/train.txt')
# image1 = file1.read().split('\n')
# image1.pop()
# f = open('./mask.txt', 'w', encoding='UTF-8')
# for i in image1:
#     f.write('ADE20K/forgery_mask/' + i + '\n')
#
# file2 = open('./Microsoft_COCO/train.txt')
# image2 = file2.read().split('\n')
# image2.pop()
# for i in image2:
#     f.write('Microsoft_COCO/forgery_mask/' + i + '\n')
#
# f.close()
