from rembg import remove
import cv2

# rembg==2.0.25
# cv2==4.6.0

input_path = 'output.jpg'
output_path = 'removed_background.jpg'

input = cv2.imread(input_path)
# print(input)
output = remove(input)
# print(output.shape)
# output.save(output_path)
cv2.imwrite(output_path, output)
print('remove!!')