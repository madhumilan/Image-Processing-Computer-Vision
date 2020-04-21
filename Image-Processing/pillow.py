from PIL import Image

img = Image.open("images/test.jpg")

# Resizing images --------------------------
# img = Image.open("images/test.jpg")
# print(type(img))
# print(img.size)

# resized = img.resize((350, 300))
# resized.save("images/test_resized.jpg")

# img.thumbnail((350,400))
# print(img.size)
# img.save("images/test_thumbnail.jpg")

# Cropping images ---------------------------
# img = Image.open("images/test.jpg")
# cropped = img.crop((0,200, 400, 400))
# cropped.save("images/cropped_test.jpg")
# print(cropped.size)

# Cutting and pasting images
# # rect = Image.open("images/cropped_test.jpg")
# # im2 = Image.open("images/test_resized.jpg")
# cropped = img.crop((200, 0, 400, 3000))
# img.paste(cropped, None)
# # img.save("images/img_paste.jpg")
# img.show()

# Extracting R G B color values
# r, g, b = img.split()			# Returns the individual color channels (Not human readable)
# orig = Image.merge("RGB", (r, g, b))
# orig.show()

# Rotate and flip the images
rotate90 = img.transpose(Image.ROTATE_90)			# Rotates counter-clockwise
rotate90.show()
flip_LR = img.transpose(Image.FLIP_LEFT_RIGHT)		# Mirror image
flip_LR.show()


