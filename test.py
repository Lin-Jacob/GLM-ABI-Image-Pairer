from PIL import Image

# Open an image file
abiImg = Image.open('images/G16_ABI_B03_s20233172000203_e20233172009511_x00000y-02400.nc.png')
# Get the size of the image
abiWidth, abiHeight = abiImg.size

glmImg = Image.open('images/G16_GLM_2023_11_13_200402_x-00100y00000.nc.png')
glmWidth, glmHeight = glmImg.size

print(f"ABI image width: {abiWidth} pixels, height: {abiHeight} pixels.")
print(f"GLM image width: {glmWidth} pixels, height: {glmHeight} pixels.")
print(f"ABI/GLM width: {abiWidth/glmWidth}, ABI/GLM height: {abiHeight/glmHeight}")
