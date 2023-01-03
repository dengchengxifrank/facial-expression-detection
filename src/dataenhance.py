import PIL.Image as Image
import os
from torchvision import transforms as transforms
import torchvision.transforms.functional as TF

# 此处是基础函数，就是读取图像，是使用PIL格式的
# pytorch提供的torchvision主要使用PIL的Image类进行处理，所以它数据增强函数大多数都是以PIL作为输入，并且以PIL作为输出。
# 选取PIL库是因为这是python的原生库，兼容性比较强
def read_PIL(image_path):
    """ read image in specific path
    and return PIL.Image instance"""
    image = Image.open(image_path)
    return image

# 中心裁剪
def center_crop(image):
    CenterCrop = transforms.CenterCrop(size=(300, 300))
    cropped_image= CenterCrop(image)
    return cropped_image

# 随机裁剪
def random_crop(image):
    RandomCrop = transforms.RandomCrop(size=(200, 200))
    random_image = RandomCrop(image)
    return random_image

# 定心定义尺寸
def resize(image):
    Resize = transforms.Resize(size=(100, 150))  # 指定长宽比
    resized_image = Resize(image)
    return resized_image

# 水平翻转
def horizontal_flip(image):
    HF = transforms.RandomHorizontalFlip()
    hf_image = HF(image)
    return hf_image

# 垂直翻转
def vertical_flip(image):
    VF = transforms.RandomVerticalFlip()
    vf_image = VF(image)
    return vf_image

# 随机角度旋转
def random_rotation(image):
    RR = transforms.RandomRotation(degrees=(10, 80))
    rr_image = RR(image)
    return rr_image

# 色度、亮度、饱和度、对比度的变化
def BCSH_transform(image):
    im = transforms.ColorJitter(brightness=1)(image)
    im = transforms.ColorJitter(contrast=1)(im)
    im = transforms.ColorJitter(saturation=0.6)(im)
    im = transforms.ColorJitter(hue=0.4)(im)
    return im

# 随机灰度化
def random_gray(image):
    gray_image = transforms.RandomGrayscale(p=0.5)(image)    # 以0.5的概率进行灰度化
    return gray_image

# Padding (将原始图padding成正方形)
def pad(image):
    pad_image = transforms.Pad((0, (im.size[0]-im.size[1])//2))(im)
    return pad_image


def erase_image(image, position, size):
    """
    按照指定的位置和长宽擦除
    :param image_path: 输入图像
    :param position: 擦除的左上角坐标
    :param size: 擦除的长宽值
    :return: 返回擦除后的图像
    """
    img = TF.to_tensor(image)
    erased_image = TF.to_pil_image(TF.erase(img=img,
                            i=position[0],
                            j=position[1],
                            h=size[0],
                            w=size[1],
                            v=1))
    return erased_image

def gamma_transform(image, gamma_value):
    """
    进行伽马变换
    :param image_path: 输入图片路径
    :param gamma_value: 伽马值
    :return: 伽马变换后的图像
    """
    gamma_image = TF.adjust_gamma(img=image, gamma=gamma_value)
    return gamma_image



#im = read_PIL('D:/new_learn/images/src1.jpeg')
im = read_PIL('D:/new_learn/images/src.jpg')
print(im.size)  # 得到尺寸

outDir = 'D:/new_learn/images/'
os.makedirs(outDir, exist_ok=True)

center_cropped_image = center_crop(im)  # 中心裁剪
center_cropped_image.save(os.path.join(outDir, 'center_cropped_image.jpg'))

random_cropped_image = random_crop(im)  # 随机裁剪
random_cropped_image.save(os.path.join(outDir, 'random_cropped_image.jpg'))

resized_image = resize(im)  # 重新resize
resized_image.save(os.path.join(outDir, 'resized_image.jpg'))

hf_image = horizontal_flip(im)  # 水平翻转
hf_image.save(os.path.join(outDir, 'hf_image.jpg'))

vf_image = vertical_flip(im)  # 垂直翻转
vf_image.save(os.path.join(outDir, 'vf_image.jpg'))

rr_image = random_rotation(im)  # 随机翻转
rr_image.save(os.path.join(outDir, 'rr_image.jpg'))

bcsh_image = BCSH_transform(im)  # 色度、亮度、饱和度、对比度的变化
bcsh_image.save(os.path.join(outDir, 'bcsh_image.jpg'))

random_gray_image = random_gray(im)  # 随机灰度化
random_gray_image.save(os.path.join(outDir, 'random_gray_image.jpg'))

pad_image = pad(im)  # 将图形加上padding，行成正方形
pad_image.save(os.path.join(outDir, 'pad_image.jpg'))

erased_image = erase_image(im, (100, 100), (50, 200))  # 指定区域擦除
erased_image.save(os.path.join(outDir, 'erased_image.jpg'))

gamma_image = gamma_transform(im, 0.1)  # γ变换
gamma_image.save(os.path.join(outDir, 'gamma_image.jpg'))

# 使用Compose函数生成一个PiPeLine, 经过这样处理后，我们就可以直接使用data_transform来进行图像的变换
data_transform={'train':transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(size=(100, 150)),
                    transforms.CenterCrop(size=(100, 150)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])}

