import dlib
import cv2
import torch
from facenet_pytorch import InceptionResnetV1

# 加载 dlib 的面部检测器
detector = dlib.get_frontal_face_detector()

# 加载 facenet 的模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(img_path):
    # 加载图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading the image: {img_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 使用 dlib 进行面部检测
    faces = detector(img_rgb)
    if len(faces) == 0:
        print(f"No face detected in {img_path}")
        return None

    # 选择第一个检测到的面部（如果有多个面部，可以进行其他处理）
    face = faces[0]
    face_img = img_rgb[face.top():face.bottom(), face.left():face.right()]
    
    # 调整图像大小以适应模型的输入
    face_img = cv2.resize(face_img, (160, 160))

    # 使用 FaceNet 获取嵌入向量
    embedding = resnet(torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device))
    return embedding.detach().cpu().numpy()

def compare_faces(embedding1, embedding2):
    distance = torch.nn.functional.pairwise_distance(torch.tensor(embedding1), torch.tensor(embedding2))
    return distance.item()

# 提取图像特征
image1_path = "/Users/shijiewang/Desktop/face/face_identification/000/0.jpg"
image2_path = "/Users/shijiewang/Desktop/face/face_identification/137/0.jpg"

embedding1 = get_face_embedding(image1_path)
embedding2 = get_face_embedding(image2_path)

# 比较两张图片中的人脸
if embedding1 is not None and embedding2 is not None:
    distance = compare_faces(embedding1, embedding2)
    threshold = 0.5
    if distance < threshold:
        print("The two faces are of the same person!")
    else:
        print("The two faces are of different persons!")
