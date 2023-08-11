import dlib
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
import random

dataset = []
root = '/Users/shijiewang/Desktop/face/face_identification'
# 加载 dlib 的面部检测器
detector = dlib.get_frontal_face_detector()

# 加载 facenet 的模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(img_path):
    # 加载图像
    img = cv2.imread(img_path)
    if img is None or img.size == 0:
        print(f"Error loading the image: {img_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整图像大小以适应模型的输入
    face_img = cv2.resize(img_rgb, (160, 160))

    # 使用 FaceNet 获取嵌入向量
    embedding = resnet(torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device))
    return embedding.detach().cpu().numpy()

def compare_faces(embedding1, embedding2):
    distance = torch.nn.functional.pairwise_distance(torch.tensor(embedding1), torch.tensor(embedding2))
    return distance.item()

def gen_randint(low, high, discard):
	result_list = list(range(low, high))
	result_list.remove(discard)
	np.random.shuffle(result_list)
	return result_list.pop()

for i in range(500):
    for j in range(5):
        rnum1 = random.randint(0,499)
        rnum2 = gen_randint(0,5,j)
        data_same =  (root+'/%.3d/%d.jpg'%(i,j), root+'/%.3d/%d.jpg'%(i,rnum2), True)
        data_diff = (root+'/%.3d/%d.jpg'%(i,j), root+'/%.3d/%d.jpg'%(rnum1,j), False)
        dataset.append(data_same)
        dataset.append(data_diff)

def evaluate_threshold(dataset, threshold):
    correct_predictions = 0
    total_predictions = len(dataset)
    
    for img_path1, img_path2, are_same_person in dataset:
        embedding1 = get_face_embedding(img_path1)
        embedding2 = get_face_embedding(img_path2)
        
        if embedding1 is not None and embedding2 is not None:
            distance = compare_faces(embedding1, embedding2)
            
            if distance < threshold and are_same_person:
                correct_predictions += 1
            elif distance >= threshold and not are_same_person:
                correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

# 设定一系列要测试的阈值
thresholds = np.linspace(0.2, 1.5, 30)  # 例如从 0.2 到 1.5 的 30 个值

best_threshold = None
best_accuracy = 0

for threshold in thresholds:
    accuracy = evaluate_threshold(dataset, threshold)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold is {best_threshold} with accuracy of {best_accuracy}")
