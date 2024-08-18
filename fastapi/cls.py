# Use a pipeline as a high-level helper
# From: https://huggingface.co/dima806/facial_emotions_image_detection
# Model: Facial emotions image detection ViT 
# Backbone: ViT

# STEP1: import modules
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch


# STEP2: Load classifier
pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")


# STEP3: Load input data
# from capture.capture import capture_image
# test_frame = capture_image('test.png') # capture
image = Image.open('../images/test.jpg') # sample image
inputs = processor(images=image, return_tensors="pt") # image: PIL로 읽은 이미지


# STEP4: Inference
with torch.no_grad():
    outputs = model(**inputs)


# STEP5: results and post-processing ############################################################

# logits 가져와서 최대값 인덱스 출력
logits = outputs.logits
predicted_class_id = logits.argmax().item() # argmax index
predicted_class_name = model.config.id2label[predicted_class_id] # classname(index to label)
# print(model) # 모델 레이어 
# print(model.config) # 모델에 저장된 configs
# print(model.config.id2label) # configs중 label 정보 (dict)
# print(predicted_class_name) # 가장 높은 확률값에 해당하는 label

# 확률값으로 변환
probabilities = torch.nn.functional.softmax(logits, dim=-1)

################################################################

# 클래스별 예측값, 클래스 이름, 확률을 리스트로 저장, 전체 결과 출력
results = []
for i, (logit, probability) in enumerate(zip(logits[0], probabilities[0])):
    class_name = model.config.id2label[i]
    rounded_probability = round(probability.item(), 2)
    results.append((class_name, logit.item(), rounded_probability))
    print(f"Class: {class_name},\t Prob: {rounded_probability}")

# 확률에 따라 내림차순으로 정렬
results.sort(key=lambda x: x[2], reverse=True)

print('\n===========================================')
# 상위 3개의 결과를 순위와 함께 출력
for rank, (class_name, logit, probability) in enumerate(results[:3], start=1):
    print(f"Rank: {rank}, Class: {class_name}\t Probability: {probability}")
    
