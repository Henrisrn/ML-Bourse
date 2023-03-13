import os
import shutil
from deepface import DeepFace 

data_dir = "datasetimage"

for directory in os.listdir(data_dir):
    first_file = os.listdir(os.path.join(data_dir,directory))[1]
    shutil.copyfile(os.path.join(data_dir,directory,first_file),os.path.join("Samples",f"{directory}.jpg"))
smallest_distance = None
for file in os.listdir("Samples"):
    if file.endswith(".jpg"):
        result = DeepFace.verify("Person1.jpg",f"Samples/{file}", model_name='Facenet', distance_metric='euclidean')
        print(result)
        if result["verified"]:
            print("This person looks exactly like : ",file.split(".")[0])
            break
        if smallest_distance is None:
            smallest_distance = (file.split(".")[0], result['distance'])
        else:
            smallest_distance = (file.split(".")[0], result["distance"]) if result["distance"] < smallest_distance[1] else smallest_distance
else:
    print(f"No match found, close match is : {smallest_distance}")

#result = DeepFace.verify("Person1.jpg",f"Samples/Angelina Jolie.jpg", model_name='Facenet', distance_metric='euclidean')