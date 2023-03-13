from deepface import DeepFace

# Path to the two images to compare
img1_path = 'C://Users//henri//Downloads//IMG_20230313_165759.jpg'
img2_path = 'C://Users//henri//Downloads//IMG_20230313_165756.jpg'

# Run facial recognition on both images
img1 = DeepFace.extract_faces(img1_path, detector_backend='opencv')
img2 = DeepFace.extract_faces(img2_path, detector_backend='opencv')
# Extract the first face in the list of detected faces (assuming only one face is present in each image)


# Compare the faces in the images using the Facenet model
result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name='Facenet', distance_metric='euclidean')
# Print the result
if result['verified']:
    print("The same person is in both images.")
else:
    print("The two images show different people.")
