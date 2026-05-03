import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Dataset
# -------------------------------
def load_dataset(dataset_path='dataset'):
    images = []
    labels = []

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            images.append(img)
            labels.append(person)

    return images, labels


# -------------------------------
# 2. Preprocess Images
# -------------------------------
def preprocess_images(images):
    processed = []

    for img in images:
        resized = cv2.resize(img, (100, 100))
        flattened = resized.flatten()
        processed.append(flattened)

    A = np.array(processed).T
    return A


# -------------------------------
# 3. Compute Mean Face
# -------------------------------
def compute_mean(A):
    return np.mean(A, axis=1, keepdims=True)


# -------------------------------
# 4. Normalize Data
# -------------------------------
def normalize_data(A, mean_face):
    return A - mean_face


# -------------------------------
# 5. Compute Eigenfaces
# -------------------------------
def compute_eigenfaces(A_normalized, k):
    cov_matrix = np.dot(A_normalized.T, A_normalized)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx]

    eigenfaces = np.dot(A_normalized, eigenvectors)
    eigenfaces = eigenfaces[:, :k]

    for i in range(k):
        eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])

    return eigenfaces


# -------------------------------
# 6. Project Faces
# -------------------------------
def project_faces(A_normalized, eigenfaces):
    return np.dot(eigenfaces.T, A_normalized)


# -------------------------------
# 7. Recognize Face
# -------------------------------
def recognize_face(test_image, mean_face, eigenfaces, projected_data):
    test_image = test_image.reshape(-1, 1)

    test_normalized = test_image - mean_face
    test_projection = np.dot(eigenfaces.T, test_normalized)

    distances = np.linalg.norm(projected_data - test_projection, axis=0)

    return np.argmin(distances), np.min(distances)


# -------------------------------
# 8. Main Function
# -------------------------------
if __name__ == "__main__":

    print("Loading dataset...")
    images, labels = load_dataset()
    print("Number of images:", len(images))

    print("Preprocessing...")
    A = preprocess_images(images)
    print("Matrix shape:", A.shape)

    print("Computing mean face...")
    mean_face = compute_mean(A)

    print("Normalizing data...")
    A_normalized = normalize_data(A, mean_face)

    print("Computing eigenfaces...")
    k = 10
    eigenfaces = compute_eigenfaces(A_normalized, k)

    print("Projecting dataset...")
    projected_data = project_faces(A_normalized, eigenfaces)

    # ---------------------------
    # Test Image (CHANGE NAME IF NEEDED)
    # ---------------------------
    print("Testing...")

    test_image_path = "test/40_4.jpg"   # 🔥 change this if needed
    test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    if test_img is None:
        print("Error: Test image not found")
        exit()

    test_img = cv2.resize(test_img, (100, 100))
    test_vector = test_img.flatten()

    index, distance = recognize_face(test_vector, mean_face, eigenfaces, projected_data)

    print("\n🔍 RESULT:")
    print("Recognized Person:", labels[index])
    print("Distance:", distance)

    # ---------------------------
    # Threshold Decision
    # ---------------------------
    threshold = 4000

    if distance < threshold:
        print("Face Recognized ✅")
    else:
        print("Unknown Face ❌")

    # ---------------------------
    # Show Images (VERY IMPORTANT)
    # ---------------------------
    matched_image = images[index]

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.title("Test Image")
    plt.imshow(test_img, cmap='gray')

    plt.subplot(1,2,2)
    plt.title("Matched Image")
    plt.imshow(matched_image, cmap='gray')

    plt.show()
