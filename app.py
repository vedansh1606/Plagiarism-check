import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Collect all the student note files (ending with .txt)
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]

# Step 2: Read each file into a list of text content
student_notes = []
for _file in student_files:
    with open(_file, encoding='utf-8') as file:
        student_notes.append(file.read())

# Step 3: Convert the student notes to vectors using TF-IDF
# This function will turn the list of student notes into numerical vectors
def vectorize(texts):
    return TfidfVectorizer().fit_transform(texts).toarray()

# Step 4: Calculate the cosine similarity between two documents (vectors)
def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])[0][1]

# Step 5: Vectorize the student notes
vectors = vectorize(student_notes)

# Combine the filenames and their corresponding vectors into one list for easy comparison
s_vectors = list(zip(student_files, vectors))

# A set to store the plagiarism results (to avoid duplicate pairs)
plagiarism_results = set()

# Step 6: Check for plagiarism by comparing each pair of student notes
def check_plagiarism():
    global s_vectors
    # Loop through each student's note
    for student_a, text_vector_a in s_vectors:
        # Make a copy of the list, excluding the current student's note
        remaining_vectors = [item for item in s_vectors if item[0] != student_a]
        
        # Compare the current student's note with every other student's note
        for student_b, text_vector_b in remaining_vectors:
            # Calculate the cosine similarity between the two student notes
            sim_score = similarity(text_vector_a, text_vector_b)
            
            # Store the result as a tuple: (student_a, student_b, similarity_score)
            student_pair = sorted((student_a, student_b))  # Always store in a sorted order to avoid duplicates
            plagiarism_results.add((student_pair[0], student_pair[1], sim_score))
    
    return plagiarism_results

# Step 7: Run the plagiarism check and print results
for result in check_plagiarism():
    print(result)
