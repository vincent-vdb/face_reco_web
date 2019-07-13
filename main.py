import os
from os import listdir

from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from werkzeug.utils import secure_filename

import pickle
import glob


import matplotlib.pyplot as plt
import numpy as np
import face_recognition
import cv2



app = Flask(__name__)


# instantiate an object that will upload images, with name 'photos'
photos = UploadSet('photos', IMAGES)

# tell the app where to store the images
# print('hello')

# path_to_save = os.join('static/img/'+person)
# os.makedirs(path_to_save)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img/'

configure_uploads(app, photos)

@app.route('/')
def home():
    return render_template('home.html')

# allows to upload an image for the <person> to 
# be able to perform future face recognition on this person
@app.route('/upload/<person>', methods=['GET', 'POST'])
def upload(person):	
	person = person.lower()

	if request.method == 'POST' and 'photo' in request.files:

		path_to_save = os.path.join('static/img/'+person)

		if not os.path.isdir(path_to_save):
			os.makedirs(path_to_save)
		app.config['UPLOADED_PHOTOS_DEST'] = path_to_save
		configure_uploads(app, photos)

		#filename = photos.save(request.files['photo'])
		for f in request.files.getlist('photo'):
			filename = secure_filename(f.filename)

			f.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
			print(filename)		
			encode_image(path_to_save+'/'+filename)

		return "upload and encoding complete"
	return render_template('upload.html', person=person)

# compute the face recognition on the image given in the form of
# face_recon.html, and display the result in output.html
@app.route('/face_recon', methods=['GET', 'POST'])
def face_recon():
	if request.method == 'POST' and 'photo' in request.files:

		# path to save the image to recon
		path_to_save = os.path.join('static/to_recon')
		app.config['UPLOADED_PHOTOS_DEST'] = path_to_save
		configure_uploads(app, photos)

		# save the image
		filename = photos.save(request.files['photo'])

		# compute the recon
		output = compute_recon(path_to_save+'/'+filename)

		# display the image with recon
		return render_template('output.html', output_image=output)

	return render_template('face_recon.html')


#@app.route('/camera_recon')

# encode an image located in path_to_image into face encodings
# and returns the path to the encodings
def encode_image(path_to_image):
	img = plt.imread(path_to_image)
	encodings = face_recognition.face_encodings(img)
	encodings_path = path_to_image.split('.')[0]+'.pkl'
	pickle.dump(encodings, open(encodings_path, 'wb'))
	return encodings_path

# return all the encodings of the previously uploaded images
# into a dictionary (person name as key, list of encodings as values)
def get_encodings():

	output_dict = {}
	for folder in os.listdir('static/img'):
		tmp_encodings = []
		for img in glob.glob('static/img/'+folder+'/*.pkl'):
			tmp_encodings.append(pickle.load(open(img, 'rb')))
		output_dict[folder] = tmp_encodings

	return output_dict

# compute the recognition on the image located at image_path
# then saves the result and returns the path to this result image
def compute_recon(image_path):
	img = plt.imread(image_path)

	# Encode the faces
	img_encodings = face_recognition.face_encodings(img)

	encodings = get_encodings()

	person = []

	# Loop over each encoded face
	for img_encoding in img_encodings:
	    
	    min_dist = 0.6
	    min_key = "unknown"
	    for key in encodings:
	        for encoding in encodings[key]:
	            dist = np.linalg.norm(encoding-img_encoding)
	            if np.linalg.norm(encoding-img_encoding)<min_dist:
	                min_dist = np.linalg.norm(encoding-img_encoding)
	                min_key = key
	    person.append(min_key)


    # Get the faces boxes for viz
	boxes = face_recognition.face_locations(img)
	    

	for (top, right, bottom, left), member in zip(boxes, person):
	    
	    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
	    cv2.putText(img, member, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

	plt.figure(figsize=(16,10)) 
	plt.axis('off')
	plt.imshow(img)

	output_path = image_path.split('/')[-1].split('.')[0]
	output_path = 'static/'+output_path+'.jpg'

	plt.savefig(output_path)

	return output_path

if __name__ == '__main__':
    app.run(debug=True)


