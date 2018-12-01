import mxnet as mx
import matplotlib as mpl
import numpy as np
import Tkinter as tk
import array
from mxnet import nd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

def transform (data, label):
	return (nd.floor (data/128)).astype (np.float32), label.astype (np.float32)

print "Getting Training Data..."
mnist_train = mx.gluon.data.vision.MNIST (root="./mnist", train=True, transform=transform)
mnist_test  = mx.gluon.data.vision.MNIST (root="./mnist", train=False, transform=transform)

ycount = nd.ones (shape= (10))
xcount = nd.ones (shape= (784, 10))

print "Training Neural Network..."
for data, label in mnist_train:
	x = data.reshape ((784,))
	y = int (label)
	ycount[y] += 1
	xcount[:, y] += x

print "Computing Training Distributions..."
for i in range (10):
	xcount[:, i] = xcount[:, i]/ycount[i]

py = ycount / nd.sum (ycount)

#################
## Classifiers ##
#################

class NumericalClassifier:
	def __init__(self, py):
		self.py = py

	#default behavior is random
	def classifyNumber (self, image):
		probabilities = nd.ones (10) / 10
		return nd.sample_multinomial (probabilities)[0]

class NaiveBayesClassifier (NumericalClassifier):
	def classifyNumber (self, image):
		logxcount = nd.log (xcount)
		logxcountneg = nd.log (1-xcount)
		logpy = nd.log (self.py)
		x = image.reshape ((784,))

		# we need to incorporate the prior probability p(y) since p(y|x) is
		# proportional to p(x|y) p(y)
		logpx = logpy.copy ()
		for i in range (10):
			# compute the log probability for a digit
			logpx[i] += nd.dot (logxcount[:, i], x) + nd.dot (logxcountneg[:, i], 1-x)

		logpx -= nd.max (logpx)
		# and compute the softmax using logpx
		px = nd.exp (logpx).asnumpy ()
		px /= np.sum (px)

		return np.argmax (px)

print "Selecting NaiveBayesClassifier..."
classifier = NaiveBayesClassifier (py)

#########################
## Testing Correctness ##
#########################

print "Classifying whole test set..."

correct = 0
for data, label in mnist_test:

	guess = classifier.classifyNumber(data);
	if int (guess) == int (label):
		correct += 1

print "Test Data Correctly Guessed: " + str ((float (correct) / float (len (mnist_test)) * 100.0))

####################
## User Interface ##
####################

print "launching UI to let you try it out."

class Application (tk.Frame):
  
	global py

	lastx = 0
	lasty = 0
	numericalClassifier = {}

	def mouseDownEventHandler (self, event):
		self.lastx, self.lasty = event.x, event.y

	def mouseUpEventHandler (self, event):
		self.addLine (event)
		data = self.canvasToImage ()
		predictedNumber = self.numericalClassifier.classifyNumber (data)
		self.predictedNumber.set (str (predictedNumber))

	def addLine (self, event):
		self.canvas.create_line ((self.lastx, self.lasty, event.x, event.y), fill="white", width=25)
		self.draw.line ([self.lastx, self.lasty, event.x, event.y], fill=255, width=25)
		self.lastx, self.lasty = event.x, event.y

	def clearCanvas (self):
		self.canvas.delete ("all")

		self.image = Image.new (mode="L", size=(400, 400), color=0)
		self.draw  = ImageDraw.Draw (self.image)

		self.predictedNumber.set ("Blank")

	def canvasToImage (self):
		smaller = self.image.resize ((28, 28), resample=Image.NEAREST)
		bytes = array.array("B", bytearray (smaller.tobytes ()))
		reshaped = nd.array (bytes).reshape (28,28,1)
		normalized = reshaped / 255.0
		return normalized

	def createWidgets (self):
		self.canvas = tk.Canvas (self, width=400, height=400, background="black")
		self.canvas.pack (side="top", fill="both", expand=True)
		self.canvas.bind ("<Button-1>", self.mouseDownEventHandler)
		self.canvas.bind ("<ButtonRelease-1>", self.mouseUpEventHandler)
		self.canvas.bind ("<B1-Motion>", self.addLine)

		self.image = Image.new (mode="L", size=(400, 400), color=0)
		self.draw  = ImageDraw.Draw (self.image)

		self.clear = tk.Button (self, text="Clear", fg="white", command=self.clearCanvas)
		self.clear.pack (side="bottom", fill="both")

		self.predictedNumber = tk.StringVar ()
		self.predictedNumber.set ("Blank")
		self.output = tk.Label (self, fg="white", textvariable=self.predictedNumber)
		self.output.pack (side="bottom", fill="both")

	def __init__ (self, master, numericalClassifier):
		tk.Frame.__init__ (self, master)
		self.master = master
		self.numericalClassifier = numericalClassifier
		self.pack ()
		self.createWidgets ()

if __name__ == "__main__":
	root = tk.Tk ()
	Application (root, classifier).pack (fill="both", expand=True)
	root.mainloop ()
	
#################
## Data Parser ##
#################
# def nextInt(file):
# 	return int(file.read(4).encode('hex'), 16)

# def assertInt(file, expected):
# 	byte = file.tell()
# 	actual = nextInt(file)
# 	if not actual == expected:
# 		raise Exception(file.name + ":" + str(byte) + " - " + str(actual) + " found " + str(expected) + " expected")

# def parseLabels(fileName):
# 	with open(fileName, "rb") as file:
# 		assertInt(file, 2049)
# 		quantity = nextInt(file)
# 		labels = bytearray()
# 		labels.extend(file.read(quantity))
# 		return labels