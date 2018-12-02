import mxnet as mx
import numpy as np
import Tkinter as tk
import array
from PIL import Image, ImageDraw

def transform (data, label):
	return (mx.nd.floor (data/128)).astype (np.float32), label.astype (np.float32)

print "Getting Training Data..."
trainData = mx.gluon.data.vision.MNIST (root="./mnist", train=True,  transform=transform)
testData  = mx.gluon.data.vision.MNIST (root="./mnist", train=False, transform=transform)

#################
## Classifiers ##
#################
# will be right as many times as 4 appears in test data 9ish%
class NumericalClassifier:
	def __init__(self):
		pass
	def train (self, trainingData, testingData):
		pass
	def classify (self, image):
		return 4 # chosen by fair dice roll. garunteed to be random

########
# Bogo #
########
# le joke. 10% correct
# https://en.wikipedia.org/wiki/Bogosort
class BogoClassifier (NumericalClassifier):
	def classify (self, image):
		probabilities = mx.nd.ones (10) / 10
		random = mx.nd.sample_multinomial (probabilities)
		return random.asnumpy().item(0)

##############
# NaiveBayes #
##############
# trains 1 time. should be right about 84.26% of the time.
class NaiveBayesClassifier (NumericalClassifier):

	def train (self, trainData, testData):
		self.ycount = mx.nd.ones (shape= (10))
		self.xcount = mx.nd.ones (shape= (784, 10))

		for data, label in trainData:
			x = data.reshape ((784,))
			y = int (label)
			self.ycount[y] += 1
			self.xcount[:, y] += x

		for i in range (10):
			self.xcount[:, i] = self.xcount[:, i]/self.ycount[i]

		self.py = self.ycount / mx.nd.sum (self.ycount)

	def classify (self, image):
		logxcount = mx.nd.log (self.xcount)
		logxcountneg = mx.nd.log (1-self.xcount)
		logpy = mx.nd.log (self.py)
		x = image.reshape ((784,))

		# we need to incorporate the prior probability p(y) since p(y|x) is
		# proportional to p(x|y) p(y)
		logpx = logpy.copy ()
		for i in range (10):
			# compute the log probability for a digit
			logpx[i] += mx.nd.dot (logxcount[:, i], x) + mx.nd.dot (logxcountneg[:, i], 1-x)

		logpx -= mx.nd.max (logpx)
		# and compute the softmax using logpx
		px = mx.nd.exp (logpx).asnumpy ()
		px /= np.sum (px)

		return np.argmax (px)

#####################
# AdamRMSClassifier #
#####################
class AdamRMSClassifier (NumericalClassifier):

	def train (self, trainData, testData):
		pass

	def classify (self, image):
		return 4

###########
# OPTIONS #
###########

# classifier = NumericalClassifier ()
# classifier = BogoClassifier ()
classifier = NaiveBayesClassifier ()
# classifier = AdamRMSClassifier ()
useGPU = False

print "options:"
print "\tclassifier: " + classifier.__class__.__name__
print "\tuseGPU    : " + str (useGPU)

if useGPU:
	MXNET_DEVICE = mx.gpu
	BATCH_SIZE = 128
	trainDataOnDevice = mx.nd.array(trainData, ctx=MXNET_DEVICE())
	testDataOnDevice  = mx.nd.array(testData, ctx=MXNET_DEVICE())
else:
	MXNET_DEVICE = mx.cpu
	BATCH_SIZE = 16
	trainDataOnDevice = trainData
	testDataOnDevice = testData



print "Training Neural Network..."
classifier.train(trainDataOnDevice, testDataOnDevice)


print "Testing Correctness..."
correct = 0
for data, label in testDataOnDevice:
	guess = classifier.classify(data);
	if int (guess) == int (label):
		correct += 1
print "Percent Correct: " + str ((float (correct) / float (len (testDataOnDevice)) * 100.0)) + "%"


print "launching UI."
class Application (tk.Frame):
  
	global py

	lastx = 0
	lasty = 0
	classifier = {}

	def mouseDownEventHandler (self, event):
		self.lastx, self.lasty = event.x, event.y

	def mouseUpEventHandler (self, event):
		self.addLine (event)
		data = self.canvasToImage ()
		predictedNumber = self.classifier.classify (data)
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
		reshaped = mx.nd.array (bytes).reshape (28,28,1)
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

	def __init__ (self, master, classifier):
		tk.Frame.__init__ (self, master)
		self.master = master
		self.classifier = classifier
		self.pack ()
		self.createWidgets ()

if __name__ == "__main__":
	root = tk.Tk ()
	Application (root, classifier).pack (fill="both", expand=True)
	root.mainloop ()
