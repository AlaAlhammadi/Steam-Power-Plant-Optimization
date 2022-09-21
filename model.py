'''
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the 'NeuralNetwork' class.	
Example:

	model = NeuralNetwork()	
	sample = [input_1, input_2, input_3, input_4, ...] 	 	
	outputs = model.calculate_output(sample)

	Inputs Names: 	
	1 )Pressure of Superheater In
	2 )Pressure of Superheater Out
	3 )Condenser Pressure in
	4 )Condenser Pressure out
	5 )Condenser Temperature
	6 )Feedwater Pipe Presssure in
	7 )Feedwater Pipe Presssure out
	8 )Pump Pressure in
	9 )Pump Pressure out
	10 )Steam Drum Pressure in
	11 )Steam Drum Pressure out
	12 )Steam Drum temperature

You can predict with a batch of samples using calculate_batch_output method	
IMPORTANT: input batch must be <class 'numpy.ndarray'> type	
Example_1:	
	model = NeuralNetwork()	
	input_batch = np.array([[1, 2], [4, 5]], np.int32)	
	outputs = model.calculate_batch_output(input_batch)
Example_2:	
	input_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})	
	outputs = model.calculate_batch_output(input_batch.values)
'''

import numpy as np

class NeuralNetwork:
 
	def __init__(self):
 
		self.parameters_number = 161
 
	def scaling_layer(self,inputs):

		outputs = [None] * 12

		outputs[0] = (inputs[0]-7.751890182)/5.03674984
		outputs[1] = (inputs[1]-7.751780033)/5.036769867
		outputs[2] = (inputs[2]-0.05080270022)/0.003137930064
		outputs[3] = (inputs[3]-0.05080270022)/0.003137930064
		outputs[4] = (inputs[4]-354.803009)/1.535480022
		outputs[5] = (inputs[5]-7.751900196)/5.03674984
		outputs[6] = (inputs[6]-7.751900196)/5.03674984
		outputs[7] = (inputs[7]-0.05080270022)/0.003137930064
		outputs[8] = (inputs[8]-7.751900196)/5.03674984
		outputs[9] = (inputs[9]-7.751890182)/5.03674984
		outputs[10] = (inputs[10]-7.751890182)/5.03674984
		outputs[11] = (inputs[11]-555.2579956)/42.94380188

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 6

		combinations[0] = -0.207603 -0.0620983*inputs[0] -0.0619792*inputs[1] +0.0734029*inputs[2] +0.0733924*inputs[3] +0.0833378*inputs[4] -0.0621306*inputs[5] -0.0621183*inputs[6] +0.0734483*inputs[7] -0.0620243*inputs[8] -0.0619326*inputs[9] -0.0620486*inputs[10] -0.139767*inputs[11] 
		combinations[1] = 0.187224 +0.0650137*inputs[0] +0.06502*inputs[1] -0.0750219*inputs[2] -0.0749924*inputs[3] -0.0856556*inputs[4] +0.0651697*inputs[5] +0.0650914*inputs[6] -0.0751199*inputs[7] +0.0650157*inputs[8] +0.0651426*inputs[9] +0.0651853*inputs[10] +0.138566*inputs[11] 
		combinations[2] = 0.273301 -0.212767*inputs[0] -0.212783*inputs[1] -0.0948353*inputs[2] -0.0949182*inputs[3] -0.107873*inputs[4] -0.212702*inputs[5] -0.212628*inputs[6] -0.0947702*inputs[7] -0.212702*inputs[8] -0.212548*inputs[9] -0.212749*inputs[10] -0.158649*inputs[11] 
		combinations[3] = -0.0114549 +0.0605191*inputs[0] +0.0602628*inputs[1] -0.242259*inputs[2] -0.242319*inputs[3] -0.266714*inputs[4] +0.0604773*inputs[5] +0.0603477*inputs[6] -0.242453*inputs[7] +0.060433*inputs[8] +0.0603975*inputs[9] +0.060424*inputs[10] +0.216528*inputs[11] 
		combinations[4] = -0.137966 -0.104103*inputs[0] -0.104014*inputs[1] -0.378098*inputs[2] -0.378318*inputs[3] -0.356888*inputs[4] -0.104076*inputs[5] -0.103961*inputs[6] -0.378206*inputs[7] -0.103933*inputs[8] -0.103944*inputs[9] -0.104007*inputs[10] -0.339776*inputs[11] 
		combinations[5] = -1.34163e-05 -0.0662012*inputs[0] -0.0663218*inputs[1] +0.00895827*inputs[2] +0.00884207*inputs[3] +0.00122037*inputs[4] -0.0661669*inputs[5] -0.0662991*inputs[6] +0.0089358*inputs[7] -0.0662493*inputs[8] -0.0662696*inputs[9] -0.0661182*inputs[10] -0.0334322*inputs[11] 
		
		activations = [None] * 6

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])
		activations[4] = np.tanh(combinations[4])
		activations[5] = np.tanh(combinations[5])

		return activations;


	def perceptron_layer_2(self,inputs):

		combinations = [None] * 4

		combinations[0] = 0.0129756 -0.0820154*inputs[0] +0.0767563*inputs[1] -0.215234*inputs[2] -0.0213575*inputs[3] +0.080391*inputs[4] -0.0994525*inputs[5] 
		combinations[1] = -0.0209731 +0.0932515*inputs[0] -0.0905928*inputs[1] +0.272943*inputs[2] +0.0372807*inputs[3] -0.10439*inputs[4] +0.125812*inputs[5] 
		combinations[2] = -0.58349 +0.360742*inputs[0] -0.350162*inputs[1] -0.142568*inputs[2] -0.531343*inputs[3] +0.820666*inputs[4] +0.0648213*inputs[5] 
		combinations[3] = 0.0632587 -0.0908339*inputs[0] +0.0953902*inputs[1] -0.324842*inputs[2] -0.0833073*inputs[3] +0.171756*inputs[4] -0.157822*inputs[5] 
		
		activations = [None] * 4

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])

		return activations;


	def perceptron_layer_3(self,inputs):

		combinations = [None] * 3

		combinations[0] = -0.0412359 -0.130739*inputs[0] +0.158172*inputs[1] -0.403625*inputs[2] -0.185624*inputs[3] 
		combinations[1] = -0.134802 -0.164883*inputs[0] +0.244836*inputs[1] -0.663154*inputs[2] -0.35435*inputs[3] 
		combinations[2] = 0.0964586 +0.174487*inputs[0] -0.224605*inputs[1] +0.567644*inputs[2] +0.28081*inputs[3] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def perceptron_layer_4(self,inputs):

		combinations = [None] * 3

		combinations[0] = -0.0273477 +0.288493*inputs[0] +0.462768*inputs[1] -0.398902*inputs[2] 
		combinations[1] = -0.0191166 +0.274842*inputs[0] +0.434023*inputs[1] -0.367435*inputs[2] 
		combinations[2] = 0.0613438 -0.294165*inputs[0] -0.529843*inputs[1] +0.475074*inputs[2] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def perceptron_layer_5(self,inputs):

		combinations = [None] * 3

		combinations[0] = -0.0333218 +0.409335*inputs[0] +0.374276*inputs[1] -0.457643*inputs[2] 
		combinations[1] = 0.0282889 -0.384887*inputs[0] -0.358611*inputs[1] +0.447339*inputs[2] 
		combinations[2] = 0.0195005 -0.407025*inputs[0] -0.38051*inputs[1] +0.460481*inputs[2] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def perceptron_layer_6(self,inputs):

		combinations = [None] * 3

		combinations[0] = -0.034315 +0.436116*inputs[0] -0.417674*inputs[1] -0.433704*inputs[2] 
		combinations[1] = -0.0340504 +0.435807*inputs[0] -0.418874*inputs[1] -0.433742*inputs[2] 
		combinations[2] = 0.0395351 -0.433064*inputs[0] +0.405732*inputs[1] +0.426859*inputs[2] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def perceptron_layer_7(self,inputs):

		combinations = [None] * 1

		combinations[0] = -0.110788 +0.79443*inputs[0] +0.791077*inputs[1] -0.779738*inputs[2] 
		
		activations = [None] * 1

		activations[0] = combinations[0]

		return activations;


	def unscaling_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]*0.1181939989+1.080929995

		return outputs


	def bounding_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]

		return outputs


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

		output_perceptron_layer_2 = self.perceptron_layer_2(output_perceptron_layer_1)

		output_perceptron_layer_3 = self.perceptron_layer_3(output_perceptron_layer_2)

		output_perceptron_layer_4 = self.perceptron_layer_4(output_perceptron_layer_3)

		output_perceptron_layer_5 = self.perceptron_layer_5(output_perceptron_layer_4)

		output_perceptron_layer_6 = self.perceptron_layer_6(output_perceptron_layer_5)

		output_perceptron_layer_7 = self.perceptron_layer_7(output_perceptron_layer_6)

		output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_7)

		output_bounding_layer = self.bounding_layer(output_unscaling_layer)

		return output_bounding_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

			output_perceptron_layer_2 = self.perceptron_layer_2(output_perceptron_layer_1)

			output_perceptron_layer_3 = self.perceptron_layer_3(output_perceptron_layer_2)

			output_perceptron_layer_4 = self.perceptron_layer_4(output_perceptron_layer_3)

			output_perceptron_layer_5 = self.perceptron_layer_5(output_perceptron_layer_4)

			output_perceptron_layer_6 = self.perceptron_layer_6(output_perceptron_layer_5)

			output_perceptron_layer_7 = self.perceptron_layer_7(output_perceptron_layer_6)

			output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_7)

			output_bounding_layer = self.bounding_layer(output_unscaling_layer)

			output = np.append(output,output_bounding_layer, axis=0)

		return output
