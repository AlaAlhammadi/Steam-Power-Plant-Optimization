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
	1 )LP turbine Out
	2 )Pressure of Superheater In
	3 )Pressure of Superheater Out
	4 )Condenser Pressure in
	5 )Condenser Pressure out
	6 )Condenser Temperature
	7 )Feedwater Pipe Presssure in
	8 )Feedwater Pipe Presssure out
	9 )Pump Pressure in
	10 )Pump Pressure out
	11 )Steam Drum Pressure in
	12 )Steam Drum Pressure out
	13 )Steam Drum temperature

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
 
		self.parameters_number = 173
 
	def scaling_layer(self,inputs):

		outputs = [None] * 13

		outputs[0] = (inputs[0]-0.05172270164)/0.003231480019
		outputs[1] = (inputs[1]-7.777349949)/5.041019917
		outputs[2] = (inputs[2]-7.777239799)/5.041049957
		outputs[3] = (inputs[3]-0.0508074984)/0.003146710107
		outputs[4] = (inputs[4]-0.0508074984)/0.003146710107
		outputs[5] = (inputs[5]-354.8049927)/1.539829969
		outputs[6] = (inputs[6]-7.777359962)/5.041019917
		outputs[7] = (inputs[7]-7.777359962)/5.041019917
		outputs[8] = (inputs[8]-0.0508074984)/0.003146710107
		outputs[9] = (inputs[9]-7.777359962)/5.041019917
		outputs[10] = (inputs[10]-7.777349949)/5.041019917
		outputs[11] = (inputs[11]-7.777349949)/5.041019917
		outputs[12] = (inputs[12]-555.4949951)/42.96229935

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 7

		combinations[0] = 0.432272 +0.338107*inputs[0] +0.00372725*inputs[1] +0.00394156*inputs[2] +0.301952*inputs[3] +0.302081*inputs[4] +0.335724*inputs[5] +0.00336102*inputs[6] +0.00341115*inputs[7] +0.302107*inputs[8] +0.00360442*inputs[9] +0.00336869*inputs[10] +0.00344363*inputs[11] -0.10471*inputs[12] 
		combinations[1] = -0.242002 +0.0930648*inputs[0] +0.234384*inputs[1] +0.234314*inputs[2] +0.0998352*inputs[3] +0.0999097*inputs[4] +0.0910037*inputs[5] +0.234586*inputs[6] +0.234255*inputs[7] +0.100029*inputs[8] +0.234583*inputs[9] +0.234265*inputs[10] +0.234295*inputs[11] +0.187029*inputs[12] 
		combinations[2] = -0.456074 +0.0643697*inputs[0] -0.107018*inputs[1] -0.10674*inputs[2] +0.120051*inputs[3] +0.119797*inputs[4] +0.130496*inputs[5] -0.107017*inputs[6] -0.107213*inputs[7] +0.119919*inputs[8] -0.107047*inputs[9] -0.106952*inputs[10] -0.106954*inputs[11] -0.335383*inputs[12] 
		combinations[3] = 0.400815 +0.249778*inputs[0] +0.123017*inputs[1] +0.122972*inputs[2] +0.223643*inputs[3] +0.223571*inputs[4] +0.226295*inputs[5] +0.122944*inputs[6] +0.123231*inputs[7] +0.223358*inputs[8] +0.123083*inputs[9] +0.12307*inputs[10] +0.122976*inputs[11] +0.410164*inputs[12] 
		combinations[4] = 0.272062 +0.0825702*inputs[0] -0.111439*inputs[1] -0.111529*inputs[2] +0.0443559*inputs[3] +0.0443763*inputs[4] +0.0299963*inputs[5] -0.111638*inputs[6] -0.11158*inputs[7] +0.0444224*inputs[8] -0.111461*inputs[9] -0.111644*inputs[10] -0.111584*inputs[11] -0.00475015*inputs[12] 
		combinations[5] = 0.00279813 -0.00699383*inputs[0] -0.0381094*inputs[1] -0.0380295*inputs[2] -0.0233248*inputs[3] -0.0234489*inputs[4] -0.0317172*inputs[5] -0.0378163*inputs[6] -0.0379391*inputs[7] -0.0235295*inputs[8] -0.0380049*inputs[9] -0.0381498*inputs[10] -0.0377962*inputs[11] +0.0405266*inputs[12] 
		combinations[6] = 0.154916 -0.332072*inputs[0] -0.0276218*inputs[1] -0.0277693*inputs[2] -0.310627*inputs[3] -0.310582*inputs[4] -0.303446*inputs[5] -0.0278059*inputs[6] -0.0278325*inputs[7] -0.31082*inputs[8] -0.0278665*inputs[9] -0.0277184*inputs[10] -0.0279363*inputs[11] -0.244017*inputs[12] 
		
		activations = [None] * 7

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])
		activations[4] = np.tanh(combinations[4])
		activations[5] = np.tanh(combinations[5])
		activations[6] = np.tanh(combinations[6])

		return activations;


	def perceptron_layer_2(self,inputs):

		combinations = [None] * 4

		combinations[0] = 0.00599896 +0.379365*inputs[0] +0.349495*inputs[1] +0.328327*inputs[2] -0.401272*inputs[3] -0.270393*inputs[4] -0.102976*inputs[5] +0.342723*inputs[6] 
		combinations[1] = -0.00240333 -0.234432*inputs[0] -0.227232*inputs[1] -0.215363*inputs[2] +0.267465*inputs[3] +0.164899*inputs[4] +0.06367*inputs[5] -0.231878*inputs[6] 
		combinations[2] = -0.0147465 +0.170698*inputs[0] +0.167704*inputs[1] +0.172567*inputs[2] -0.219841*inputs[3] -0.115693*inputs[4] -0.0465407*inputs[5] +0.19033*inputs[6] 
		combinations[3] = -0.001277 -0.263235*inputs[0] -0.249138*inputs[1] -0.249121*inputs[2] +0.302689*inputs[3] +0.191167*inputs[4] +0.070847*inputs[5] -0.260964*inputs[6] 
		
		activations = [None] * 4

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])

		return activations;


	def perceptron_layer_3(self,inputs):

		combinations = [None] * 3

		combinations[0] = -0.00140396 -0.527776*inputs[0] +0.325388*inputs[1] -0.249079*inputs[2] +0.366452*inputs[3] 
		combinations[1] = 0.0130369 +0.501458*inputs[0] -0.324794*inputs[1] +0.248865*inputs[2] -0.364086*inputs[3] 
		combinations[2] = -0.000376314 +0.517842*inputs[0] -0.31544*inputs[1] +0.249836*inputs[2] -0.360439*inputs[3] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def perceptron_layer_4(self,inputs):

		combinations = [None] * 3

		combinations[0] = 0.0181135 -0.450478*inputs[0] +0.435749*inputs[1] +0.441308*inputs[2] 
		combinations[1] = -0.0172892 +0.450812*inputs[0] -0.43823*inputs[1] -0.443936*inputs[2] 
		combinations[2] = -0.0171985 +0.460332*inputs[0] -0.447555*inputs[1] -0.452574*inputs[2] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def perceptron_layer_5(self,inputs):

		combinations = [None] * 3

		combinations[0] = 0.0560631 +0.466226*inputs[0] -0.468578*inputs[1] -0.478521*inputs[2] 
		combinations[1] = 0.0540019 +0.46196*inputs[0] -0.466843*inputs[1] -0.476723*inputs[2] 
		combinations[2] = 0.0534469 +0.466113*inputs[0] -0.468393*inputs[1] -0.48184*inputs[2] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def perceptron_layer_6(self,inputs):

		combinations = [None] * 1

		combinations[0] = -0.263459 -0.899874*inputs[0] -0.89277*inputs[1] -0.903774*inputs[2] 
		
		activations = [None] * 1

		activations[0] = combinations[0]

		return activations;


	def unscaling_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]*0.08328899741+1.088500023

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

		output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_6)

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

			output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_6)

			output_bounding_layer = self.bounding_layer(output_unscaling_layer)

			output = np.append(output,output_bounding_layer, axis=0)

		return output
