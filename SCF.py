'''============================================================================================================'''
'''        This is an implementation of the Keras layer described in https://arxiv.org/abs/2003.00063          '''
'''                    Make sure to cite the paper if you use this code for your research                      '''
'''                                                                                                            '''
'''   The layer performs fusion of uni-modal embeddings (e.g. audio/video) into a multi-modal representation   '''
'''   The layer's concept is theoretically scalable to the fusion of N different modalities, free free to try  '''
'''   and implement this functionality. If you do let me know, I'd be happy to hear about it :)                '''
'''                                                                                                            '''
'''   by Gustavo Assunção (2020)                                                                               '''
'''   Provided under a GPL-3.0 License                                                                         '''
'''============================================================================================================'''


from keras.models import Model
from keras.layers import Input, Dense, Flatten, Layer
from keras_vggface.vggface import VGGFace
from keras import backend as K
from keras import activations as A

import numpy as np
import math as mt
import keras

K.clear_session()


class SCF(Layer):

	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		self.N = output_dim
		self.M = output_dim

		#Visual
		self.L_v_ex = 1.6
		self.sigma_v_ex = 3.5
		self.L_v_in = 1.23
		self.sigma_v_in = 6.3
		self.Fv = 1
		self.tau_v = 3
		self.upsilon_v = 3
		self.p_v = 0.3
		self.k_v = 7

		#Auditory
		self.L_a_ex = 1
		self.sigma_a_ex = 5.3
		self.L_a_in = 0.8
		self.sigma_a_in = 11.8 
		self.Fa = 1
		self.tau_a = 3
		self.upsilon_a = 3
		self.p_a = 0.3
		self.k_a = 3

		#SC
		self.L_m_ex = 3.8
		self.sigma_m_ex = 3.5
		self.L_m_in = 3.3
		self.sigma_m_in = 6.2
		self.tau_m = 3
		self.upsilon_m = 3
		self.p_m = 0.3

		super(SCF, self).__init__(**kwargs)

	#The input shape is the shape of the stimulus (in our case these are vectors)
	def build(self, input_shape):

		#This represents R_s_ij
		self.receptive_fields_v = self.add_weight(name='kernel_1', shape=(self.N, self.M, input_shape[0][1]), initializer='lecun_normal', trainable=True)
		self.receptive_fields_a = self.add_weight(name='kernel_2', shape=(self.N, self.M, input_shape[1][1]), initializer='lecun_normal', trainable=True)
		
		#This represents z_s_ij
		self.neuron_activity_v = self.add_weight(name='kernel_3', shape=(self.N, self.M), initializer='lecun_normal', trainable=False)
		self.neuron_activity_a = self.add_weight(name='kernel_4', shape=(self.N, self.M), initializer='lecun_normal', trainable=False)
		self.neuron_activity_m = self.add_weight(name='kernel_5', shape=(self.N, self.M), initializer='lecun_normal', trainable=False)

		super(SCF, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):

		'''
		Visual and Auditory Components
		'''
		video = K.repeat_elements(K.expand_dims(x[0], axis=1), rep=self.N, axis=1)
		video = K.repeat_elements(K.expand_dims(video, axis=1), rep=self.M, axis=1)
		audio = K.repeat_elements(K.expand_dims(x[1], axis=1), rep=self.N, axis=1)
		audio = K.repeat_elements(K.expand_dims(audio, axis=1), rep=self.M, axis=1)

		#This represents r_s_ij
		self.external_input_v = K.sum(video * self.receptive_fields_v, axis=-1)
		self.external_input_a = K.sum(audio * self.receptive_fields_a, axis=-1)

		#This represents L_s_ij_hk as the strength of the synaptic connection from the pre-synaptic neuron hk to the post-synaptic neuron ij
		syn_sen_v = np.empty((self.N, self.M, self.N, self.M))
		syn_sen_a = np.empty((self.N, self.M, self.N, self.M))
		syn_sen_m = np.empty((self.N, self.M, self.N, self.M))
		for i in range(self.N):
			for j in range(self.M):
				for h in range(self.N):
					for k in range(self.M):
						[dx, dy] = self._distance_helper(i,j,h,k)
						aux = (dx**2 + dy**2)
						syn_sen_v[i, j, h, k] =  self.L_v_ex * mt.exp( -aux / 2*self.sigma_v_ex**2 ) - self.L_v_in * mt.exp( -aux / 2*self.sigma_v_in**2 )
						syn_sen_a[i, j, h, k] =  self.L_a_ex * mt.exp( -aux / 2*self.sigma_a_ex**2 ) - self.L_a_in * mt.exp( -aux / 2*self.sigma_a_in**2 )
						syn_sen_m[i, j, h, k] =  self.L_m_ex * mt.exp( -aux / 2*self.sigma_m_ex**2 ) - self.L_m_in * mt.exp( -aux / 2*self.sigma_m_in**2 )

		#This represents l_s_ij
		self.neighbor_input_v = K.sum(K.sum(self.neuron_activity_v * K.variable(value=syn_sen_v, dtype='float32', name='synaptic_strength_v'), axis=-1), axis=-1)
		self.neighbor_input_a = K.sum(K.sum(self.neuron_activity_a * K.variable(value=syn_sen_a, dtype='float32', name='synaptic_strength_a'), axis=-1), axis=-1)

		#This represents f_s_ij
		self.colliculus_input_v = self.Fv * self.neuron_activity_m
		self.colliculus_input_a = self.Fa * self.neuron_activity_m

		'''
		Multisensory Component
		'''
		#Update auditory and visual neuron activity z_s_ij
		self.neuron_activity_v = self.neuron_activity_v * (self.tau_v - 1)/self.tau_v + (1/self.tau_v) * A.sigmoid( (self.external_input_v + self.neighbor_input_v - self.upsilon_v + self.colliculus_input_v) * self.p_v)
		self.neuron_activity_a = self.neuron_activity_a * (self.tau_a - 1)/self.tau_a + (1/self.tau_a) * A.sigmoid( (self.external_input_a + self.neighbor_input_a - self.upsilon_a + self.colliculus_input_a) * self.p_a)

		#This represents l_m_ij
		self.neighbor_input_m = K.sum(K.sum(self.neuron_activity_m * K.variable(value=syn_sen_m, dtype='float32', name='synaptic_strength_v'), axis=-1), axis=-1)
		
		#This represents the multisensorial neuron activity z_m_ij
		self.neuron_activity_m = self.neuron_activity_m * (self.tau_m - 1)/self.tau_m + (1/self.tau_m) * A.sigmoid( (self.k_v * self.neuron_activity_v + self.k_a * self.neuron_activity_a + self.neighbor_input_m - self.upsilon_m) * self.p_m)

		return self.neuron_activity_m

	def compute_output_shape(self, input_shape):
		return (input_shape[0][0], self.output_dim, self.output_dim)
	
	def _distance_helper(self, i, j, h, k):
		if(abs(i-h) <= self.N / 2):
			dx = abs(i-h)
		else:
			dx = self.N - abs(i-h)

		if(abs(j-k) <= self.M / 2):
			dy = abs(j-k)		
		else:
			dy = self.M - abs(j-k)

		return [dx, dy]

'''================================================================================================================'''
'''  The following is a mere dummy demonstration of how the layer can be called and associated to others in Keras  '''
'''     It is purposefully meant to do nothing. After integration in your model you should be able to call 'fit'   '''
'''================================================================================================================'''


vid_size = 1024	#Completely arbitrary, it depends on your own network and application
aud_size = 2048	#Completely arbitrary, it depends on your own network and application

video = Input((vid_size,))
audio = Input((aud_size,))
unimodal_embedding_v = Dense(64, activation='relu', kernel_initializer='he_uniform')(video)
unimodal_embedding_a = Dense(64, activation='relu', kernel_initializer='he_uniform')(audio)
multimodal_embeddings = SCF(17)([video, audio])
flat = Flatten()(multimodal_embeddings)
output = Dense(1, activation='sigmoid')(flat)
model = Model(inputs=[video, audio], outputs=output)

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


