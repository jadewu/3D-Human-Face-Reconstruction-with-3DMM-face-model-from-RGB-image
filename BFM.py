import torch
from torch import nn
import numpy as np
from scipy.io import loadmat, savemat
from array import array

class BFM():
	"""
	This is a numpy implementation of BFM model
		for visualization purpose, not used in the DNN model
	"""
	def __init__(self):
		model_path = './BFM/BFM_model_front.mat'
		model = loadmat(model_path)
		self.meanshape = model['meanshape'].T # mean face shape 
		self.idBase = model['idBase'] # identity basis
		self.exBase = model['exBase'] # expression basis
		self.meantex = model['meantex'].T # mean face texture
		self.texBase = model['texBase'] # texture basis
		self.point_buf = model['point_buf'] # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
		self.tri = model['tri'] # vertex index for each triangle face, starts from 1
		self.keypoints = np.squeeze(model['keypoints']).astype(np.int32) - 1 # 68 face landmark index, starts from 0

class BFM_torch(nn.Module):
	"""
	This is a torch implementation of the BFM model
	Used in the DNN model, comes with gradient support
	"""
	def __init__(self):
		super(BFM_torch, self).__init__()
		model_path = './BFM/BFM_model_front.mat'
		model = loadmat(model_path)
		# [107127, 1]
		self.register_buffer("meanshape", torch.tensor(model['meanshape'].T, dtype=torch.float32))
		# [107127, 80]
		self.register_buffer("idBase", torch.tensor(model['idBase'], dtype=torch.float32))
		# [107127, 64]
		self.register_buffer("exBase", torch.tensor(model['exBase'], dtype=torch.float32))
		# [107127, 1]
		self.register_buffer("meantex", torch.tensor(model['meantex'].T, dtype=torch.float32))
		# [107121, 80]
		self.register_buffer('texBase', torch.tensor(model['texBase'], dtype=torch.float32))
		# [70789, 3]
		self.register_buffer('tri', torch.tensor(model['tri'], dtype=torch.int32))
		# [35709, 8] Max is 70789;
		self.register_buffer('point_buf', torch.tensor(model['point_buf'], dtype=torch.int32))
		# [68]
		self.register_buffer('keypoints', torch.tensor(np.squeeze(model['keypoints']).astype(np.int32) - 1, dtype=torch.int32))

	def get_shape(self, id_param, ex_param):
		"""
		Perform shape assembly from index parameter and expression parameter
		id_param: [bs, 80]
		ex_param: [bs, 64]
		return: [bs, 107127, 1]
		"""
		assert id_param.shape[0] == ex_param.shape[0]
		bs = id_param.shape[0]

		id_base = self.idBase[None,:,:].expand(bs,-1,-1)
		ex_base = self.exBase[None,:,:].expand(bs,-1,-1)
		
		face_shape = self.meanshape+torch.bmm(id_base,id_param[:,:,None])+torch.bmm(ex_base,ex_param[:,:,None])
		face_shape = face_shape.reshape(bs,-1, 3)
		face_shape = face_shape - torch.mean(self.meanshape[None,:,:].reshape(1,-1,3), dim=1, keepdim=True)        
		return face_shape

	def get_texture(self, tex_param):
		"""
		Perform texture assembly from texture parameter
		tex_param: [bs, 80]
		return: [bs, 107127, 1]
		"""
		bs = tex_param.shape[0]
		tex_base = self.texBase[None,:,:].expand(bs,-1,-1)
		
		return self.meantex+torch.bmm(tex_base,tex_param[:,:,None])

	def compute_rotation_matrix(self, rotate_param):
		"""
		Perform rotation based on the batch rotation parameter
		rotate_param: [bs, 3]
		return: [bs, 3, 3]
		"""
		pitch, yaw, roll = rotate_param[:,0], rotate_param[:,1], rotate_param[:,2]
		bs = rotate_param.shape[0]
		device = rotate_param.device

		pitch_matrix = torch.eye(3, device=device)[None,:,:].expand(bs,-1,-1).clone()
		yaw_matrix = torch.eye(3, device=device)[None,:,:].expand(bs,-1,-1).clone()
		roll_matrix  = torch.eye(3, device=device)[None,:,:].expand(bs,-1,-1).clone()

		pitch_matrix[:,1,1] = torch.cos(pitch)
		pitch_matrix[:,2,2] = torch.cos(pitch)
		pitch_matrix[:,1,2] = -torch.sin(pitch)
		pitch_matrix[:,2,1] = torch.sin(pitch)

		yaw_matrix[:,0,0] = torch.cos(yaw)
		yaw_matrix[:,2,2] = torch.cos(yaw)
		yaw_matrix[:,0,2] = torch.sin(yaw)
		yaw_matrix[:,2,0] = -torch.sin(yaw)

		roll_matrix[:,0,0] = torch.cos(roll)
		roll_matrix[:,1,1] = torch.cos(roll)
		roll_matrix[:,0,1] = -torch.sin(roll)
		roll_matrix[:,1,0] = torch.sin(roll)

		return torch.bmm(torch.bmm(roll_matrix, yaw_matrix), pitch_matrix).permute(0,2,1)