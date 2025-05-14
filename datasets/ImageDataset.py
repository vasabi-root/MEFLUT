import os
from pathlib import Path
import functools
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from PIL import Image
from math import ceil
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
	"""Checks if a file is an allowed extension.
	Args:
		filename (string): path to a file
		extensions (iterable of strings): extensions to consider (lowercase)
	Returns:
		bool: True if the filename ends with one of given extensions
	"""
	filename_lower = filename.lower()
	return any(filename_lower.endswith(ext) for ext in extensions)


def image_seq_loader(img_seq_dir):
	img_seq_dir = os.path.expanduser(img_seq_dir)

	img_seq = []
	for root, _, fnames in sorted(os.walk(img_seq_dir)):
		for fname in sorted(fnames):
			if has_file_allowed_extension(fname, IMG_EXTENSIONS):
				image_name = os.path.join(root, fname)
				img_seq.append(Image.open(image_name))

	return img_seq

def get_default_img_seq_loader():
	return functools.partial(image_seq_loader)


class ImageSeqDataset(Dataset):
	def __init__(self, 
				hr_img_seq_dir,
				n_frames,
			  	csv_file=None,
				hr_transform=None,
				lr_transform=None,
				get_loader=get_default_img_seq_loader):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			hr_img_seq_dir (string): Directory with all the high resolution image sequences.
			transform (callable, optional): transform to be applied on a sample.
		"""
		self.hr_root = Path(hr_img_seq_dir)
		self.csv_file = csv_file
		self.n_frames = n_frames
		self.hr_transform = hr_transform
		self.lr_transform = lr_transform

		self.seqs = self.get_seqs()
		self.filter_seqs_by_n_frames()
		self.loader = get_loader()

	def get_seqs(self):
		if self.csv_file:
			return pd.read_csv(self.csv_file, header=None)[0]
		return [name for name in os.listdir(self.hr_root) if os.path.isdir(self.hr_root / name)]

	def filter_seqs_by_n_frames(self):
		seqs_to_remove = set()
		for i, seq in enumerate(self.seqs):
			seq_len = len(os.listdir(self.hr_root / seq))
			if seq_len < self.n_frames:
				seqs_to_remove.add(seq)
				print(f'Sequence dir {seq!r} contains too little frames to process ({seq_len} < {self.n_frames}). It will be ignored.')
		self.seqs = [seq for seq in self.seqs if seq not in seqs_to_remove]
	
 
	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			samples: a Tensor that represents a video segment.
		"""
		hr_seq_dir = os.path.join(self.hr_root, str(self.seqs[index]))
		I = self.loader(hr_seq_dir)
		
		if self.hr_transform is not None:
			I_hr = self.hr_transform(I)
		if self.lr_transform is not None:
			I_lr = self.lr_transform(I)

		I_hr = sorted(I_hr, key=lambda x: x.mean())
		I_lr = sorted(I_lr, key=lambda x: x.mean())

		I_hr = self._remove_extra_exposures(I_hr)
		I_lr = self._remove_extra_exposures(I_lr)

		I_hr = torch.stack(I_hr, 0).contiguous()
		I_lr = torch.stack(I_lr, 0).contiguous()

		sample = {'I_hr': I_hr, 'I_lr': I_lr, 'case':str(self.seqs[index])}
		return sample
	
	def _remove_extra_exposures(self, bunch):
		stride = ceil(len(bunch) / self.n_frames)
		images = [bunch[i] for i in range(1, (self.n_frames-1)*stride, stride)]
		images.append(bunch[-1])
		return images

	def __len__(self):
		return len(self.seqs)

	@staticmethod
	def _reorderBylum(seq):
		I = torch.sum(torch.sum(torch.sum(seq, 1), 1), 1)
		_, index = torch.sort(I)
		result = seq[index, :]
		return result
