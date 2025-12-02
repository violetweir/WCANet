import glob
import importlib

from util.registry import Registry
TRAINER = Registry('Trainer')

files = glob.glob('trainer/[!_]*.py')
for file in files:
	if "\\" in file:
		model_lib = importlib.import_module(file.split('.')[0].replace('\\', '.'))
	else:
		model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))



def get_trainer(cfg):
	return TRAINER.get_module(cfg.trainer.name)(cfg)
